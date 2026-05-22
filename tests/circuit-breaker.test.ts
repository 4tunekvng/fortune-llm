import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_TRIP_DURATION_MS,
  getCircuitState,
  isQuotaError,
  resolveTripDurationMs,
  tripCircuit,
} from "../src/circuit-breaker.js";

class FakeKv {
  private store = new Map<string, { value: string; expiresAt: number | null }>();
  constructor(private now: () => number = () => Date.now()) {}

  async get(key: string): Promise<string | null> {
    const entry = this.store.get(key);
    if (!entry) return null;
    if (entry.expiresAt !== null && entry.expiresAt <= this.now()) {
      this.store.delete(key);
      return null;
    }
    return entry.value;
  }

  async put(
    key: string,
    value: string,
    opts?: { expirationTtl?: number },
  ): Promise<void> {
    const expiresAt =
      opts?.expirationTtl !== undefined ? this.now() + opts.expirationTtl * 1000 : null;
    this.store.set(key, { value, expiresAt });
  }

  // Test helpers — not part of the KVNamespace surface
  raw(key: string): string | undefined {
    return this.store.get(key)?.value;
  }
  size(): number {
    return this.store.size;
  }
}

const fakeKv = (now?: () => number) =>
  new FakeKv(now) as unknown as KVNamespace & { raw(k: string): string | undefined; size(): number };

describe("isQuotaError", () => {
  it("true-positives quota / rate-limit signals", () => {
    expect(isQuotaError(new Error("RESOURCE_EXHAUSTED: quota exceeded"))).toBe(true);
    expect(isQuotaError(new Error("429 Too Many Requests"))).toBe(true);
    expect(isQuotaError(new Error("rate limit exceeded"))).toBe(true);
    expect(isQuotaError(new Error("Rate-Limit: 0 remaining"))).toBe(true);
    expect(isQuotaError(new Error("daily limit exceeded"))).toBe(true);
    expect(isQuotaError(new Error("neurons exhausted for today"))).toBe(true);
    expect(isQuotaError("HTTP 429: Too Many Requests")).toBe(true);
  });

  it("true-negatives transient / non-quota errors", () => {
    expect(isQuotaError(new Error("ECONNRESET"))).toBe(false);
    expect(isQuotaError(new Error("fetch failed: timeout"))).toBe(false);
    expect(isQuotaError(new Error("500 Internal Server Error"))).toBe(false);
    expect(isQuotaError(new Error("max_tokens exceeded"))).toBe(false); // "exceeded" alone is not enough
    expect(isQuotaError(new Error("validation failed"))).toBe(false);
    expect(isQuotaError(null)).toBe(false);
    expect(isQuotaError(undefined)).toBe(false);
    expect(isQuotaError("")).toBe(false);
  });

  it("does not false-positive on 429 substring inside arbitrary numbers", () => {
    // "1429" or "4290" should not match
    expect(isQuotaError(new Error("processed 1429 tokens"))).toBe(false);
    expect(isQuotaError(new Error("response was 4290 bytes"))).toBe(false);
  });
});

describe("getCircuitState", () => {
  it("returns closed when KV is undefined (dev / unbound)", async () => {
    const state = await getCircuitState("workers-ai", undefined);
    expect(state).toEqual({ open: false });
  });

  it("returns closed when no record exists", async () => {
    const kv = fakeKv();
    expect(await getCircuitState("gemini", kv)).toEqual({ open: false });
  });

  it("returns open with until/reason when an unexpired record exists", async () => {
    const kv = fakeKv();
    await tripCircuit("workers-ai", kv, 60_000, "RESOURCE_EXHAUSTED");
    const state = await getCircuitState("workers-ai", kv);
    expect(state.open).toBe(true);
    expect(state.until).toBeGreaterThan(Date.now());
    expect(state.reason).toContain("RESOURCE_EXHAUSTED");
  });

  it("returns closed when the stored record is past `until` (even if not yet purged)", async () => {
    const kv = fakeKv();
    const past = Date.now() - 1000;
    await kv.put(
      "circuit:gemini",
      JSON.stringify({ openedAt: past - 60_000, until: past, reason: "expired" }),
    );
    expect(await getCircuitState("gemini", kv)).toEqual({ open: false });
  });

  it("returns closed when the stored JSON is malformed", async () => {
    const kv = fakeKv();
    await kv.put("circuit:workers-ai", "not json");
    expect(await getCircuitState("workers-ai", kv)).toEqual({ open: false });
  });

  it("returns closed when the stored record is missing `until`", async () => {
    const kv = fakeKv();
    await kv.put("circuit:workers-ai", JSON.stringify({ openedAt: Date.now(), reason: "x" }));
    expect(await getCircuitState("workers-ai", kv)).toEqual({ open: false });
  });

  it("degrades to closed when KV.get throws (infra blip — don't block traffic)", async () => {
    const kv: KVNamespace = {
      get: async () => {
        throw new Error("KV upstream timeout");
      },
    } as unknown as KVNamespace;
    expect(await getCircuitState("gemini", kv)).toEqual({ open: false });
  });
});

describe("tripCircuit", () => {
  it("no-ops when KV is undefined", async () => {
    await expect(tripCircuit("workers-ai", undefined, 60_000, "reason")).resolves.toBeUndefined();
  });

  it("writes a record with openedAt/until/reason", async () => {
    const kv = fakeKv();
    const before = Date.now();
    await tripCircuit("workers-ai", kv, 3600_000, "Gemini RESOURCE_EXHAUSTED");
    const stored = (kv as unknown as FakeKv).raw("circuit:workers-ai");
    expect(stored).toBeTruthy();
    const parsed = JSON.parse(stored!);
    expect(parsed.openedAt).toBeGreaterThanOrEqual(before);
    expect(parsed.until).toBe(parsed.openedAt + 3600_000);
    expect(parsed.reason).toBe("Gemini RESOURCE_EXHAUSTED");
  });

  it("re-tripping replaces the record (extends the window)", async () => {
    const kv = fakeKv();
    await tripCircuit("gemini", kv, 60_000, "first");
    const first = JSON.parse((kv as unknown as FakeKv).raw("circuit:gemini")!);
    await new Promise((r) => setTimeout(r, 5)); // let clock advance
    await tripCircuit("gemini", kv, 120_000, "second");
    const second = JSON.parse((kv as unknown as FakeKv).raw("circuit:gemini")!);
    expect(second.openedAt).toBeGreaterThan(first.openedAt);
    expect(second.until).toBeGreaterThan(first.until);
    expect(second.reason).toBe("second");
  });

  it("bounds the reason to 500 chars (keeps KV value small)", async () => {
    const kv = fakeKv();
    const huge = "x".repeat(5000);
    await tripCircuit("workers-ai", kv, 60_000, huge);
    const parsed = JSON.parse((kv as unknown as FakeKv).raw("circuit:workers-ai")!);
    expect(parsed.reason.length).toBe(500);
  });

  it("silently swallows KV.put failures (degrades to today's behavior)", async () => {
    const kv: KVNamespace = {
      put: async () => {
        throw new Error("KV write failed");
      },
    } as unknown as KVNamespace;
    await expect(tripCircuit("gemini", kv, 60_000, "reason")).resolves.toBeUndefined();
  });
});

describe("resolveTripDurationMs", () => {
  it("returns the default when env var is unset", () => {
    expect(resolveTripDurationMs(undefined)).toBe(DEFAULT_TRIP_DURATION_MS);
  });

  it("returns the parsed value when valid", () => {
    expect(resolveTripDurationMs("1800000")).toBe(1800000);
  });

  it("falls back to default on non-numeric / non-positive input", () => {
    expect(resolveTripDurationMs("garbage")).toBe(DEFAULT_TRIP_DURATION_MS);
    expect(resolveTripDurationMs("0")).toBe(DEFAULT_TRIP_DURATION_MS);
    expect(resolveTripDurationMs("-100")).toBe(DEFAULT_TRIP_DURATION_MS);
  });

  it("caps at 24h to prevent a typo from locking a tier for a week", () => {
    const week = 7 * 24 * 60 * 60 * 1000;
    expect(resolveTripDurationMs(String(week))).toBe(24 * 60 * 60 * 1000);
  });
});

describe("end-to-end: trip then read", () => {
  let kv: KVNamespace;
  beforeEach(() => {
    kv = fakeKv();
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-22T12:00:00Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("read returns open immediately after trip, closed after `until` passes", async () => {
    await tripCircuit("workers-ai", kv, 60_000, "RESOURCE_EXHAUSTED");

    const opened = await getCircuitState("workers-ai", kv);
    expect(opened.open).toBe(true);
    expect(opened.until).toBe(Date.parse("2026-05-22T12:01:00Z"));

    vi.setSystemTime(new Date("2026-05-22T12:00:59Z"));
    expect((await getCircuitState("workers-ai", kv)).open).toBe(true);

    vi.setSystemTime(new Date("2026-05-22T12:01:01Z"));
    expect((await getCircuitState("workers-ai", kv)).open).toBe(false);
  });
});
