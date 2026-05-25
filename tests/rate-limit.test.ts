import { describe, it, expect } from "vitest";
import {
  checkRateLimit,
  getClientIp,
  getRateLimitScope,
  resolveConsumerRateLimit,
  resolveRateLimitPerMin,
  DEFAULT_RATE_LIMIT_PER_MIN,
} from "../src/rate-limit.js";

const makeKv = (initial: Map<string, string> = new Map()): KVNamespace => {
  const store = new Map(initial);
  return {
    async get(k: string) {
      return store.get(k) ?? null;
    },
    async put(k: string, v: string) {
      store.set(k, v);
    },
    async delete(k: string) {
      store.delete(k);
    },
    async list() {
      return { keys: [], list_complete: true, cacheStatus: null };
    },
    async getWithMetadata() {
      return { value: null, metadata: null, cacheStatus: null };
    },
  } as unknown as KVNamespace;
};

describe("getClientIp", () => {
  it("prefers cf-connecting-ip", () => {
    const req = new Request("https://x", {
      headers: { "cf-connecting-ip": "1.2.3.4", "x-forwarded-for": "5.6.7.8" },
    });
    expect(getClientIp(req)).toBe("1.2.3.4");
  });

  it("falls back to x-forwarded-for first hop", () => {
    const req = new Request("https://x", {
      headers: { "x-forwarded-for": "1.2.3.4, 5.6.7.8" },
    });
    expect(getClientIp(req)).toBe("1.2.3.4");
  });

  it("returns 'unknown' when no IP header is present", () => {
    const req = new Request("https://x");
    expect(getClientIp(req)).toBe("unknown");
  });
});

describe("checkRateLimit", () => {
  it("allows requests when KV is undefined (fail-open)", async () => {
    const d = await checkRateLimit("1.2.3.4", 10, undefined);
    expect(d.allowed).toBe(true);
  });

  it("allows requests when limit is 0 (disabled)", async () => {
    const d = await checkRateLimit("1.2.3.4", 0, makeKv());
    expect(d.allowed).toBe(true);
  });

  it("increments the counter under the limit", async () => {
    const kv = makeKv();
    const d1 = await checkRateLimit("1.2.3.4", 5, kv);
    expect(d1.allowed).toBe(true);
    expect(d1.count).toBe(1);
    const d2 = await checkRateLimit("1.2.3.4", 5, kv);
    expect(d2.allowed).toBe(true);
    expect(d2.count).toBe(2);
  });

  it("rejects when the counter reaches the limit", async () => {
    const kv = makeKv();
    for (let i = 0; i < 3; i++) {
      await checkRateLimit("1.2.3.4", 3, kv);
    }
    const d = await checkRateLimit("1.2.3.4", 3, kv);
    expect(d.allowed).toBe(false);
    expect(d.retryAfterSeconds).toBeGreaterThan(0);
  });

  it("counters are isolated per IP", async () => {
    const kv = makeKv();
    for (let i = 0; i < 3; i++) {
      await checkRateLimit("1.2.3.4", 3, kv);
    }
    const blocked = await checkRateLimit("1.2.3.4", 3, kv);
    expect(blocked.allowed).toBe(false);
    const otherIp = await checkRateLimit("9.9.9.9", 3, kv);
    expect(otherIp.allowed).toBe(true);
  });
});

describe("getRateLimitScope", () => {
  it("returns consumer scope when valid x-fortune-consumer header present", () => {
    const req = new Request("https://x", { headers: { "x-fortune-consumer": "lena" } });
    expect(getRateLimitScope(req)).toEqual({ scope: "lena", kind: "consumer" });
  });

  it("lowercases consumer names for consistency", () => {
    const req = new Request("https://x", { headers: { "x-fortune-consumer": "LENA" } });
    expect(getRateLimitScope(req)).toEqual({ scope: "lena", kind: "consumer" });
  });

  it("falls back to IP scope when the header is missing", () => {
    const req = new Request("https://x", { headers: { "cf-connecting-ip": "1.2.3.4" } });
    expect(getRateLimitScope(req)).toEqual({ scope: "1.2.3.4", kind: "ip" });
  });

  it("falls back to IP scope when the consumer header has invalid characters", () => {
    const req = new Request("https://x", {
      headers: { "x-fortune-consumer": "bad name with spaces!", "cf-connecting-ip": "1.2.3.4" },
    });
    expect(getRateLimitScope(req)).toEqual({ scope: "1.2.3.4", kind: "ip" });
  });

  it("falls back to IP scope when the consumer header is too long", () => {
    const req = new Request("https://x", {
      headers: { "x-fortune-consumer": "x".repeat(33), "cf-connecting-ip": "1.2.3.4" },
    });
    expect(getRateLimitScope(req)).toEqual({ scope: "1.2.3.4", kind: "ip" });
  });
});

describe("resolveConsumerRateLimit", () => {
  it("returns the global default when no override exists", () => {
    expect(resolveConsumerRateLimit("knox", 200, {})).toBe(200);
  });

  it("honors RATE_LIMIT_PER_MIN_<CONSUMER> overrides", () => {
    expect(
      resolveConsumerRateLimit("lena", 200, { RATE_LIMIT_PER_MIN_LENA: "500" }),
    ).toBe(500);
  });

  it("normalizes consumer names with dashes to underscores for env lookup", () => {
    expect(
      resolveConsumerRateLimit("network-agent", 200, { RATE_LIMIT_PER_MIN_NETWORK_AGENT: "1000" }),
    ).toBe(1000);
  });

  it("rejects malformed consumer names (returns default)", () => {
    expect(
      resolveConsumerRateLimit("bad name", 200, { "RATE_LIMIT_PER_MIN_BAD NAME": "500" }),
    ).toBe(200);
  });
});

describe("resolveRateLimitPerMin", () => {
  it("returns the default when env value is undefined", () => {
    expect(resolveRateLimitPerMin(undefined)).toBe(DEFAULT_RATE_LIMIT_PER_MIN);
  });

  it("returns 0 (disabled) when env value is '0'", () => {
    expect(resolveRateLimitPerMin("0")).toBe(0);
  });

  it("accepts a valid value", () => {
    expect(resolveRateLimitPerMin("500")).toBe(500);
  });

  it("falls back to default for non-numeric / negative", () => {
    expect(resolveRateLimitPerMin("abc")).toBe(DEFAULT_RATE_LIMIT_PER_MIN);
    expect(resolveRateLimitPerMin("-10")).toBe(DEFAULT_RATE_LIMIT_PER_MIN);
  });

  it("caps at 100k", () => {
    expect(resolveRateLimitPerMin("999999999")).toBe(100_000);
  });
});
