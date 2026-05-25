import { describe, it, expect } from "vitest";
import { readStats, recordStats, type StatsEvent } from "../src/stats.js";

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

describe("readStats", () => {
  it("returns zeroed counters when KV is undefined", async () => {
    const s = await readStats(undefined);
    expect(s.totals.requests).toBe(0);
    expect(s.totals.cache_hits).toBe(0);
  });

  it("returns zeroed counters when no entry exists", async () => {
    const s = await readStats(makeKv());
    expect(s.totals.requests).toBe(0);
  });

  it("uses today's UTC date when not provided", async () => {
    const s = await readStats(makeKv());
    const today = new Date().toISOString().slice(0, 10);
    expect(s.date).toBe(today);
  });
});

describe("recordStats", () => {
  it("increments counters in a single batched write", async () => {
    const kv = makeKv();
    const events: StatsEvent[] = [
      { kind: "request" },
      { kind: "cache_miss" },
      { kind: "tier_ok", tier: "groq" },
    ];
    await recordStats(events, kv);
    const s = await readStats(kv);
    expect(s.totals.requests).toBe(1);
    expect(s.totals.cache_misses).toBe(1);
    expect(s.per_tier.groq).toEqual({ ok: 1, fail: 0 });
  });

  it("accumulates across multiple batches (same day)", async () => {
    const kv = makeKv();
    await recordStats([{ kind: "request" }, { kind: "cache_hit" }], kv);
    await recordStats([{ kind: "request" }, { kind: "cache_hit" }], kv);
    await recordStats([{ kind: "request" }, { kind: "cache_miss" }], kv);
    const s = await readStats(kv);
    expect(s.totals.requests).toBe(3);
    expect(s.totals.cache_hits).toBe(2);
    expect(s.totals.cache_misses).toBe(1);
  });

  it("tracks per-tier ok/fail counts", async () => {
    const kv = makeKv();
    await recordStats(
      [
        { kind: "tier_fail", tier: "groq" },
        { kind: "tier_ok", tier: "cerebras" },
        { kind: "tier_ok", tier: "cerebras" },
      ],
      kv,
    );
    const s = await readStats(kv);
    expect(s.per_tier.groq).toEqual({ ok: 0, fail: 1 });
    expect(s.per_tier.cerebras).toEqual({ ok: 2, fail: 0 });
  });

  it("counts rate_limited and error events", async () => {
    const kv = makeKv();
    await recordStats([{ kind: "rate_limited" }, { kind: "error" }], kv);
    const s = await readStats(kv);
    expect(s.totals.rate_limited).toBe(1);
    expect(s.totals.errors).toBe(1);
  });

  it("no-ops when KV is undefined", async () => {
    await expect(recordStats([{ kind: "request" }], undefined)).resolves.toBeUndefined();
  });

  it("no-ops on empty events array", async () => {
    const kv = makeKv();
    await recordStats([], kv);
    const s = await readStats(kv);
    expect(s.totals.requests).toBe(0);
  });

  it("recovers from malformed stored data without throwing", async () => {
    const date = new Date().toISOString().slice(0, 10);
    const kv = makeKv(new Map([[`stats:${date}`, "not-json"]]));
    await recordStats([{ kind: "request" }], kv);
    const s = await readStats(kv);
    expect(s.totals.requests).toBe(1);
  });
});
