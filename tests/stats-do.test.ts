import { describe, it, expect, vi } from "vitest";

vi.mock("cloudflare:workers", () => ({
  DurableObject: class {
    ctx: unknown;
    constructor(ctx: unknown) {
      this.ctx = ctx;
    }
  },
}));

import { StatsDO } from "../src/stats-do.js";

/** A storage stand-in whose reads answer with the given rows by table. */
function statsWith(rows: Record<string, unknown[]>): StatsDO {
  const exec = (sql: string) => ({
    toArray: () => {
      const table = /FROM (\w+) WHERE/.exec(sql)?.[1];
      return table ? (rows[table] ?? []) : [];
    },
  });
  const ctx = {
    blockConcurrencyWhile: (fn: () => unknown) => fn(),
    storage: { sql: { exec }, transactionSync: (fn: () => unknown) => fn() },
  };
  return new StatsDO(ctx as never, {});
}

describe("StatsDO.getStats", () => {
  it("counts an app named like a built-in under its own name", async () => {
    const stats = statsWith({
      per_consumer: [
        { consumer: "constructor", metric: "requests", count: 3 },
        { consumer: "__proto__", metric: "requests", count: 2 },
      ],
      per_consumer_tier: [
        { consumer: "toString", tier: "gemini", ok: 1, fail: 0 },
      ],
    });

    const { per_consumer } = await stats.getStats();

    expect(JSON.parse(JSON.stringify(per_consumer))).toEqual({
      constructor: {
        totals: { requests: 3, cache_hits: 0, cache_misses: 0, rate_limited: 0, errors: 0 },
        per_tier: {},
      },
      ["__proto__"]: {
        totals: { requests: 2, cache_hits: 0, cache_misses: 0, rate_limited: 0, errors: 0 },
        per_tier: {},
      },
      toString: {
        totals: { requests: 0, cache_hits: 0, cache_misses: 0, rate_limited: 0, errors: 0 },
        per_tier: { gemini: { ok: 1, fail: 0 } },
      },
    });
  });
});
