/**
 * Durable Object that holds the gateway's daily stats counters.
 *
 * Why a DO when we already had KV-backed stats: KV's read-modify-write
 * counter pattern loses updates under concurrent traffic (two requests
 * read the same value, both increment, both write — net +1 instead of
 * +2). Plus KV reads cache at the edge for up to 60s so /stats lagged
 * traffic by a minute. A single DO serializes writes through one
 * actor so increments are strictly atomic and the read returns
 * up-to-the-millisecond state.
 *
 * Scale: one DO instance globally ("stats-singleton"). The whole
 * gateway's stats live there. Single-actor write throughput is plenty
 * for any traffic this gateway will ever see (low thousands of
 * requests/day; DOs handle millions). If we ever grow past that, shard
 * by date (one DO per day-bucket) — trivial change.
 *
 * Storage: SQLite tables for daily totals and per-tier counters. SQL
 * UPSERT does the atomic increment in a single statement so we never
 * have to read-then-write.
 *
 * Lifecycle: the Worker calls `recordEvents([...])` from
 * `ctx.waitUntil()` so the write doesn't add latency to the response.
 * `/stats` reads via `getStats()` — strongly consistent because all
 * writes serialize through the same DO.
 */

import { DurableObject } from "cloudflare:workers";
import type { BackendKind } from "./route.js";

export type StatsEvent =
  | { kind: "request" }
  | { kind: "cache_hit" }
  | { kind: "cache_miss" }
  | { kind: "rate_limited" }
  | { kind: "error" }
  | { kind: "tier_ok"; tier: BackendKind }
  | { kind: "tier_fail"; tier: BackendKind };

export interface DailyStats {
  date: string;
  totals: {
    requests: number;
    cache_hits: number;
    cache_misses: number;
    rate_limited: number;
    errors: number;
  };
  per_tier: Record<string, { ok: number; fail: number }>;
}

const TOTAL_METRICS = ["requests", "cache_hits", "cache_misses", "rate_limited", "errors"] as const;
type TotalMetric = (typeof TOTAL_METRICS)[number];

function todayUtcDate(): string {
  return new Date().toISOString().slice(0, 10);
}

function eventToMetric(e: StatsEvent): TotalMetric | null {
  switch (e.kind) {
    case "request":
      return "requests";
    case "cache_hit":
      return "cache_hits";
    case "cache_miss":
      return "cache_misses";
    case "rate_limited":
      return "rate_limited";
    case "error":
      return "errors";
    default:
      return null;
  }
}

export class StatsDO extends DurableObject {
  constructor(ctx: DurableObjectState, env: unknown) {
    super(ctx, env as Record<string, unknown>);
    // Schema lives in the constructor so it's there before any RPC.
    // blockConcurrencyWhile is exactly right for one-time setup —
    // never use it inside request handlers.
    ctx.blockConcurrencyWhile(async () => {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS daily_totals (
          date TEXT NOT NULL,
          metric TEXT NOT NULL,
          count INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (date, metric)
        )
      `);
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS per_tier (
          date TEXT NOT NULL,
          tier TEXT NOT NULL,
          ok INTEGER NOT NULL DEFAULT 0,
          fail INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (date, tier)
        )
      `);
    });
  }

  /**
   * Atomically increment counters for a batch of events from one
   * gateway request. Each event maps to either a `daily_totals` row
   * or a `per_tier` row; both use SQL UPSERT so no read-modify-write.
   *
   * Same-date events are tallied in-memory first, then flushed in a
   * single transaction so concurrent requests don't fight for the
   * SQLite write lock.
   */
  async recordEvents(events: StatsEvent[]): Promise<void> {
    if (events.length === 0) return;
    const date = todayUtcDate();

    // Aggregate this request's events before touching storage.
    const totalIncrements: Partial<Record<TotalMetric, number>> = {};
    const tierIncrements: Record<string, { ok: number; fail: number }> = {};
    for (const e of events) {
      const metric = eventToMetric(e);
      if (metric) {
        totalIncrements[metric] = (totalIncrements[metric] ?? 0) + 1;
        continue;
      }
      if (e.kind === "tier_ok" || e.kind === "tier_fail") {
        const slot = tierIncrements[e.tier] ?? { ok: 0, fail: 0 };
        if (e.kind === "tier_ok") slot.ok++;
        else slot.fail++;
        tierIncrements[e.tier] = slot;
      }
    }

    // Single transaction holds the write lock briefly while we batch
    // every UPSERT. SQLite gives us atomicity within the transaction.
    this.ctx.storage.transactionSync(() => {
      for (const [metric, n] of Object.entries(totalIncrements)) {
        this.ctx.storage.sql.exec(
          `INSERT INTO daily_totals (date, metric, count) VALUES (?, ?, ?)
           ON CONFLICT(date, metric) DO UPDATE SET count = count + excluded.count`,
          date,
          metric,
          n as number,
        );
      }
      for (const [tier, { ok, fail }] of Object.entries(tierIncrements)) {
        this.ctx.storage.sql.exec(
          `INSERT INTO per_tier (date, tier, ok, fail) VALUES (?, ?, ?, ?)
           ON CONFLICT(date, tier) DO UPDATE SET ok = ok + excluded.ok, fail = fail + excluded.fail`,
          date,
          tier,
          ok,
          fail,
        );
      }
    });
  }

  /**
   * Return the current day's stats snapshot. Strongly consistent
   * against all prior recordEvents() calls because both go through
   * the same DO actor.
   */
  async getStats(): Promise<DailyStats> {
    const date = todayUtcDate();
    const totals: DailyStats["totals"] = {
      requests: 0,
      cache_hits: 0,
      cache_misses: 0,
      rate_limited: 0,
      errors: 0,
    };
    const totalRows = this.ctx.storage.sql
      .exec<{ metric: string; count: number }>(
        "SELECT metric, count FROM daily_totals WHERE date = ?",
        date,
      )
      .toArray();
    for (const r of totalRows) {
      if (r.metric in totals) {
        (totals as Record<string, number>)[r.metric] = r.count;
      }
    }

    const per_tier: DailyStats["per_tier"] = {};
    const tierRows = this.ctx.storage.sql
      .exec<{ tier: string; ok: number; fail: number }>(
        "SELECT tier, ok, fail FROM per_tier WHERE date = ?",
        date,
      )
      .toArray();
    for (const r of tierRows) {
      per_tier[r.tier] = { ok: r.ok, fail: r.fail };
    }

    return { date, totals, per_tier };
  }

  /**
   * Reset the entire stats history. Not exposed via the public
   * surface — useful for tests and operator-driven reset only.
   */
  async resetAll(): Promise<void> {
    this.ctx.storage.transactionSync(() => {
      this.ctx.storage.sql.exec("DELETE FROM daily_totals");
      this.ctx.storage.sql.exec("DELETE FROM per_tier");
    });
  }
}
