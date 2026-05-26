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
 * Schema (SQLite, in-DO):
 *   daily_totals(date, metric, count)
 *     gateway-wide rollups: requests, cache_hits, cache_misses,
 *     rate_limited, errors.
 *   per_tier(date, tier, ok, fail)
 *     gateway-wide per-tier rollups for each upstream.
 *   per_consumer(date, consumer, metric, count)
 *     same rollups but split by `x-fortune-consumer` header (or
 *     "unknown" when absent). Lets us see WHICH apps drive WHICH
 *     traffic patterns.
 *   per_consumer_tier(date, consumer, tier, ok, fail)
 *     per-consumer × per-tier breakdown — the table that answers
 *     "which app is responsible for the Anthropic spend today?"
 *
 * All increments use SQL UPSERT so no read-modify-write — concurrent
 * writes can't lose increments.
 *
 * Lifecycle: the Worker calls `recordEvents(events, consumer)` from
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

export interface ConsumerStats {
  totals: {
    requests: number;
    cache_hits: number;
    cache_misses: number;
    rate_limited: number;
    errors: number;
  };
  per_tier: Record<string, { ok: number; fail: number }>;
}

export interface DailyStats {
  date: string;
  totals: ConsumerStats["totals"];
  per_tier: ConsumerStats["per_tier"];
  /** Per-consumer breakdown. "unknown" is the bucket for unscoped traffic. */
  per_consumer: Record<string, ConsumerStats>;
}

const TOTAL_METRICS = ["requests", "cache_hits", "cache_misses", "rate_limited", "errors"] as const;
type TotalMetric = (typeof TOTAL_METRICS)[number];

/** Match the rate-limit module's CONSUMER_RE so the two stay in sync. */
const CONSUMER_RE = /^[a-z0-9_-]{1,32}$/i;
const UNKNOWN = "unknown";

/** Normalize a consumer name. Anything that doesn't match goes to "unknown". */
function normalizeConsumer(raw: string | null | undefined): string {
  if (!raw) return UNKNOWN;
  const trimmed = raw.trim().toLowerCase();
  return CONSUMER_RE.test(trimmed) ? trimmed : UNKNOWN;
}

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

function emptyTotals(): ConsumerStats["totals"] {
  return { requests: 0, cache_hits: 0, cache_misses: 0, rate_limited: 0, errors: 0 };
}

export class StatsDO extends DurableObject {
  constructor(ctx: DurableObjectState, env: unknown) {
    super(ctx, env as Record<string, unknown>);
    // Schema lives in the constructor so it's there before any RPC.
    // blockConcurrencyWhile is exactly right for one-time setup —
    // never use it inside request handlers. All CREATE statements
    // use IF NOT EXISTS so adding new tables to a deployed schema
    // is a no-op for existing DO instances.
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
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS per_consumer (
          date TEXT NOT NULL,
          consumer TEXT NOT NULL,
          metric TEXT NOT NULL,
          count INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (date, consumer, metric)
        )
      `);
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS per_consumer_tier (
          date TEXT NOT NULL,
          consumer TEXT NOT NULL,
          tier TEXT NOT NULL,
          ok INTEGER NOT NULL DEFAULT 0,
          fail INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (date, consumer, tier)
        )
      `);
    });
  }

  /**
   * Atomically increment counters for a batch of events from one
   * gateway request. Each metric event lands in both `daily_totals`
   * (gateway-wide) AND `per_consumer` (scoped). Each tier event lands
   * in both `per_tier` AND `per_consumer_tier`.
   *
   * `consumer` should be the value of the `x-fortune-consumer` header
   * if the caller sent one; pass null/undefined for unscoped requests
   * (they bucket as "unknown"). The DO sanitizes/normalizes the name
   * defensively in case callers send something invalid.
   *
   * Same-batch events are tallied in-memory first, then flushed in a
   * single transactionSync so concurrent requests don't fight for the
   * SQLite write lock.
   */
  async recordEvents(events: StatsEvent[], consumer?: string | null): Promise<void> {
    if (events.length === 0) return;
    const date = todayUtcDate();
    const consumerName = normalizeConsumer(consumer);

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
        const count = n as number;
        // Gateway-wide rollup.
        this.ctx.storage.sql.exec(
          `INSERT INTO daily_totals (date, metric, count) VALUES (?, ?, ?)
           ON CONFLICT(date, metric) DO UPDATE SET count = count + excluded.count`,
          date,
          metric,
          count,
        );
        // Per-consumer rollup.
        this.ctx.storage.sql.exec(
          `INSERT INTO per_consumer (date, consumer, metric, count) VALUES (?, ?, ?, ?)
           ON CONFLICT(date, consumer, metric) DO UPDATE SET count = count + excluded.count`,
          date,
          consumerName,
          metric,
          count,
        );
      }
      for (const [tier, { ok, fail }] of Object.entries(tierIncrements)) {
        // Gateway-wide per-tier.
        this.ctx.storage.sql.exec(
          `INSERT INTO per_tier (date, tier, ok, fail) VALUES (?, ?, ?, ?)
           ON CONFLICT(date, tier) DO UPDATE SET ok = ok + excluded.ok, fail = fail + excluded.fail`,
          date,
          tier,
          ok,
          fail,
        );
        // Per-consumer per-tier — the breakdown that answers
        // "which app is driving which tier's load".
        this.ctx.storage.sql.exec(
          `INSERT INTO per_consumer_tier (date, consumer, tier, ok, fail) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(date, consumer, tier) DO UPDATE SET ok = ok + excluded.ok, fail = fail + excluded.fail`,
          date,
          consumerName,
          tier,
          ok,
          fail,
        );
      }
    });
  }

  /**
   * Return the current day's stats snapshot, including a per-consumer
   * breakdown. Strongly consistent against all prior recordEvents()
   * calls because both go through the same DO actor.
   */
  async getStats(): Promise<DailyStats> {
    const date = todayUtcDate();
    const totals = emptyTotals();
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

    const per_tier: ConsumerStats["per_tier"] = {};
    const tierRows = this.ctx.storage.sql
      .exec<{ tier: string; ok: number; fail: number }>(
        "SELECT tier, ok, fail FROM per_tier WHERE date = ?",
        date,
      )
      .toArray();
    for (const r of tierRows) {
      per_tier[r.tier] = { ok: r.ok, fail: r.fail };
    }

    // Build per-consumer breakdown: start with empty stats for every
    // consumer we've seen today, then populate from both tables.
    const per_consumer: Record<string, ConsumerStats> = {};
    const ensure = (name: string): ConsumerStats => {
      let slot = per_consumer[name];
      if (!slot) {
        slot = { totals: emptyTotals(), per_tier: {} };
        per_consumer[name] = slot;
      }
      return slot;
    };

    const consumerTotalRows = this.ctx.storage.sql
      .exec<{ consumer: string; metric: string; count: number }>(
        "SELECT consumer, metric, count FROM per_consumer WHERE date = ?",
        date,
      )
      .toArray();
    for (const r of consumerTotalRows) {
      const slot = ensure(r.consumer);
      if (r.metric in slot.totals) {
        (slot.totals as Record<string, number>)[r.metric] = r.count;
      }
    }

    const consumerTierRows = this.ctx.storage.sql
      .exec<{ consumer: string; tier: string; ok: number; fail: number }>(
        "SELECT consumer, tier, ok, fail FROM per_consumer_tier WHERE date = ?",
        date,
      )
      .toArray();
    for (const r of consumerTierRows) {
      const slot = ensure(r.consumer);
      slot.per_tier[r.tier] = { ok: r.ok, fail: r.fail };
    }

    return { date, totals, per_tier, per_consumer };
  }

  /**
   * Reset the entire stats history. Not exposed via the public
   * surface — useful for tests and operator-driven reset only.
   */
  async resetAll(): Promise<void> {
    this.ctx.storage.transactionSync(() => {
      this.ctx.storage.sql.exec("DELETE FROM daily_totals");
      this.ctx.storage.sql.exec("DELETE FROM per_tier");
      this.ctx.storage.sql.exec("DELETE FROM per_consumer");
      this.ctx.storage.sql.exec("DELETE FROM per_consumer_tier");
    });
  }
}
