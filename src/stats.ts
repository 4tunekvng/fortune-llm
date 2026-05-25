/**
 * Lightweight per-day request/cache stats counter.
 *
 * Stored as a single KV entry per UTC day, keyed `stats:YYYY-MM-DD`,
 * holding a JSON object with counters. Single-key-per-day means the
 * increment path is a get-modify-put round-trip — eventually-consistent
 * and slightly lossy under heavy concurrency, but for observability
 * purposes that's fine.
 *
 * Why a single key per day instead of per-counter keys: KV is billed
 * per operation; one key/day caps the cost regardless of traffic, and
 * lets `/stats` return the day's state with one read.
 *
 * Why per-day and not absolute totals: a sliding window gives the
 * "is this currently working?" signal which is what we care about for
 * cost monitoring; absolute totals are interesting but less actionable.
 *
 * Counter shape:
 *   { date: "2026-05-25",
 *     totals: { requests, cache_hits, cache_misses, rate_limited, errors },
 *     per_tier: { groq: {ok, fail}, gemini: {ok, fail}, ... } }
 */

import type { BackendKind } from "./route.js";

const KEY_PREFIX = "stats:";
const KEY_TTL_SECONDS = 7 * 24 * 60 * 60; // keep daily entries for a week

export interface DailyStats {
  date: string;
  totals: {
    requests: number;
    cache_hits: number;
    cache_misses: number;
    rate_limited: number;
    errors: number;
  };
  per_tier: Partial<Record<BackendKind, { ok: number; fail: number }>>;
}

export type StatsEvent =
  | { kind: "request" }
  | { kind: "cache_hit" }
  | { kind: "cache_miss" }
  | { kind: "rate_limited" }
  | { kind: "error" }
  | { kind: "tier_ok"; tier: BackendKind }
  | { kind: "tier_fail"; tier: BackendKind };

function todayUtcDate(): string {
  // YYYY-MM-DD in UTC. Trim ISO at 'T'.
  const d = new Date();
  return d.toISOString().slice(0, 10);
}

function emptyStats(date: string): DailyStats {
  return {
    date,
    totals: { requests: 0, cache_hits: 0, cache_misses: 0, rate_limited: 0, errors: 0 },
    per_tier: {},
  };
}

/**
 * Read the current day's stats. Returns an empty record if KV is
 * unavailable or no entry exists yet today.
 */
export async function readStats(
  kv: KVNamespace | undefined,
  date: string = todayUtcDate(),
): Promise<DailyStats> {
  if (!kv) return emptyStats(date);
  const key = `${KEY_PREFIX}${date}`;
  let raw: string | null;
  try {
    raw = await kv.get(key);
  } catch {
    return emptyStats(date);
  }
  if (!raw) return emptyStats(date);
  try {
    const parsed = JSON.parse(raw) as DailyStats;
    // Defensive: if a previous deploy stored an older shape, normalize.
    return {
      date: parsed.date ?? date,
      totals: {
        requests: parsed.totals?.requests ?? 0,
        cache_hits: parsed.totals?.cache_hits ?? 0,
        cache_misses: parsed.totals?.cache_misses ?? 0,
        rate_limited: parsed.totals?.rate_limited ?? 0,
        errors: parsed.totals?.errors ?? 0,
      },
      per_tier: parsed.per_tier ?? {},
    };
  } catch {
    return emptyStats(date);
  }
}

/**
 * Apply a batch of events to today's stats record and write it back.
 * Best-effort — failures are swallowed. Callers should batch events
 * for a single request rather than calling this once per event to
 * minimize KV operations.
 */
export async function recordStats(
  events: StatsEvent[],
  kv: KVNamespace | undefined,
): Promise<void> {
  if (!kv || events.length === 0) return;
  const date = todayUtcDate();
  const key = `${KEY_PREFIX}${date}`;
  let current: DailyStats;
  try {
    current = await readStats(kv, date);
  } catch {
    current = emptyStats(date);
  }
  for (const e of events) {
    switch (e.kind) {
      case "request":
        current.totals.requests++;
        break;
      case "cache_hit":
        current.totals.cache_hits++;
        break;
      case "cache_miss":
        current.totals.cache_misses++;
        break;
      case "rate_limited":
        current.totals.rate_limited++;
        break;
      case "error":
        current.totals.errors++;
        break;
      case "tier_ok": {
        const slot = current.per_tier[e.tier] ?? { ok: 0, fail: 0 };
        slot.ok++;
        current.per_tier[e.tier] = slot;
        break;
      }
      case "tier_fail": {
        const slot = current.per_tier[e.tier] ?? { ok: 0, fail: 0 };
        slot.fail++;
        current.per_tier[e.tier] = slot;
        break;
      }
    }
  }
  try {
    await kv.put(key, JSON.stringify(current), { expirationTtl: KEY_TTL_SECONDS });
  } catch {
    // Best-effort — counters are observability, not source-of-truth.
  }
}
