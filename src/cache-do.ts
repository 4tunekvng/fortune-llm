/**
 * Durable Object that holds the gateway's exact-match response cache.
 *
 * Why a DO when the KV-backed cache had the same logical surface: KV
 * reads cache at the Cloudflare edge for ≥60s, so two consecutive
 * identical requests (the common case for an agent loop firing N
 * rapid LLM calls during one task) miss the cache because the second
 * request's read hits a stale edge that hasn't seen the first
 * request's write. A DO actor serializes all reads and writes through
 * one process so the second request always sees the first request's
 * write — cache hits work immediately.
 *
 * Scale: single instance ("cache-singleton"). With 24h TTL and a few
 * hundred unique requests/day, the SQLite table holds at most a few
 * thousand rows. DOs handle millions of ops/sec — plenty of headroom.
 * Shard by hash-prefix if traffic ever justifies it.
 *
 * TTL is enforced at read time (rows where expires_at <= now are
 * treated as misses) and periodically swept by an alarm to keep the
 * table from accumulating dead rows over time.
 *
 * Storage shape:
 *   CREATE TABLE cache_entries (
 *     cache_key TEXT PRIMARY KEY,
 *     body TEXT NOT NULL,
 *     tier TEXT NOT NULL,
 *     model TEXT NOT NULL,
 *     cached_at INTEGER NOT NULL,   -- ms since epoch
 *     expires_at INTEGER NOT NULL   -- ms since epoch
 *   )
 */

import { DurableObject } from "cloudflare:workers";

export interface CachedResponse {
  body: string;
  tier: string;
  model: string;
  cachedAt: number;
}

/** Default sweep interval. Long enough to be cheap, short enough that
 *  expired rows don't accumulate for days. */
const SWEEP_INTERVAL_MS = 60 * 60 * 1000; // 1h

export class CacheDO extends DurableObject {
  constructor(ctx: DurableObjectState, env: unknown) {
    super(ctx, env as Record<string, unknown>);
    ctx.blockConcurrencyWhile(async () => {
      this.ctx.storage.sql.exec(`
        CREATE TABLE IF NOT EXISTS cache_entries (
          cache_key TEXT PRIMARY KEY,
          body TEXT NOT NULL,
          tier TEXT NOT NULL,
          model TEXT NOT NULL,
          cached_at INTEGER NOT NULL,
          expires_at INTEGER NOT NULL
        )
      `);
      this.ctx.storage.sql.exec(
        "CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache_entries(expires_at)",
      );
      // Make sure the sweep alarm is armed on first boot.
      const existing = await this.ctx.storage.getAlarm();
      if (existing === null) {
        await this.ctx.storage.setAlarm(Date.now() + SWEEP_INTERVAL_MS);
      }
    });
  }

  /**
   * Look up a cached response by key. Returns null if missing or
   * expired. Expired rows are NOT cleaned up here — that happens in
   * the alarm sweep — but they're invisible to readers.
   */
  async read(key: string): Promise<CachedResponse | null> {
    const now = Date.now();
    const row = this.ctx.storage.sql
      .exec<{ body: string; tier: string; model: string; cached_at: number; expires_at: number }>(
        "SELECT body, tier, model, cached_at, expires_at FROM cache_entries WHERE cache_key = ? AND expires_at > ?",
        key,
        now,
      )
      .toArray()[0];
    if (!row) return null;
    return { body: row.body, tier: row.tier, model: row.model, cachedAt: row.cached_at };
  }

  /**
   * Store a response. INSERT OR REPLACE so a re-cache (e.g. user
   * forced a fresh dispatch via fortune_no_cache then we ran a
   * subsequent fresh call) just overwrites the prior entry.
   */
  async write(
    key: string,
    entry: CachedResponse,
    ttlSeconds: number,
  ): Promise<void> {
    const now = Date.now();
    const expiresAt = now + Math.max(60, Math.floor(ttlSeconds)) * 1000;
    this.ctx.storage.sql.exec(
      `INSERT OR REPLACE INTO cache_entries
       (cache_key, body, tier, model, cached_at, expires_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
      key,
      entry.body,
      entry.tier,
      entry.model,
      entry.cachedAt,
      expiresAt,
    );
  }

  /** Periodic sweep — drop everything that's expired. */
  async alarm(): Promise<void> {
    const now = Date.now();
    this.ctx.storage.sql.exec("DELETE FROM cache_entries WHERE expires_at <= ?", now);
    // Reschedule.
    await this.ctx.storage.setAlarm(now + SWEEP_INTERVAL_MS);
  }

  /** Test helper / operator-driven reset. Not used in normal flow. */
  async clear(): Promise<void> {
    this.ctx.storage.sql.exec("DELETE FROM cache_entries");
  }

  /**
   * Diagnostic: return the number of live entries. Useful for /stats
   * extensions or debugging cache sizing.
   */
  async size(): Promise<number> {
    const now = Date.now();
    const row = this.ctx.storage.sql
      .exec<{ n: number }>(
        "SELECT COUNT(*) AS n FROM cache_entries WHERE expires_at > ?",
        now,
      )
      .toArray()[0];
    return row?.n ?? 0;
  }
}
