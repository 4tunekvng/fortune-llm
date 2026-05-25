/**
 * Per-IP rate limit using a KV counter bucketed by minute.
 *
 * Why: the gateway has one shared `GATEWAY_TOKEN` used by every consumer
 * app. If any one consumer (a runaway loop, a misbehaving cron job, a
 * leaked token) blasts the gateway, it drains the shared free-tier
 * quota for every other consumer. A per-IP cap keeps one bad citizen
 * from breaking the others.
 *
 * Mechanism:
 *   - Key: `rate:<ip>:<minute>`. The minute bucket auto-expires via
 *     KV TTL so we don't need to clean up.
 *   - On each request, atomically read the current count, reject if
 *     >= limit, otherwise increment. KV is eventually-consistent so
 *     there's a small race-window where a spike can sneak slightly
 *     over the limit — acceptable for our purposes (this is rough
 *     abuse protection, not strict rate-shaping).
 *   - When KV isn't bound (local dev) the limiter no-ops.
 *
 * Tuning: the default is 200 req/min per IP. That's generous enough
 * to never hit on normal use (any single consumer running at e.g. one
 * request every couple seconds), and tight enough that a runaway loop
 * burns out within minutes instead of hours.
 */

export const DEFAULT_RATE_LIMIT_PER_MIN = 200;
const KEY_PREFIX = "rate:";

export interface RateLimitDecision {
  allowed: boolean;
  /** Current count in the window after this request would land. */
  count: number;
  /** Per-window limit. */
  limit: number;
  /** Seconds until the current bucket expires. */
  retryAfterSeconds: number;
}

/**
 * Identify the consumer for rate-limiting. Order of preference:
 *   1. `x-forwarded-for` first hop (Cloudflare sets this for non-CF clients).
 *   2. `cf-connecting-ip` (Cloudflare's authoritative client IP).
 *   3. fallback string "unknown" so we don't crash if no IP is available.
 */
export function getClientIp(request: Request): string {
  const cfIp = request.headers.get("cf-connecting-ip");
  if (cfIp) return cfIp;
  const xff = request.headers.get("x-forwarded-for");
  if (xff) {
    const first = xff.split(",")[0]?.trim();
    if (first) return first;
  }
  return "unknown";
}

/**
 * Check and increment the rate-limit counter for this IP. KV-backed so
 * the count is shared across all Worker isolates. Eventually-consistent
 * by design.
 */
export async function checkRateLimit(
  ip: string,
  limit: number,
  kv: KVNamespace | undefined,
): Promise<RateLimitDecision> {
  if (!kv || limit <= 0) {
    return { allowed: true, count: 0, limit, retryAfterSeconds: 0 };
  }
  const now = Date.now();
  const minute = Math.floor(now / 60_000);
  const key = `${KEY_PREFIX}${ip}:${minute}`;
  // Seconds left in this minute bucket. KV TTL minimum is 60s, so we
  // bump the minimum to keep KV happy even at the very end of the
  // window.
  const secondsLeft = Math.max(60, 60 - Math.floor((now % 60_000) / 1000));

  let current = 0;
  try {
    const raw = await kv.get(key);
    if (raw) {
      const parsed = parseInt(raw, 10);
      if (Number.isFinite(parsed)) current = parsed;
    }
  } catch {
    // KV unavailable — fail open. Rate limit is best-effort; better to
    // miss a limit than to block all traffic on infra trouble.
    return { allowed: true, count: 0, limit, retryAfterSeconds: 0 };
  }

  const next = current + 1;
  if (current >= limit) {
    return { allowed: false, count: current, limit, retryAfterSeconds: secondsLeft };
  }

  try {
    await kv.put(key, String(next), { expirationTtl: secondsLeft });
  } catch {
    // Write failure — count was incremented in spirit but not persisted.
    // Worst case: this single request slips past the limit. Acceptable.
  }
  return { allowed: true, count: next, limit, retryAfterSeconds: 0 };
}

/**
 * Resolve the rate-limit threshold from env, with bounds. Setting to 0
 * disables the limiter entirely.
 */
export function resolveRateLimitPerMin(envValue: string | undefined): number {
  if (envValue === undefined) return DEFAULT_RATE_LIMIT_PER_MIN;
  const n = Number(envValue);
  if (!Number.isFinite(n) || n < 0) return DEFAULT_RATE_LIMIT_PER_MIN;
  // Cap at 100k/min — a hilariously-high ceiling that still prevents
  // accidental setting of e.g. "20000000" from disabling protection.
  return Math.min(Math.floor(n), 100_000);
}
