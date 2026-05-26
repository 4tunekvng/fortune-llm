/**
 * Rate limit using a KV counter bucketed by minute.
 *
 * Key shape: `rate:<scope>:<minute>` where <scope> is either:
 *   - the consumer name (from `x-fortune-consumer: <name>` header), OR
 *   - the client IP (cf-connecting-ip → x-forwarded-for → "unknown")
 *
 * Why two-tier scoping: the gateway is reached by N consumer apps
 * (knox, lena, network-agent, …) often from the same Vercel/CF edge.
 * Pure IP-scoping would let one app drain another's quota when they
 * happen to share an origin IP. With `x-fortune-consumer`, each app
 * gets its own bucket — one runaway agent only burns its own budget.
 * Consumers that don't set the header fall back to IP-scoping (the
 * old behavior).
 *
 * Per-consumer caps can be overridden in env via
 * `RATE_LIMIT_PER_MIN_<CONSUMER>` (uppercased, e.g.
 * `RATE_LIMIT_PER_MIN_LENA=500`). Unmatched consumers use the global
 * `RATE_LIMIT_PER_MIN` default.
 *
 * Mechanism:
 *   - On each request, atomically read the current count, reject if
 *     >= limit, otherwise increment. KV is eventually-consistent so
 *     there's a small race-window where a spike can sneak slightly
 *     over the limit — acceptable for rough abuse protection.
 *   - When KV isn't bound (local dev) the limiter no-ops.
 */

export const DEFAULT_RATE_LIMIT_PER_MIN = 200;
const KEY_PREFIX = "rate:";
/** Sanitize consumer name for KV key safety + env-var lookup. */
const CONSUMER_RE = /^[a-z0-9_-]{1,32}$/i;

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
 * Identify the consumer IP for rate-limiting. Order of preference:
 *   1. `cf-connecting-ip` (Cloudflare's authoritative client IP).
 *   2. `x-forwarded-for` first hop.
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
 * Resolve the rate-limit scope: the consumer name (if a valid
 * `x-fortune-consumer` header is set) or the client IP. The scope
 * becomes the bucket key for rate-limiting AND the lookup for any
 * per-consumer limit override.
 *
 * Sanitized: must match `[a-z0-9_-]{1,32}` to prevent KV-key-injection
 * or absurdly long bucket keys. Invalid headers fall back to IP-scope.
 */
export function getRateLimitScope(request: Request): { scope: string; kind: "consumer" | "ip" } {
  const raw = request.headers.get("x-fortune-consumer");
  if (raw && CONSUMER_RE.test(raw)) {
    return { scope: raw.toLowerCase(), kind: "consumer" };
  }
  return { scope: getClientIp(request), kind: "ip" };
}

/**
 * Per-consumer rate limit override. Operators can set
 * `RATE_LIMIT_PER_MIN_LENA=500` in wrangler.toml to give Lena a higher
 * ceiling than the global default. Unmatched consumers use the global
 * value.
 */
export function resolveConsumerRateLimit(
  consumer: string,
  globalDefault: number,
  envBag: Record<string, string | undefined>,
): number {
  if (!CONSUMER_RE.test(consumer)) return globalDefault;
  const envName = `RATE_LIMIT_PER_MIN_${consumer.toUpperCase().replace(/-/g, "_")}`;
  return resolveRateLimitPerMin(envBag[envName] ?? String(globalDefault));
}

/**
 * Check and increment the rate-limit counter for this scope (consumer
 * name OR IP). KV-backed so the count is shared across all Worker
 * isolates. Eventually-consistent by design.
 */
export async function checkRateLimit(
  scope: string,
  limit: number,
  kv: KVNamespace | undefined,
): Promise<RateLimitDecision> {
  if (!kv || limit <= 0) {
    return { allowed: true, count: 0, limit, retryAfterSeconds: 0 };
  }
  const now = Date.now();
  const minute = Math.floor(now / 60_000);
  const key = `${KEY_PREFIX}${scope}:${minute}`;
  // KV TTL must be >= 60s (Cloudflare minimum). Retry-After reflects the
  // actual seconds until the window rolls over (minimum 1).
  const kvTtl = 60;
  const secondsLeft = Math.max(1, 60 - Math.floor((now % 60_000) / 1000));

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
    await kv.put(key, String(next), { expirationTtl: kvTtl });
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
