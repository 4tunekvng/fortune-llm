/**
 * Per-backend circuit breaker.
 *
 * When a tier (workers-ai, gemini) blows its free-tier quota, the gateway
 * was silently retrying on every request — wasting Worker CPU and
 * propagating latency to all 12+ consumer apps. This module records
 * exhaustion in a KV namespace (`CIRCUIT`) keyed by tier and, while the
 * circuit is open, the dispatcher skips the tier entirely.
 *
 * Open-circuit shape stored at `circuit:<tier>`:
 *   { openedAt: number, until: number, reason: string }
 *
 * The KV TTL is set to `(until - now) / 1000` so the record auto-expires
 * when the circuit closes — but the `until` field is still the source of
 * truth on every read so we never serve a stale-but-not-yet-purged entry.
 *
 * Heuristic for tripping: `isQuotaError` matches quota / rate-limit
 * signals on the error message string (Gemini's `RESOURCE_EXHAUSTED`,
 * HTTP 429, "rate limit", "quota", "exceeded"). Transient 5xx and
 * network errors are intentionally NOT tripped — they're worth retrying.
 */
import type { BackendKind } from "./route.js";

const HOUR_MS = 60 * 60 * 1000;
export const DEFAULT_TRIP_DURATION_MS = HOUR_MS; // 1h

interface CircuitRecord {
  openedAt: number;
  until: number;
  reason: string;
}

export interface CircuitState {
  open: boolean;
  until?: number;
  reason?: string;
}

/**
 * Pattern-match a thrown error to decide whether it signals quota or
 * rate-limit exhaustion (vs. a transient 5xx / network blip). Both
 * Workers AI and Gemini surface these as exceptions from their SDK
 * wrappers, so we read the `.message` (or the stringified error) and
 * look for the canonical signals each provider emits.
 *
 * True-positives:
 *   - Gemini:     "RESOURCE_EXHAUSTED", "429 Too Many Requests"
 *   - Workers AI: "rate limit", "quota exceeded", "neurons exhausted"
 *   - Generic:    "429", "quota", "rate limit exceeded"
 *
 * Not-tripped (returns false): timeouts, generic 5xx, network errors,
 * "internal error", connection resets. Those should retry the same tier
 * next request — they're transient, not a quota signal.
 */
export function isQuotaError(error: unknown): boolean {
  const msg = (error instanceof Error ? error.message : String(error ?? "")).toLowerCase();
  if (!msg) return false;

  // Explicit quota / rate-limit signals
  if (msg.includes("resource_exhausted")) return true;
  if (msg.includes("quota")) return true;
  if (msg.includes("rate limit")) return true;
  if (msg.includes("rate-limit")) return true;
  if (msg.includes("ratelimit")) return true;
  if (msg.includes("too many requests")) return true;
  if (msg.includes("neurons exhausted")) return true;
  if (msg.includes("neuron limit")) return true;

  // HTTP 429 — match as a status token, not as a substring inside an
  // arbitrary number. We look for "429" preceded by start/space/colon
  // and followed by end/space/colon/punctuation.
  if (/(^|[\s:(\[])429($|[\s:,.\])])/.test(msg)) return true;

  // "exceeded" alone is ambiguous (could be "max_tokens exceeded"). Pair
  // it with quota-ish context.
  if (msg.includes("exceeded") && (msg.includes("limit") || msg.includes("daily") || msg.includes("usage"))) {
    return true;
  }

  return false;
}

/**
 * Read the current circuit state for a tier. Returns `{open: false}` if
 * no record exists, if the record has expired (`until <= now`), or if
 * the stored JSON is malformed. Never throws — a KV read failure
 * degrades to "circuit closed" so we don't accidentally block all
 * traffic on infra trouble.
 */
export async function getCircuitState(
  tier: BackendKind,
  kv: KVNamespace | undefined,
): Promise<CircuitState> {
  if (!kv) return { open: false };
  let raw: string | null;
  try {
    raw = await kv.get(`circuit:${tier}`);
  } catch {
    return { open: false };
  }
  if (!raw) return { open: false };
  let parsed: CircuitRecord;
  try {
    parsed = JSON.parse(raw) as CircuitRecord;
  } catch {
    return { open: false };
  }
  if (typeof parsed.until !== "number" || parsed.until <= Date.now()) {
    return { open: false };
  }
  return { open: true, until: parsed.until, reason: parsed.reason };
}

/**
 * Open (or re-open / extend) the circuit for a tier. The KV entry is
 * given a TTL that matches `durationMs` so the record self-cleans, but
 * the dispatcher still gates on the stored `until` timestamp on every
 * read — TTLs in Cloudflare KV are best-effort.
 *
 * Re-tripping while already open *replaces* the record with the new
 * `until`, which extends the window (intentional: a second quota error
 * during an already-open window means upstream is still angry).
 */
export async function tripCircuit(
  tier: BackendKind,
  kv: KVNamespace | undefined,
  durationMs: number,
  reason: string,
): Promise<void> {
  if (!kv) return;
  const now = Date.now();
  const until = now + durationMs;
  const record: CircuitRecord = {
    openedAt: now,
    until,
    reason: reason.slice(0, 500), // bound to keep KV value small
  };
  // KV `expirationTtl` is in seconds and must be >= 60.
  const ttlSeconds = Math.max(60, Math.ceil(durationMs / 1000));
  try {
    await kv.put(`circuit:${tier}`, JSON.stringify(record), { expirationTtl: ttlSeconds });
  } catch {
    // KV write failures degrade silently — the next request will just
    // attempt the tier again, which is the same as today's behavior.
  }
}

/** Floor and ceiling for a short (per-second / per-minute) trip. */
export const SHORT_TRIP_MIN_MS = 30 * 1000;
export const SHORT_TRIP_MAX_MS = 5 * 60 * 1000;
/** A bare 429 that doesn't say which window it hit. */
export const AMBIGUOUS_TRIP_MS = 2 * 60 * 1000;

/**
 * How long to keep a tier's circuit open for this quota error. Every 429
 * used to trip for the full window (1h): one burst past Mistral's 1 req/s,
 * Groq's tokens-per-minute or Gemini's requests-per-minute took that free
 * tier out for an hour and sent every app down the chain to the slow tiers.
 * A daily or billing limit still trips for `longMs`; a per-second or
 * per-minute limit trips only as long as the provider says it lasts
 * ("try again in 2.3s", "retryDelay": "17s"), clamped to 30s-5min; a 429
 * that doesn't say which window gets 2 minutes.
 */
export function tripDurationFor(error: unknown, longMs: number): number {
  const msg = (error instanceof Error ? error.message : String(error ?? "")).toLowerCase();
  const daily =
    /per\s*-?\s*day|perday|daily|\brpd\b|tokens per day|payment|billing|credits|insufficient|neurons exhausted|neuron limit/.test(msg);
  if (daily) return longMs;
  const short =
    /per\s*-?\s*(minute|second)|perminute|persecond|\b(tpm|rpm|rps)\b|try again in|retry\s*-?\s*(after|delay)|retrydelay/.test(msg);
  if (!short) return Math.min(longMs, AMBIGUOUS_TRIP_MS);
  // "9.5s", "1m30.5s", "20m", "47s", "350ms": sum each number by its unit.
  const m = /(?:try again in|retry\s*-?\s*(?:after|delay)\W*)\s*((?:\d+(?:\.\d+)?\s*(?:ms|h|m|s)?\s*)+)/.exec(msg);
  let ms = 60 * 1000;
  if (m) {
    const unit = { ms: 1, s: 1000, m: 60 * 1000, h: HOUR_MS } as Record<string, number>;
    let total = 0;
    for (const [, n, u] of (m[1] ?? "").matchAll(/(\d+(?:\.\d+)?)\s*(ms|h|m|s)?/g)) total += Number(n) * (unit[u ?? "s"] ?? 1000);
    if (total > 0) ms = total + 1000;
  }
  return Math.min(longMs, Math.max(SHORT_TRIP_MIN_MS, Math.min(SHORT_TRIP_MAX_MS, ms)));
}

/**
 * Resolve the trip duration from env, with bounds. Negative or non-
 * numeric values fall back to the default. Capped at 24h so a typo
 * doesn't lock a tier out for a week.
 */
export function resolveTripDurationMs(envValue: string | undefined): number {
  if (!envValue) return DEFAULT_TRIP_DURATION_MS;
  const parsed = Number(envValue);
  if (!Number.isFinite(parsed) || parsed <= 0) return DEFAULT_TRIP_DURATION_MS;
  const MAX = 24 * 60 * 60 * 1000;
  return Math.min(parsed, MAX);
}
