/**
 * Time limits for the free tiers, so one slow or hanging provider can't
 * hold a request. Two limits apply to every free tier call:
 *
 *   - Its own limit, for the tiers measured fast. On 2026-11-21 gemini's
 *     slowest reply was 6.8 s (an 8K-token curriculum) and Knox logged
 *     groq at about 3 s, so 20 s is about 3x the slowest reply: past it the
 *     provider is hanging, not working. openrouter (good replies up to 60 s)
 *     and workers-ai (good replies up to 64 s and more) have no own limit;
 *     they sit last in the chain and only the chain budget bounds them.
 *     Override per tier with `TIER_TIME_LIMIT_MS_<TIER>`.
 *   - The chain budget shared by all free tiers: `CHAIN_TIME_LIMIT_MS`
 *     (default 80 s, under Knox's 90 s and Leggo's 100 s deadlines), cut to
 *     the caller's own timeout less 2 s when the SDK sends it in
 *     `X-Stainless-Timeout`, so the caller gets an answer before it hangs up.
 *
 * Paid Anthropic gets neither: it is the last resort and runs as before.
 */

import type { BackendKind } from "./route.js";

export const DEFAULT_CHAIN_TIME_LIMIT_MS = 80_000;
const FAST_TIER_LIMIT_MS = 20_000;
const FAST_TIERS: ReadonlySet<BackendKind> = new Set<BackendKind>([
  "groq",
  "cerebras",
  "mistral",
  "github-models",
  "gemini",
]);
/** Time kept back from the caller's own timeout for the reply to travel. */
const CLIENT_MARGIN_MS = 2_000;
/** Below this the chain could not reach even one tier. */
const MIN_BUDGET_MS = 5_000;
/** How long a tier that ran out its own limit stays skipped. */
export const TIMEOUT_TRIP_MS = 60_000;

function positiveMs(raw: string | undefined | null): number | undefined {
  if (raw === undefined || raw === null || raw.trim() === "") return undefined;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

/**
 * The free tiers' shared budget for one request. The TypeScript SDK sends
 * `X-Stainless-Timeout` as whole seconds ("90"); the Python SDK sends a
 * float ("600.0") or "NOT_GIVEN". Anything that isn't a positive number is
 * ignored.
 */
export function chainBudgetMs(
  envValue: string | undefined,
  stainlessTimeout: string | null,
): number {
  const base = positiveMs(envValue) ?? DEFAULT_CHAIN_TIME_LIMIT_MS;
  const clientSeconds = positiveMs(stainlessTimeout);
  if (clientSeconds === undefined) return base;
  const client = clientSeconds * 1000 - CLIENT_MARGIN_MS;
  return Math.min(base, Math.max(MIN_BUDGET_MS, client));
}

/** A tier's own limit, or undefined when only the chain budget bounds it. */
export function tierLimitMs(
  tier: BackendKind,
  env: Record<string, unknown>,
): number | undefined {
  const name = `TIER_TIME_LIMIT_MS_${tier.toUpperCase().replace(/-/g, "_")}`;
  const raw = env[name];
  const override = positiveMs(typeof raw === "string" ? raw : undefined);
  if (override !== undefined) return override;
  return FAST_TIERS.has(tier) ? FAST_TIER_LIMIT_MS : undefined;
}

/** Thrown when a tier call runs past its limit. */
export class TierTimeoutError extends Error {
  constructor(
    tier: BackendKind,
    readonly ms: number,
    /** True when the tier's own limit fired, not the chain budget. */
    readonly ownLimit: boolean,
  ) {
    super(`${tier} timed out after ${ms} ms`);
    this.name = "TierTimeoutError";
  }
}

/**
 * Run `call` with an abort signal that fires after `ms`. The call gets the
 * signal so its fetch is cancelled; the race makes sure the chain moves on
 * even if some step ignores it. The timer is cleared once the call returns,
 * so a stream that has started is never cut.
 */
export async function withTimeLimit<T>(
  call: (signal: AbortSignal) => Promise<T>,
  ms: number,
  onTimeout: () => Error,
): Promise<T> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expired = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      const err = onTimeout();
      controller.abort(err);
      reject(err);
    }, ms);
  });
  const running = call(controller.signal);
  // The losing call may still reject after the race is settled.
  running.catch(() => {});
  try {
    return await Promise.race([running, expired]);
  } finally {
    clearTimeout(timer);
  }
}
