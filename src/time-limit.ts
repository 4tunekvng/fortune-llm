/**
 * Time limits for the free tiers, so one slow or hanging provider can't
 * hold a request. Two limits apply to every free tier call:
 *
 *   - Its own limit, for the tiers measured fast: 20 s plus 5 ms for each
 *     token the reply may run to (`max_tokens`). On 2026-11-21 gemini's
 *     slowest reply was 6.8 s, a 3.5K-token curriculum, about 600 tokens a
 *     second, and Knox logged groq at about 3 s; 5 ms a token is a third of
 *     that speed, so a long reply is never cut for being long. openrouter
 *     (good replies up to 60 s) and workers-ai (64 s and more) have no own
 *     limit; they sit last in the chain and only the chain budget bounds
 *     them. Override per tier with `TIER_TIME_LIMIT_MS_<TIER>`.
 *   - The chain budget shared by all free tiers: `CHAIN_TIME_LIMIT_MS`
 *     (default 80 s, under Knox's 90 s and Leggo's 100 s deadlines), cut to
 *     the caller's own timeout less 2 s when the SDK sends it in
 *     `X-Stainless-Timeout`, so the caller gets an answer before it hangs up.
 *
 * Paid Anthropic gets neither: it is the last resort and runs as before,
 * except that it is not started when the caller has told us it will hang
 * up within PAID_MIN_LEFT_MS: the reply would be billed and thrown away.
 *
 * A long-context request (over route.ts's 100K-token mark) has no own
 * tier limit either: the speeds above were measured on prompts under 10K
 * tokens.
 *
 * A timeout never shuts the tier for other requests: one slow reply says
 * nothing certain about the next (a 60 s trip on 2026-11-21 sent every app
 * to the slow tiers within a minute of going live).
 */

import type { BackendKind } from "./route.js";

export const DEFAULT_CHAIN_TIME_LIMIT_MS = 80_000;
const FAST_TIER_BASE_MS = 20_000;
const FAST_TIER_MS_PER_TOKEN = 5;
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
/**
 * Paid Anthropic is not started with less than this left before the
 * caller's own timeout: a Claude reply of a few thousand tokens takes
 * 15 s or more, so it would be billed after the caller gave up.
 */
const PAID_MIN_LEFT_MS = 15_000;

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

/** When the caller will hang up, from the SDK's timeout header, if it said. */
export function callerDeadline(
  stainlessTimeout: string | null,
  startedAt: number,
): number | undefined {
  const seconds = positiveMs(stainlessTimeout);
  return seconds === undefined ? undefined : startedAt + seconds * 1000;
}

/** Whether there is still time for a paid reply to reach the caller. */
export function paidHasTime(deadline: number | undefined, now: number): boolean {
  return deadline === undefined || deadline - now >= PAID_MIN_LEFT_MS;
}

/** A tier's own limit, or undefined when only the chain budget bounds it. */
export function tierLimitMs(
  tier: BackendKind,
  env: Record<string, unknown>,
  maxTokens: number,
): number | undefined {
  const name = `TIER_TIME_LIMIT_MS_${tier.toUpperCase().replace(/-/g, "_")}`;
  const raw = env[name];
  const override = positiveMs(typeof raw === "string" ? raw : undefined);
  if (override !== undefined) return override;
  if (!FAST_TIERS.has(tier)) return undefined;
  return FAST_TIER_BASE_MS + FAST_TIER_MS_PER_TOKEN * maxTokens;
}

/** Thrown when a tier call runs past its limit. */
export class TierTimeoutError extends Error {
  constructor(
    tier: BackendKind,
    readonly ms: number,
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
