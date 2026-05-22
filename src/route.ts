/**
 * Per-request routing decision: an ordered fallback chain. The dispatcher
 * tries each tier in order until one succeeds.
 *
 * Default policy (free first, paid as last-resort):
 *   - plain text chat        →  [workers-ai, gemini]  (small model fine; save Gemini quota)
 *   - has tools[]            →  [workers-ai, gemini]  (Gemma 4 26B A4B has native structured
 *                                                      tool use; Gemini is the fallback when
 *                                                      Workers AI quota trips)
 *   - image content present  →  [gemini]              (image translation not yet implemented
 *                                                      for Workers AI path)
 *   - very long context      →  [gemini, workers-ai]  (Gemini's window is bigger)
 *
 * When `options.anthropicFallback === true` (i.e. ANTHROPIC_API_KEY is
 * configured on the worker), Anthropic is appended as the last-resort
 * tier to every default chain. The free tiers are still tried first; we
 * only escalate to paid when *every* free option has failed or is
 * circuit-broken. This is what makes the gateway "always works" for
 * consumers — Lena, knox, the agents — even when free quotas are dry.
 *
 * Explicit per-request overrides via `metadata.fortune_route`:
 *   - "anthropic"   →  [anthropic]                    (force paid)
 *   - "free"        →  [workers-ai, gemini] (or rule-derived equivalent)
 *                                                    (free-only, no paid fallback,
 *                                                     even if anthropic is configured)
 *   - "workers-ai"  →  [workers-ai]
 *   - "gemini"      →  [gemini]
 *
 * When the chain is exhausted (every tier failed or rate-limited) the
 * gateway fails loudly with a 503. Auto-fallback to Anthropic only kicks
 * in when the *worker* has an Anthropic key — there is no silent
 * escalation if the operator hasn't opted in by configuring it.
 */

import type { AnthropicMessagesRequest, AnthropicContentBlock } from "./types.js";

export type BackendKind = "workers-ai" | "gemini" | "anthropic";

export interface RouteChain {
  tiers: BackendKind[];
  reason: string;
}

export interface DecideRouteOptions {
  /**
   * Worker has ANTHROPIC_API_KEY configured. When true, Anthropic is
   * appended to every default (non-explicit-override) chain as the
   * last-resort tier, used only when every free tier has failed or
   * been circuit-broken.
   */
  anthropicFallback?: boolean;
}

const LONG_CONTEXT_THRESHOLD = 100_000;

export function decideRoute(
  req: AnthropicMessagesRequest,
  options: DecideRouteOptions = {},
): RouteChain {
  const meta = req.metadata as { fortune_route?: string } | undefined;

  // Explicit per-request overrides — these short-circuit the auto-fallback.
  if (meta?.fortune_route === "anthropic") {
    return { tiers: ["anthropic"], reason: "explicit metadata.fortune_route=anthropic" };
  }
  if (meta?.fortune_route === "workers-ai") {
    return { tiers: ["workers-ai"], reason: "explicit metadata.fortune_route=workers-ai" };
  }
  if (meta?.fortune_route === "gemini") {
    return { tiers: ["gemini"], reason: "explicit metadata.fortune_route=gemini" };
  }
  // "free" forces the default free chain even when anthropicFallback is on.
  // Lets a caller opt out of paid escalation per-request (e.g. background
  // jobs that should fail gracefully rather than burn dollars).
  if (meta?.fortune_route === "free") {
    return {
      tiers: defaultFreeChain(req),
      reason: "explicit metadata.fortune_route=free (no paid fallback)",
    };
  }

  const free = defaultFreeChain(req);
  const baseReason = freeChainReason(req, free);
  if (options.anthropicFallback) {
    return {
      tiers: [...free, "anthropic"],
      reason: `${baseReason}; anthropic appended as last-resort (free chain failed → paid)`,
    };
  }
  return { tiers: free, reason: baseReason };
}

/**
 * The free-only chain for a given request shape. Pure: no env, no options.
 * Use this both as the default chain and as the result for the `free`
 * metadata override.
 */
function defaultFreeChain(req: AnthropicMessagesRequest): BackendKind[] {
  if (containsImage(req)) return ["gemini"];

  const approxTokens = estimateInputTokens(req);
  if (approxTokens > LONG_CONTEXT_THRESHOLD) return ["gemini", "workers-ai"];

  // Tools[] present → Claude-Code-style agent traffic. Gemma 4 26B A4B
  // has native structured tool use; use it first. Gemini is the fallback
  // when Workers AI quota trips (the circuit breaker handles it).
  if (Array.isArray(req.tools) && req.tools.length > 0) {
    return ["workers-ai", "gemini"];
  }

  return ["workers-ai", "gemini"];
}

function freeChainReason(req: AnthropicMessagesRequest, chain: BackendKind[]): string {
  if (containsImage(req)) return "image content block present";
  const approxTokens = estimateInputTokens(req);
  if (approxTokens > LONG_CONTEXT_THRESHOLD) {
    return `approx ${approxTokens} input tokens > ${LONG_CONTEXT_THRESHOLD}`;
  }
  if (Array.isArray(req.tools) && req.tools.length > 0) {
    return `tools=${req.tools.length}; workers-ai (Gemma 4) handles tool use natively`;
  }
  return chain[0] === "workers-ai"
    ? "plain text chat; workers-ai default"
    : `default free chain ${chain.join(", ")}`;
}

function containsImage(req: AnthropicMessagesRequest): boolean {
  for (const m of req.messages) {
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (isImageBlock(block)) return true;
      }
    }
  }
  if (Array.isArray(req.system)) {
    for (const block of req.system) {
      if (isImageBlock(block)) return true;
    }
  }
  return false;
}

function isImageBlock(block: AnthropicContentBlock): boolean {
  return block.type === "image";
}

/**
 * Crude character-count → token approximation. Anthropic's tokenizer
 * averages ~3.7 chars/token in English; we use 4 for routing purposes.
 */
export function estimateInputTokens(req: AnthropicMessagesRequest): number {
  let chars = 0;
  if (typeof req.system === "string") chars += req.system.length;
  if (Array.isArray(req.system)) {
    for (const block of req.system) chars += blockCharLength(block);
  }
  for (const m of req.messages) {
    if (typeof m.content === "string") {
      chars += m.content.length;
    } else if (Array.isArray(m.content)) {
      for (const block of m.content) chars += blockCharLength(block);
    }
  }
  return Math.ceil(chars / 4);
}

function blockCharLength(block: AnthropicContentBlock): number {
  if (block.type === "text" && typeof (block as { text?: unknown }).text === "string") {
    return ((block as { text: string }).text).length;
  }
  return 200;
}
