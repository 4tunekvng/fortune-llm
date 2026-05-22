/**
 * Per-request routing decision: an ordered fallback chain of free
 * backends. The dispatcher tries them in order until one succeeds.
 *
 * Default policy (zero-paid):
 *   - plain text chat        →  [workers-ai, gemini]  (small model fine; save Gemini quota)
 *   - has tools[]            →  [gemini, workers-ai]  (Llama 4 Scout reliably bounces
 *                                                      Claude-Code-style agent prompts —
 *                                                      "Your input is incomplete" — so
 *                                                      complex agentic traffic prefers
 *                                                      Gemini, with Workers AI as backup
 *                                                      via the circuit-breaker fallback)
 *   - image content present  →  [gemini]              (Workers AI has no vision)
 *   - very long context      →  [gemini, workers-ai]  (Gemini's window is bigger)
 *
 * Explicit per-request overrides via `metadata.fortune_route`:
 *   - "anthropic"   →  [anthropic]   (escape valve — paid)
 *   - "workers-ai"  →  [workers-ai]
 *   - "gemini"      →  [gemini]
 *
 * When the chain is exhausted (every tier failed or rate-limited) the
 * gateway fails loudly. There is intentionally no silent escalation to
 * paid Anthropic.
 */

import type { AnthropicMessagesRequest, AnthropicContentBlock } from "./types.js";

export type BackendKind = "workers-ai" | "gemini" | "anthropic";

export interface RouteChain {
  tiers: BackendKind[];
  reason: string;
}

const LONG_CONTEXT_THRESHOLD = 100_000;

export function decideRoute(req: AnthropicMessagesRequest): RouteChain {
  const meta = req.metadata as { fortune_route?: string } | undefined;

  if (meta?.fortune_route === "anthropic") {
    return { tiers: ["anthropic"], reason: "explicit metadata.fortune_route=anthropic" };
  }
  if (meta?.fortune_route === "workers-ai") {
    return { tiers: ["workers-ai"], reason: "explicit metadata.fortune_route=workers-ai" };
  }
  if (meta?.fortune_route === "gemini") {
    return { tiers: ["gemini"], reason: "explicit metadata.fortune_route=gemini" };
  }

  if (containsImage(req)) {
    return { tiers: ["gemini"], reason: "image content block present" };
  }

  const approxTokens = estimateInputTokens(req);
  if (approxTokens > LONG_CONTEXT_THRESHOLD) {
    return {
      tiers: ["gemini", "workers-ai"],
      reason: `approx ${approxTokens} input tokens > ${LONG_CONTEXT_THRESHOLD}`,
    };
  }

  // Tools[] present → Claude-Code-style agent traffic. Llama 4 Scout
  // reliably bounces these with "Your input is incomplete" (empirically
  // verified 2026-05-22). Route to Gemini first; Workers AI is the
  // fallback when Gemini's quota trips (the circuit breaker handles it).
  if (Array.isArray(req.tools) && req.tools.length > 0) {
    return {
      tiers: ["gemini", "workers-ai"],
      reason: `tools=${req.tools.length}; gemini handles agent prompts more reliably than workers-ai`,
    };
  }

  return { tiers: ["workers-ai", "gemini"], reason: "plain text chat; workers-ai default" };
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
