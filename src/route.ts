/**
 * Per-request routing decision: "OSS via Workers AI" vs "real Anthropic".
 *
 * We default to OSS — it's free for the user. We only escalate when the
 * request meaningfully needs a capability OSS Llama can't reliably
 * deliver:
 *
 *   - Tool use:   `tools` array present (Llama's native tool-call accuracy
 *                 is improving but inconsistent across providers).
 *   - Vision:     any message includes an image content block.
 *   - Long ctx:   approximate input tokens > LONG_CONTEXT_THRESHOLD.
 *                 Llama 3.3 70B handles 128k natively but quality
 *                 degrades; Anthropic's recall is stronger.
 *   - Explicit hint: caller sets `metadata.fortune_route === "anthropic"`.
 *
 * Everything else — plain chat, system prompt + user turn, structured-
 * looking JSON output asks — goes to Workers AI.
 *
 * The function is pure and tested in tests/route.test.ts.
 */

import type { AnthropicMessagesRequest, AnthropicContentBlock } from "./types.js";

export type RouteDecision =
  | { kind: "workers-ai"; reason: string }
  | { kind: "anthropic"; reason: string };

const LONG_CONTEXT_THRESHOLD = 16_000; // approx tokens; we estimate from char count

export function decideRoute(req: AnthropicMessagesRequest): RouteDecision {
  // 1. Explicit caller override wins.
  const meta = req.metadata as { fortune_route?: string } | undefined;
  if (meta?.fortune_route === "anthropic") {
    return { kind: "anthropic", reason: "explicit metadata.fortune_route override" };
  }
  if (meta?.fortune_route === "workers-ai") {
    return { kind: "workers-ai", reason: "explicit metadata.fortune_route override" };
  }

  // 2. Tool use → Anthropic.
  if (Array.isArray(req.tools) && req.tools.length > 0) {
    return { kind: "anthropic", reason: `tools=${req.tools.length}` };
  }

  // 3. Vision → Anthropic.
  if (containsImage(req)) {
    return { kind: "anthropic", reason: "image content block present" };
  }

  // 4. Long context → Anthropic.
  const approxTokens = estimateInputTokens(req);
  if (approxTokens > LONG_CONTEXT_THRESHOLD) {
    return {
      kind: "anthropic",
      reason: `approx ${approxTokens} input tokens > ${LONG_CONTEXT_THRESHOLD}`,
    };
  }

  return { kind: "workers-ai", reason: "default route (no escalation triggers)" };
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
 * averages ~3.7 chars/token in English; we use 4 as a generous upper
 * bound for routing purposes. Off by ~10–15% but the threshold has
 * margin built in.
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
    } else {
      for (const block of m.content) chars += blockCharLength(block);
    }
  }
  return Math.ceil(chars / 4);
}

function blockCharLength(block: AnthropicContentBlock): number {
  if (block.type === "text" && typeof (block as { text?: unknown }).text === "string") {
    return ((block as { text: string }).text).length;
  }
  // Conservative fallback for opaque blocks.
  return 200;
}
