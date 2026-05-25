/**
 * Per-request routing decision: an ordered fallback chain. The dispatcher
 * tries each tier in order until one succeeds.
 *
 * Default policy (free first, paid as last-resort):
 *   - plain text chat        →  [groq, workers-ai, gemini, openrouter]
 *                                                     (four independent free-quota pools
 *                                                      — exhausting all on the same day
 *                                                      is what makes the paid fallback rare)
 *   - has tools[]            →  [groq, workers-ai, gemini, openrouter]
 *                                                     (groq llama-3.3-70b has the best free
 *                                                      tool-use; gemma-4 / gemini / openrouter
 *                                                      are independent fallbacks)
 *   - image content present  →  [gemini, openrouter]  (gemini is the best free vision; OpenRouter
 *                                                      has free Llama 4 vision as a safety net)
 *   - very long context      →  [gemini, openrouter, workers-ai]
 *                                                     (gemini 2.5 flash has 1M+ context;
 *                                                      OpenRouter's free Llama 4 has 256k+;
 *                                                      workers-ai's gemma-4 caps lower)
 *   - has output_config      →  [anthropic]           (json_schema structured output is an
 *                                                      Anthropic-native SDK feature — free
 *                                                      backends don't understand the field
 *                                                      and return code-fenced JSON that the
 *                                                      SDK's text parser then chokes on)
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
 *   - "free"        →  rule-derived free chain        (no paid fallback even if anthropic
 *                                                      is configured — for background jobs
 *                                                      that should fail gracefully)
 *   - "workers-ai"  →  [workers-ai]
 *   - "gemini"      →  [gemini]
 *   - "groq"        →  [groq]
 *   - "openrouter"  →  [openrouter]
 *
 * When the chain is exhausted (every tier failed or rate-limited) the
 * gateway fails loudly with a 503. Auto-fallback to Anthropic only kicks
 * in when the *worker* has an Anthropic key — there is no silent
 * escalation if the operator hasn't opted in by configuring it.
 */

import type { AnthropicMessagesRequest, AnthropicContentBlock } from "./types.js";

export type BackendKind =
  | "workers-ai"
  | "gemini"
  | "groq"
  | "cerebras"
  | "openrouter"
  | "anthropic";

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
  if (meta?.fortune_route === "groq") {
    return { tiers: ["groq"], reason: "explicit metadata.fortune_route=groq" };
  }
  if (meta?.fortune_route === "cerebras") {
    return { tiers: ["cerebras"], reason: "explicit metadata.fortune_route=cerebras" };
  }
  if (meta?.fortune_route === "openrouter") {
    return { tiers: ["openrouter"], reason: "explicit metadata.fortune_route=openrouter" };
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

  // Structured output via output_config flows through the regular free
  // chain as of Phase 4. Every free provider now translates the schema
  // to its native structured-output feature in code:
  //   - Groq / Cerebras / OpenRouter / Workers AI → response_format: { type:"json_schema", ... }
  //   - Gemini → generationConfig.responseSchema + responseMimeType:"application/json"
  // If a provider's model rejects the schema (strict-mode-incompatible
  // or model lacks support), the call 400s and the dispatcher advances
  // to the next tier. Anthropic remains the last-resort auto-fallback.
  const free = defaultFreeChain(req);
  const baseReason = freeChainReason(req, free);
  const structured = hasOutputConfig(req) ? " (structured output via response_format / responseSchema)" : "";
  if (options.anthropicFallback) {
    return {
      tiers: [...free, "anthropic"],
      reason: `${baseReason}${structured}; anthropic appended as last-resort (free chain failed → paid)`,
    };
  }
  return { tiers: free, reason: `${baseReason}${structured}` };
}

function hasOutputConfig(req: AnthropicMessagesRequest): boolean {
  const oc = (req as { output_config?: { format?: { type?: string } } }).output_config;
  return oc?.format?.type === "json_schema";
}

/**
 * The free-only chain for a given request shape. Pure: no env, no options.
 * Use this both as the default chain and as the result for the `free`
 * metadata override.
 *
 * Chain ordering principle: stack as many *independent* free-quota pools
 * as possible. Each provider has its own daily/RPM quota, so a chain of
 * N providers gives roughly N× the per-day request ceiling vs. any one
 * of them alone. The point of the gateway: when one quota dries up,
 * silently slide to the next pool.
 */
function defaultFreeChain(req: AnthropicMessagesRequest): BackendKind[] {
  if (containsImage(req)) {
    // Gemini is the strongest free vision model; OpenRouter's free Llama 4
    // is the backup. Workers AI Gemma 4 supports vision but image-block
    // translation isn't implemented in workers-ai.ts yet.
    return ["gemini", "openrouter"];
  }

  const approxTokens = estimateInputTokens(req);
  if (approxTokens > LONG_CONTEXT_THRESHOLD) {
    // gemini-2.5-flash: 1M+ context. OpenRouter free Llama 4: 256k+.
    // Workers AI Gemma 4: 128k. Groq llama-3.3 caps at 128k (skipped here).
    return ["gemini", "openrouter", "workers-ai"];
  }

  // Default for both plain chat and tool-using calls. Stacks five
  // independent free-quota pools in order:
  //   groq        — fastest, native tool use, biggest free RPM
  //   cerebras    — also very fast (custom silicon), independent Cerebras quota
  //   workers-ai  — Cloudflare account-scoped neurons
  //   gemini      — per-Google-API-key quota (multi-key rotation supported)
  //   openrouter  — meta-router with its own free pool + multi-model fallback
  // Five independent quotas = "rarely hits anthropic" in practice.
  return ["groq", "cerebras", "workers-ai", "gemini", "openrouter"];
}

function freeChainReason(req: AnthropicMessagesRequest, chain: BackendKind[]): string {
  if (containsImage(req)) return "image content block present; gemini-led vision chain";
  const approxTokens = estimateInputTokens(req);
  if (approxTokens > LONG_CONTEXT_THRESHOLD) {
    return `approx ${approxTokens} input tokens > ${LONG_CONTEXT_THRESHOLD}; long-context chain`;
  }
  if (Array.isArray(req.tools) && req.tools.length > 0) {
    return `tools=${req.tools.length}; multi-provider free chain (${chain.join(",")})`;
  }
  return `plain text chat; default free chain ${chain.join(",")}`;
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
