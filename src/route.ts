/**
 * Per-request routing decision: an ordered fallback chain. The dispatcher
 * tries each tier in order until one succeeds.
 *
 * Default policy (free first, paid as last-resort):
 *   - plain text chat or tools[] →  [gemini, groq, cerebras, mistral,
 *                                     github-models, openrouter, workers-ai]
 *       (seven independent free-quota pools, fastest first; exhausting all
 *       on the same day is what makes the paid fallback rare. A private
 *       caller gets [groq, cerebras, workers-ai].)
 *   - image content present  →  [gemini, openrouter, workers-ai]
 *                                                     (gemini is best free vision; OpenRouter
 *                                                      Llama 4 is next; Gemma 4 26B on Workers AI
 *                                                      is the third free tier — image blocks are
 *                                                      now translated to data-URL / URL in
 *                                                      workers-ai.ts)
 *   - very long context      →  [gemini, openrouter, workers-ai]
 *                                                     (gemini 2.5 flash has 1M+ context;
 *                                                      OpenRouter's free Llama 4 has 256k+;
 *                                                      workers-ai's gemma-4 caps lower)
 *   - has output_config      →  the same chains       (each free tier gets the schema in
 *                                                      its own form; a reply off the schema
 *                                                      counts as a failure and falls through,
 *                                                      see schema-check.ts)
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
  | "github-models"
  | "mistral"
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
  /**
   * The caller's prompts must never reach a tier whose free terms let the
   * provider keep them or learn from them (Knox's health letters).
   */
  privateConsumer?: boolean;
}

/**
 * Free tiers whose providers may keep what they are sent or learn from it:
 * Gemini's unpaid tier, OpenRouter's `:free` model hosts, Mistral's free
 * plan, and GitHub Models (prompts are logged for abuse monitoring). A
 * private caller's chain leaves them out.
 */
const MAY_KEEP_PROMPTS: ReadonlySet<BackendKind> = new Set<BackendKind>([
  "gemini",
  "openrouter",
  "mistral",
  "github-models",
]);

export const LONG_CONTEXT_THRESHOLD = 100_000;

export function decideRoute(
  req: AnthropicMessagesRequest,
  options: DecideRouteOptions = {},
): RouteChain {
  const chain = chooseChain(req, options);
  if (!options.privateConsumer) return chain;
  return {
    tiers: chain.tiers.filter((t) => !MAY_KEEP_PROMPTS.has(t)),
    reason: `${chain.reason}; private caller, tiers that may keep prompts left out`,
  };
}

function chooseChain(
  req: AnthropicMessagesRequest,
  options: DecideRouteOptions,
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
  if (meta?.fortune_route === "github-models") {
    return { tiers: ["github-models"], reason: "explicit metadata.fortune_route=github-models" };
  }
  if (meta?.fortune_route === "mistral") {
    return { tiers: ["mistral"], reason: "explicit metadata.fortune_route=mistral" };
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
      reason: `${baseReason}${structured}; anthropic appended as last-resort (free chain failed -> paid)`,
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
    // is next; Workers AI Gemma 4 26B is the third free-tier option now that
    // image-block translation (base64 → data-URL, URL pass-through) is
    // implemented in workers-ai.ts.
    return ["gemini", "openrouter", "workers-ai"];
  }

  const approxTokens = estimateInputTokens(req);
  if (approxTokens > LONG_CONTEXT_THRESHOLD) {
    // gemini-2.5-flash: 1M+ context. OpenRouter free Llama 4: 256k+.
    // Workers AI Gemma 4: 128k. Groq llama-3.3 caps at 128k (skipped here).
    return ["gemini", "openrouter", "workers-ai"];
  }

  // Default for both plain chat and tool-using calls. Stacks SEVEN
  // independent free-quota pools, fastest first, so the slow pools are the
  // reserve rather than the first stop (replays of 2026-11-21):
  //   gemini         — 1-7 s; per-Google-API-key quota (multi-key rotation)
  //   groq           — about 3 s, native tool use. Its 200K tokens a day are
  //                    shared by every app and were spent by morning on
  //                    2026-11-21, and it is the one fast tier a private
  //                    caller may use, so apps that may use gemini go there
  //                    first and leave groq for the ones that can't.
  //   cerebras       — fast when it works; fails in under 1 s (402 today)
  //   mistral        — free experimental tier; fails in under 1 s (429 today)
  //   github-models  — Azure-backed, free with any GitHub account
  //   openrouter     — 2-60 s; meta-router with multi-model fallback
  //   workers-ai     — 11-64 s and more; Cloudflare account-scoped neurons
  // The pools are the same as before, so when gemini's quota runs out the
  // chain lands where it used to; it is faster whenever gemini answers.
  return [
    "gemini",
    "groq",
    "cerebras",
    "mistral",
    "github-models",
    "openrouter",
    "workers-ai",
  ];
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
