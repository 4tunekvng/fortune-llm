/**
 * OpenRouter backend. Meta-router that exposes a large catalog of models
 * (including many `:free` variants) behind one API key. Provides an
 * independent free-quota pool that doesn't share with Google / Cloudflare
 * / Groq.
 *
 * Why this file is more than a thin config: free models on OpenRouter
 * are routed through community-supplied upstream providers, and any
 * single `:free` model can be hammered by other OpenRouter users at any
 * moment — when that happens, the upstream returns 429. To make this
 * tier durable we leverage OpenRouter's *built-in* multi-model fallback:
 * pass a `models: [...]` array and OpenRouter tries each in order,
 * surfacing only the one that succeeds. We also set
 * `provider.sort: "throughput"` so OpenRouter prefers the least-loaded
 * upstream provider per model.
 *
 * The default model list is chosen for diversity (different upstream
 * providers, different model families) so it's unlikely all of them are
 * rate-limited at the same instant. Override via DEFAULT_OPENROUTER_MODELS
 * (comma-separated) in wrangler.toml.
 *
 * OpenRouter expects `HTTP-Referer` and `X-Title` for attribution; we
 * set them to the gateway URL so OpenRouter analytics show the right
 * consumer.
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

/**
 * Default fallback list. Ordered for diversity — different model
 * families and different upstream providers — so a rate-limit on any
 * one model is unlikely to also affect the next in line.
 *
 * Updated 2026-05-25 from OpenRouter's free-tier catalog. When this list
 * goes stale, override via DEFAULT_OPENROUTER_MODELS without code edits.
 *
 * IMPORTANT: OpenRouter limits the `models` array to 3 entries. The
 * list-resolver enforces this cap so longer override lists silently
 * truncate to the first three rather than erroring at the API.
 */
export const DEFAULT_OPENROUTER_MODELS: readonly string[] = [
  "deepseek/deepseek-v4-flash:free",
  "openai/gpt-oss-120b:free",
  "qwen/qwen3-coder:free",
];

/** OpenRouter's API rejects `models` arrays longer than this. */
export const OPENROUTER_MAX_MODELS = 3;

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

export interface CallOpenRouterOptions {
  apiKey: string;
  /** Ordered list of model ids. The first is sent as `model:`; the full list as `models:`. */
  models: string[];
  /** Optional gateway origin for OpenRouter attribution headers. */
  referer?: string;
}

/**
 * Parse a comma-separated model list (from env) into a clean array.
 * Falls back to DEFAULT_OPENROUTER_MODELS when empty.
 */
export function resolveOpenRouterModels(envValue: string | undefined): string[] {
  if (!envValue) return [...DEFAULT_OPENROUTER_MODELS];
  const out = envValue
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
  const list = out.length > 0 ? out : [...DEFAULT_OPENROUTER_MODELS];
  // OpenRouter rejects `models` arrays of more than 3 entries; truncate
  // silently rather than failing the request at the API.
  return list.slice(0, OPENROUTER_MAX_MODELS);
}

export async function callOpenRouter(
  opts: CallOpenRouterOptions,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  if (opts.models.length === 0) {
    throw new Error("OpenRouter called with no models configured");
  }
  // `model` is the primary; `models` is the ordered fallback list. When
  // OpenRouter sees both, it uses `models` as the fallback order with
  // `model` ignored (we keep `model` set for consumers that introspect
  // logs by model id). `provider.sort: "throughput"` tells OpenRouter to
  // pick the least-loaded upstream provider per model — important for
  // `:free` models whose default upstreams are heavily contested.
  return callOpenAICompatible(
    {
      label: "OpenRouter",
      url: OPENROUTER_URL,
      apiKey: opts.apiKey,
      model: opts.models[0] as string,
      extraHeaders: {
        "HTTP-Referer": opts.referer ?? "https://fortune-llm.fortunee.workers.dev",
        "X-Title": "fortune-llm gateway",
      },
      extraBody: {
        models: opts.models,
        provider: { sort: "throughput", allow_fallbacks: true },
      },
    },
    req,
  );
}
