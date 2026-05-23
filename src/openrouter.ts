/**
 * OpenRouter backend. Meta-router that exposes a large catalog of models
 * (including several `:free` variants) behind one API key. Provides an
 * independent free-quota pool that doesn't share with Google/Cloudflare/Groq.
 *
 * Free models rotate — recent stable picks include:
 *   - meta-llama/llama-4-maverick:free
 *   - deepseek/deepseek-r1-distill-llama-70b:free
 *   - qwen/qwen-2.5-72b-instruct:free
 *
 * Default: meta-llama/llama-3.3-70b-instruct:free (strong tool use, stable).
 * Override via DEFAULT_OPENROUTER_MODEL.
 *
 * OpenRouter expects `HTTP-Referer` and `X-Title` headers for attribution;
 * we set them to the fortune-llm gateway URL so OpenRouter analytics
 * show the right consumer.
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free";
const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";

export interface CallOpenRouterOptions {
  apiKey: string;
  model: string;
  /** Optional gateway origin for OpenRouter attribution headers. */
  referer?: string;
}

export async function callOpenRouter(
  opts: CallOpenRouterOptions,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  return callOpenAICompatible(
    {
      label: "OpenRouter",
      url: OPENROUTER_URL,
      apiKey: opts.apiKey,
      model: opts.model,
      extraHeaders: {
        "HTTP-Referer": opts.referer ?? "https://fortune-llm.fortunee.workers.dev",
        "X-Title": "fortune-llm gateway",
      },
    },
    req,
  );
}
