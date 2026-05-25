/**
 * Cerebras backend. Free tier with very fast inference on their custom
 * silicon, OpenAI-chat-compat API, native tool use. Independent quota
 * pool from Groq / Cloudflare / Google / OpenRouter.
 *
 * Default: llama-4-scout-17b-16e-instruct (newer, strong tool use).
 * Alternatives that work well on free tier: qwen-3-32b, llama-3.3-70b.
 * Override via DEFAULT_CEREBRAS_MODEL.
 *
 * Endpoint: https://api.cerebras.ai/v1/chat/completions
 * Auth: Authorization: Bearer <key>
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_CEREBRAS_MODEL = "llama-4-scout-17b-16e-instruct";
const CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions";

export interface CallCerebrasOptions {
  apiKey: string;
  model: string;
}

export async function callCerebras(
  opts: CallCerebrasOptions,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  return callOpenAICompatible(
    {
      label: "Cerebras",
      url: CEREBRAS_URL,
      apiKey: opts.apiKey,
      model: opts.model,
    },
    req,
  );
}
