/**
 * Cerebras backend. Free Inference tier on their custom silicon —
 * native OpenAI chat-completions, native tool use, an order of
 * magnitude faster than typical GPU inference. Independent quota pool
 * from Groq / Cloudflare / Google / OpenRouter.
 *
 * Production-available models (verified 2026-05-25 against
 * inference-docs.cerebras.ai/models/overview):
 *   - `gpt-oss-120b`   ← default. 120B OpenAI OSS, native tool use, strong general purpose.
 *   - `llama3.1-8b`    smaller / faster but weaker on tool use.
 *
 * Preview models (subject to change):
 *   - qwen-3-235b-a22b-instruct-2507
 *   - zai-glm-4.7
 *
 * Override via DEFAULT_CEREBRAS_MODEL in wrangler.toml.
 *
 * Endpoint: https://api.cerebras.ai/v1/chat/completions
 * Auth: Authorization: Bearer <key>
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_CEREBRAS_MODEL = "gpt-oss-120b";
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
