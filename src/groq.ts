/**
 * Groq backend. Free tier with generous limits (varies by model, ~30 RPM
 * and a daily token cap that's roughly 1000+ requests/day for typical
 * chat-sized payloads). Native tool use on Llama 3.3 70B and Llama 4.
 *
 * Speaks the canonical OpenAI chat-completions shape, so this is a thin
 * wrapper over the shared adapter — no provider-specific translation
 * code lives here.
 *
 * Defaults to llama-3.3-70b-versatile. Override via DEFAULT_GROQ_MODEL.
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile";
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

export interface CallGroqOptions {
  apiKey: string;
  model: string;
}

export async function callGroq(
  opts: CallGroqOptions,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  return callOpenAICompatible(
    {
      label: "Groq",
      url: GROQ_URL,
      apiKey: opts.apiKey,
      model: opts.model,
    },
    req,
  );
}
