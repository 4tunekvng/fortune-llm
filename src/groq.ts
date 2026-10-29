/**
 * Groq backend. Free Dev-tier with per-model rate limits. Speaks the
 * canonical OpenAI chat-completions shape, so this is a thin wrapper
 * over the shared adapter.
 *
 * Default model is `openai/gpt-oss-120b`, a Groq production model with
 * native tool use. Llama 4 Scout (the previous default, chosen for its 30K
 * TPM) was retired by Groq and 404s as of 2026-10-28.
 *
 * Why TPM matters: under real chat load the 70B model trips its 12K TPM
 * limit within minutes, repeatedly opening Groq's circuit breaker and
 * pushing traffic to the next tier. The 30K TPM Llama 4 Scout keeps
 * more requests on the cheapest/fastest free provider before falling
 * through.
 *
 * Reference TPM ceilings on Groq Dev tier (subject to change):
 *   llama-3.1-8b-instant                          : 6K
 *   qwen/qwen3-32b                                : 6K
 *   llama-3.3-70b-versatile                       : 12K
 *   openai/gpt-oss-120b                           : current default
 *
 * Override via DEFAULT_GROQ_MODEL if you want a different quality/TPM tradeoff.
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b";
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
