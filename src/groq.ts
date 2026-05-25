/**
 * Groq backend. Free Dev-tier with per-model rate limits. Speaks the
 * canonical OpenAI chat-completions shape, so this is a thin wrapper
 * over the shared adapter.
 *
 * Default model is `meta-llama/llama-4-scout-17b-16e-instruct` (Llama 4
 * Scout, 17B MoE) — chosen for the highest TPM allowance on Groq's free
 * Dev tier (30K TPM as of 2026-05-25) and native tool use.
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
 *   meta-llama/llama-4-scout-17b-16e-instruct     : 30K  ← current default
 *
 * Override via DEFAULT_GROQ_MODEL if you want a different quality/TPM tradeoff.
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";
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
