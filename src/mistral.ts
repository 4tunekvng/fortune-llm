/**
 * Mistral La Plateforme backend. Free experimental tier across several
 * Mistral and Codestral models — independent quota pool, OpenAI-chat-
 * compatible, native tool use, supports response_format json_schema
 * (Phase 4 structured-output compatible).
 *
 * Endpoint: https://api.mistral.ai/v1/chat/completions
 * Auth:     Authorization: Bearer <key>
 *
 * Default: mistral-small-latest — currently the strongest free-tier
 * choice (24B parameters, 128k context, tool use, structured outputs).
 * Other production-free choices include `open-mistral-nemo` and
 * `ministral-3b-latest`. Override via DEFAULT_MISTRAL_MODEL.
 *
 * Free tier rate limits (Experimental plan, subject to change):
 *   ~1 RPS / 500k tokens-per-minute / 1B tokens-per-month.
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_MISTRAL_MODEL = "mistral-small-latest";
const MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions";

export interface CallMistralOptions {
  apiKey: string;
  model: string;
}

export async function callMistral(
  opts: CallMistralOptions,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  return callOpenAICompatible(
    {
      label: "Mistral",
      url: MISTRAL_URL,
      apiKey: opts.apiKey,
      model: opts.model,
    },
    req,
  );
}
