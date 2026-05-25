/**
 * GitHub Models backend. Free with any GitHub account — Azure-backed
 * OpenAI-compatible API exposing OpenAI / Llama / Phi / Mistral models.
 * Independent quota pool from every other tier.
 *
 * Free tier limits (verified 2026-05-25 against docs.github.com/github-models):
 *   - 15 RPM, 150 RPD, ~8k token outputs. Tight, but it's a SIXTH
 *     independent free pool stacked on top of the others.
 *
 * Endpoint: https://models.github.ai/inference/chat/completions
 * Auth:     Authorization: Bearer <github-pat>
 *
 * Default: openai/gpt-4o-mini — fast, supports tool use, supports
 * response_format json_schema (Phase 4 structured-output compatible).
 *
 * GitHub requires the model to be qualified with the publisher prefix
 * (`openai/gpt-4o-mini`, `meta/llama-3.3-70b-instruct`, etc.); the
 * model id in the request body is the qualified form.
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { callOpenAICompatible } from "./openai-compatible.js";

export const DEFAULT_GITHUB_MODELS_MODEL = "openai/gpt-4o-mini";
const GITHUB_MODELS_URL = "https://models.github.ai/inference/chat/completions";

export interface CallGitHubModelsOptions {
  apiKey: string;
  model: string;
}

export async function callGitHubModels(
  opts: CallGitHubModelsOptions,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  return callOpenAICompatible(
    {
      label: "GitHub Models",
      url: GITHUB_MODELS_URL,
      apiKey: opts.apiKey,
      model: opts.model,
    },
    req,
  );
}
