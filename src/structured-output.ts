/**
 * Structured-output translation. The Anthropic SDK's `messages.parse()`
 * helper sends `output_config: { format: { type: "json_schema", schema } }`.
 * Until Phase 4 the gateway routed any `output_config` request straight
 * to Anthropic because "free tiers don't speak it." That cost 60-80% of
 * consumer traffic in dollars.
 *
 * Reality (verified against @anthropic-ai/sdk source, lib/beta-parser.js
 * and helpers/zod.js):
 *   - the wire format is just `{ type: "json_schema", schema }`. The
 *     `parse` function on `zodOutputFormat(...)` lives client-side and
 *     never crosses the wire.
 *   - the response is a normal Anthropic message whose text content IS
 *     the JSON object. The SDK calls JSON.parse(content) locally to
 *     populate `parsed_output`.
 *   - the schema arrives pre-sanitized for OpenAI-style strict mode
 *     (additionalProperties:false, all object-types declared, etc.)
 *     because the SDK's `transformJSONSchema` enforces it before send.
 *
 * So our job is:
 *   1. Detect output_config.format.type === "json_schema"
 *   2. Translate to each provider's native structured-output feature:
 *        OpenAI-compat:  `response_format: { type, json_schema: { name, schema, strict } }`
 *        Gemini:         `generationConfig.{responseMimeType, responseSchema}`
 *        Workers AI:     `response_format: { type, json_schema }`
 *   3. Provider returns text content that's pure JSON.
 *   4. Wrap as a normal Anthropic message — SDK's beta-parser handles
 *      the rest locally.
 *
 * Falls through gracefully: if the upstream model rejects the schema
 * or produces invalid JSON, the dispatcher's quota/error handling
 * advances to the next tier. Anthropic remains the last-resort safety
 * net via the existing auto-append.
 */

import type { AnthropicMessagesRequest } from "./types.js";

/** The wire-format shape of `output_config.format` when type is json_schema. */
export interface OutputJsonSchemaFormat {
  type: "json_schema";
  schema: Record<string, unknown>;
  /** Optional name; default to "response". Used by OpenAI's strict mode. */
  name?: string;
}

export interface OutputConfigOnRequest {
  format?: OutputJsonSchemaFormat;
}

/**
 * Return the JSON-schema output format from a request, or null if the
 * request doesn't have one (or it's not the json_schema variant).
 */
export function extractOutputSchema(req: AnthropicMessagesRequest): OutputJsonSchemaFormat | null {
  const oc = (req as { output_config?: OutputConfigOnRequest }).output_config;
  const fmt = oc?.format;
  if (!fmt || fmt.type !== "json_schema") return null;
  if (!fmt.schema || typeof fmt.schema !== "object") return null;
  return fmt;
}

/** Convenience: returns just the schema object, or null. */
export function getOutputSchemaObject(req: AnthropicMessagesRequest): Record<string, unknown> | null {
  return extractOutputSchema(req)?.schema ?? null;
}

/**
 * Build the OpenAI-style `response_format` body field from an Anthropic
 * `output_config.format`. Used by Groq / Cerebras / OpenRouter and any
 * other OpenAI-chat-compatible upstream.
 *
 * `strict: true` is supported by OpenAI's `gpt-*` models, Groq's Llama 4
 * and newer Llama 3.3+, and most modern OpenAI-compat providers. When
 * a model rejects strict mode the request 400s and our dispatcher
 * advances to the next tier — that's the right escape valve.
 */
export function outputSchemaToOpenAIResponseFormat(fmt: OutputJsonSchemaFormat): {
  response_format: {
    type: "json_schema";
    json_schema: { name: string; schema: Record<string, unknown>; strict: true };
  };
} {
  // `name` is required by OpenAI strict mode. Use the caller-provided
  // name if present, otherwise a generic placeholder. The model only
  // uses this for its internal bookkeeping, never in the output.
  const name = (fmt.name && fmt.name.match(/^[a-zA-Z0-9_-]+$/) ? fmt.name : "response").slice(0, 64);
  return {
    response_format: {
      type: "json_schema",
      json_schema: {
        name,
        schema: fmt.schema,
        strict: true,
      },
    },
  };
}

/**
 * Build the Gemini generationConfig fields for a structured output
 * request. Gemini accepts a sanitized subset of JSON Schema as
 * `responseSchema`; the sanitizer (in gemini.ts) walks the schema and
 * drops fields Gemini doesn't recognize.
 *
 * NOTE: this function returns the partial config; the caller merges it
 * into the existing generationConfig. We don't sanitize here because
 * sanitizeJsonSchemaForGemini lives in gemini.ts (close to the rest of
 * the Gemini-specific schema-massaging logic).
 */
export function outputSchemaToGeminiGenerationConfig(
  schema: Record<string, unknown>,
): { responseMimeType: "application/json"; responseSchema: Record<string, unknown> } {
  return {
    responseMimeType: "application/json",
    responseSchema: schema,
  };
}
