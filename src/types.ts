/**
 * Subset of the Anthropic Messages API request body that the gateway
 * actually inspects. We don't need the full type — we only need enough
 * to (a) decide where to route the request and (b) translate text-only
 * shapes into Workers AI input.
 *
 * Anything we don't recognize is forwarded as-is to Anthropic.
 */

export interface AnthropicMessagesRequest {
  model: string;
  messages: AnthropicMessage[];
  system?: string | AnthropicContentBlock[];
  max_tokens: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stop_sequences?: string[];
  stream?: boolean;
  // We route to Anthropic when any of these are present.
  tools?: unknown[];
  tool_choice?: unknown;
  metadata?: Record<string, unknown>;
  thinking?: unknown;
  // Pass-through for anything else.
  [extra: string]: unknown;
}

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

export type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: unknown }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: unknown; is_error?: boolean }
  | { type: string; [extra: string]: unknown };

/**
 * Anthropic non-streaming response shape (the parts we synthesize).
 */
export interface AnthropicMessageResponse {
  id: string;
  type: "message";
  role: "assistant";
  model: string;
  content: Array<{ type: "text"; text: string }>;
  stop_reason: "end_turn" | "max_tokens" | "stop_sequence" | "tool_use" | null;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

/**
 * Workers AI Llama chat completions input. Matches @cf/meta/llama-4-scout-17b-16e-instruct
 * and friends — see https://developers.cloudflare.com/workers-ai/models/llama-4-scout-17b-16e-instruct/
 */
export interface WorkersAiChatRequest {
  messages: Array<{ role: "system" | "user" | "assistant"; content: string }>;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stream?: boolean;
}

export interface WorkersAiChatResponse {
  response: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}
