/**
 * Subset of the Anthropic Messages API request body that the gateway
 * actually inspects, plus the OpenAI- and Gemini-shaped types we
 * translate into on the way to free backends.
 *
 * Anything we don't recognize on the request is forwarded as-is to
 * Anthropic (the explicit-opt-in escape valve).
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
  tools?: AnthropicTool[];
  tool_choice?: AnthropicToolChoice;
  metadata?: Record<string, unknown>;
  thinking?: unknown;
  [extra: string]: unknown;
}

export interface AnthropicMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
}

export type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "image"; source: AnthropicImageSource }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: unknown; is_error?: boolean }
  | { type: string; [extra: string]: unknown };

export interface AnthropicImageSource {
  type: "base64" | "url" | string;
  media_type?: string;
  data?: string;
  url?: string;
  [extra: string]: unknown;
}

export interface AnthropicTool {
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
  [extra: string]: unknown;
}

export type AnthropicToolChoice =
  | { type: "auto" | "any" | "none" }
  | { type: "tool"; name: string }
  | { type: string; [extra: string]: unknown };

export type AnthropicStopReason =
  | "end_turn"
  | "max_tokens"
  | "stop_sequence"
  | "tool_use"
  | null;

export interface AnthropicMessageResponse {
  id: string;
  type: "message";
  role: "assistant";
  model: string;
  content: AnthropicResponseContentBlock[];
  stop_reason: AnthropicStopReason;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
}

export type AnthropicResponseContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown };

/**
 * OpenAI-style chat-completions shape used by Cloudflare Workers AI for
 * Llama 4 and similar tool-capable models.
 */
export interface WorkersAiChatRequest {
  messages: WorkersAiChatMessage[];
  tools?: OpenAITool[];
  tool_choice?: "auto" | "none" | "required" | { type: "function"; function: { name: string } };
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  stream?: boolean;
  /** OpenAI-style stop sequences (up to 4). Forwarded to Groq, OpenRouter, and Workers AI. */
  stop?: string | string[];
  /**
   * OpenAI-style structured output. Forwarded to providers that
   * understand `response_format` (Groq, Cerebras, OpenRouter,
   * newer Workers AI models). The Anthropic gateway translates
   * the SDK's `output_config.format` into this field.
   */
  response_format?: {
    type: "json_schema";
    json_schema: { name: string; schema: Record<string, unknown>; strict?: boolean };
  };
}

export interface WorkersAiChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_call_id?: string;
  tool_calls?: OpenAIToolCall[];
}

export interface OpenAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
}

export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

/**
 * Workers AI response — covers BOTH the legacy `{response, tool_calls}`
 * shape (Llama 3.x, Mistral, older models) and the newer OpenAI-compat
 * `{choices: [{message: {content, tool_calls}}]}` shape that Gemma 4 and
 * other recently-added chat models return. The extractor downstream
 * tries the legacy field first, then falls through to the choices array.
 *
 * Without this, gemma-4-26b returns `{choices: [...]}` with no top-level
 * `response`, so the translator silently emitted empty text while still
 * billing output tokens — visible to consumers as `end_turn` with
 * `content: [{type:"text", text:""}]` and a non-zero output_tokens count.
 */
export interface WorkersAiChatResponse {
  response?: string;
  tool_calls?: OpenAIToolCall[];
  choices?: Array<{
    message?: {
      role?: string;
      content?: string | null;
      tool_calls?: OpenAIToolCall[];
    };
    finish_reason?: string;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

/**
 * Minimal Gemini generateContent API shape. We only model what the
 * gateway translates into and out of.
 */
export interface GeminiGenerateRequest {
  contents: GeminiContent[];
  systemInstruction?: { parts: Array<{ text: string }> };
  tools?: Array<{ functionDeclarations: GeminiFunctionDeclaration[] }>;
  toolConfig?: { functionCallingConfig: { mode: "AUTO" | "ANY" | "NONE"; allowedFunctionNames?: string[] } };
  generationConfig?: {
    temperature?: number;
    topP?: number;
    topK?: number;
    maxOutputTokens?: number;
    stopSequences?: string[];
    thinkingConfig?: { thinkingBudget?: number; includeThoughts?: boolean };
  };
}

export interface GeminiContent {
  role: "user" | "model";
  parts: GeminiPart[];
}

export type GeminiPart =
  | { text: string }
  | { inlineData: { mimeType: string; data: string } }
  | { functionCall: { name: string; args: Record<string, unknown> } }
  | { functionResponse: { name: string; response: Record<string, unknown> } };

export interface GeminiFunctionDeclaration {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
}

export interface GeminiResponse {
  candidates?: Array<{
    content?: { parts?: GeminiPart[]; role?: string };
    finishReason?: string;
  }>;
  usageMetadata?: {
    promptTokenCount?: number;
    candidatesTokenCount?: number;
    totalTokenCount?: number;
  };
}
