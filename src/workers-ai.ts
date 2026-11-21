/**
 * Workers AI path. Translates an Anthropic Messages request into an
 * OpenAI-style chat-completions call (the shape Cloudflare Workers AI
 * exposes for Llama 4 and similar), runs it, and translates the response
 * (or streamed chunks) back into Anthropic's message / SSE format so
 * @anthropic-ai/sdk on the consumer side parses it transparently.
 *
 * What this covers:
 *   - text-in / text-out chat
 *   - tool use: Anthropic tools[] ↔ OpenAI function-calling, both
 *     directions including tool_result follow-ups
 *   - streaming for the text path; tool-call responses are synthesized
 *     into Anthropic SSE shape from a buffered non-streamed call
 *   - images on user turns, sent as OpenAI vision content parts
 */

import type {
  AnthropicMessagesRequest,
  AnthropicMessageResponse,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicImageSource,
  AnthropicResponseContentBlock,
  AnthropicStopReason,
  AnthropicTool,
  OpenAITool,
  OpenAIToolCall,
  WorkersAiChatRequest,
  WorkersAiChatMessage,
  WorkersAiContentBlock,
  WorkersAiChatResponse,
} from "./types.js";
import { extractOutputSchema, outputSchemaToOpenAIResponseFormat } from "./structured-output.js";
import {
  messageStartEvent,
  parseSseDataLine,
  readSseFrames,
  randomId,
  sseEventWriter,
  sseResponse,
} from "./sse.js";

export interface AiBinding {
  run(
    model: string,
    input: unknown,
    options?: { signal?: AbortSignal },
  ): Promise<unknown>;
}

export function flattenContent(content: AnthropicMessage["content"]): string {
  if (typeof content === "string") return content;
  return content
    .map((block) => {
      if (block.type === "text" && typeof (block as { text?: unknown }).text === "string") {
        return (block as { text: string }).text;
      }
      if (block.type === "tool_result") {
        return toolResultText((block as { content?: unknown }).content) ?? "";
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

/**
 * Text of a tool_result's content: a string as is, an array as its text
 * parts joined by newlines (other parts become empty lines). null for any
 * other shape, so each caller picks its own fallback.
 */
function toolResultText(content: unknown): string | null {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return null;
  return content
    .map((c) =>
      typeof c === "object" && c && "text" in c &&
      typeof (c as { text: unknown }).text === "string"
        ? (c as { text: string }).text
        : "",
    )
    .join("\n");
}

/**
 * Translate Anthropic tools (`{name, description, input_schema}`) into
 * OpenAI/Workers AI tool declarations (`{type: function, function: {...}}`).
 */
export function translateTools(tools: AnthropicTool[] | undefined): OpenAITool[] | undefined {
  if (!tools || tools.length === 0) return undefined;
  return tools.map((t) => ({
    type: "function" as const,
    function: {
      name: t.name,
      description: t.description,
      parameters: (t.input_schema as Record<string, unknown>) ?? { type: "object", properties: {} },
    },
  }));
}

/**
 * Translate Anthropic's `tool_choice` to OpenAI's.
 */
export function translateToolChoice(
  choice: AnthropicMessagesRequest["tool_choice"],
): WorkersAiChatRequest["tool_choice"] {
  if (!choice) return undefined;
  if (typeof choice !== "object") return undefined;
  const c = choice as { type?: string; name?: string };
  switch (c.type) {
    case "auto":
      return "auto";
    case "none":
      return "none";
    case "any":
      return "required";
    case "tool":
      if (typeof c.name === "string") {
        return { type: "function", function: { name: c.name } };
      }
      return "required";
    default:
      return undefined;
  }
}

/**
 * Build the Workers AI request body from an Anthropic request, including
 * tool declarations and the message history with any tool_use /
 * tool_result blocks converted into OpenAI's assistant-with-tool_calls
 * and `role: "tool"` messages respectively.
 */
export function buildWorkersAiInput(req: AnthropicMessagesRequest): WorkersAiChatRequest {
  const messages: WorkersAiChatMessage[] = [];

  let systemText = "";
  if (typeof req.system === "string") {
    systemText = req.system;
  } else if (Array.isArray(req.system)) {
    systemText = (req.system as AnthropicContentBlock[])
      .map((b) => (b.type === "text" && "text" in b ? String(b.text) : ""))
      .filter(Boolean)
      .join("\n");
  }
  if (systemText) {
    messages.push({ role: "system", content: systemText });
  }

  for (const m of req.messages) {
    messages.push(...translateMessage(m));
  }

  // Structured output via output_config — translate to OpenAI-style
  // response_format. Lives here in buildWorkersAiInput so the same body
  // shape flows to the Workers AI binding AND to every HTTP OpenAI-compat
  // provider (Groq/Cerebras/OpenRouter all consume buildWorkersAiInput's
  // output via openai-compatible.ts).
  const outputFmt = extractOutputSchema(req);
  const responseFormat = outputFmt ? outputSchemaToOpenAIResponseFormat(outputFmt).response_format : undefined;

  return {
    messages,
    tools: translateTools(req.tools),
    tool_choice: translateToolChoice(req.tool_choice),
    max_tokens: clampMaxTokens(req.max_tokens),
    temperature: typeof req.temperature === "number" ? req.temperature : undefined,
    top_p: typeof req.top_p === "number" ? req.top_p : undefined,
    top_k: typeof req.top_k === "number" ? req.top_k : undefined,
    stream: Boolean(req.stream),
    ...(Array.isArray(req.stop_sequences) && req.stop_sequences.length > 0
      ? { stop: req.stop_sequences }
      : {}),
    ...(responseFormat ? { response_format: responseFormat } : {}),
  };
}

/**
 * One Anthropic message can become multiple OpenAI messages:
 *   - assistant with text + tool_use blocks → one assistant message with
 *     `content` set to the text and `tool_calls` set from the tool_use blocks
 *   - user with tool_result blocks → one `role: "tool"` message per result,
 *     plus an optional `role: "user"` message for any plain text alongside
 *   - user with image blocks → content becomes a WorkersAiContentBlock[]
 *     array (OpenAI vision format) so Gemma 4 can process the image
 */
function translateMessage(m: AnthropicMessage): WorkersAiChatMessage[] {
  if (typeof m.content === "string") {
    return [{ role: m.role, content: m.content }];
  }

  const textBuf: string[] = [];
  const imageBuf: WorkersAiContentBlock[] = [];
  const toolCalls: OpenAIToolCall[] = [];
  const toolMessages: WorkersAiChatMessage[] = [];

  for (const block of m.content) {
    if (block.type === "text" && typeof (block as { text?: unknown }).text === "string") {
      textBuf.push((block as { text: string }).text);
    } else if (block.type === "image" && m.role === "user") {
      const url = imageUrl((block as { source: AnthropicImageSource }).source);
      // Unrecognised source shape — skip to avoid a broken request.
      if (url) imageBuf.push({ type: "image_url", image_url: { url } });
    } else if (block.type === "tool_use" && m.role === "assistant") {
      toolCalls.push(toOpenAIToolCall(block as ToolUseBlock));
    } else if (block.type === "tool_result" && m.role === "user") {
      toolMessages.push(toToolMessage(block as ToolResultBlock));
    }
  }

  const textJoined = textBuf.join("\n");
  if (m.role === "assistant") return assistantMessages(textJoined, toolCalls);
  return [...userMessages(textJoined, imageBuf), ...toolMessages];
}

/** Data URL for base64 sources, the URL itself for URL sources, else null. */
function imageUrl(src: AnthropicImageSource): string | null {
  if (src.type === "base64" && src.data) {
    return `data:${src.media_type ?? "image/jpeg"};base64,${src.data}`;
  }
  if ((src.type === "url" || src.type === "image_url") && src.url) return src.url;
  return null;
}

type ToolUseBlock = { id: string; name: string; input: unknown };
type ToolResultBlock = { tool_use_id: string; content: unknown };

function toOpenAIToolCall(tu: ToolUseBlock): OpenAIToolCall {
  return {
    id: tu.id,
    type: "function",
    function: {
      name: tu.name,
      arguments:
        typeof tu.input === "string" ? tu.input : JSON.stringify(tu.input ?? {}),
    },
  };
}

function toToolMessage(tr: ToolResultBlock): WorkersAiChatMessage {
  const fallback = tr.content != null ? JSON.stringify(tr.content) : "";
  const body = toolResultText(tr.content) ?? fallback;
  return { role: "tool", tool_call_id: tr.tool_use_id, content: body };
}

function assistantMessages(
  text: string,
  toolCalls: OpenAIToolCall[],
): WorkersAiChatMessage[] {
  if (!text && toolCalls.length === 0) return [];
  return [
    {
      role: "assistant",
      content: text,
      ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
    },
  ];
}

/**
 * User messages with images use the multi-part content array (OpenAI
 * vision format). Pure-text messages stay as a plain string so non-vision
 * models are unaffected.
 */
function userMessages(
  text: string,
  images: WorkersAiContentBlock[],
): WorkersAiChatMessage[] {
  if (images.length > 0) {
    const parts: WorkersAiContentBlock[] = text ? [{ type: "text", text }] : [];
    return [{ role: "user", content: [...parts, ...images] }];
  }
  return text ? [{ role: "user", content: text }] : [];
}

function clampMaxTokens(requested: number): number {
  if (!Number.isFinite(requested) || requested <= 0) return 1024;
  return requested;
}

export async function callWorkersAi(
  ai: AiBinding,
  model: string,
  req: AnthropicMessagesRequest,
  signal?: AbortSignal,
): Promise<Response> {
  const input = buildWorkersAiInput(req);
  const options = signal ? { signal } : undefined;
  const wantsStream = Boolean(input.stream);
  const hasTools = Array.isArray(input.tools) && input.tools.length > 0;

  // Tool-using calls are always run non-streamed and then synthesized
  // into SSE if the caller asked for it. This keeps the streaming
  // re-encoder simple and avoids partial-JSON parsing in tool args.
  if (hasTools) {
    const nonStreamInput = { ...input, stream: false };
    const raw = (await ai.run(
      model,
      nonStreamInput,
      options,
    )) as WorkersAiChatResponse;

    // Tool-silence detection. Opt-in via metadata.fortune_require_tools=true.
    // Gemma 4 26B A4B supports native tool calling, but consumers that need
    // a guaranteed tool call (e.g. strict agentic loops) can set this flag
    // so a silent plain-text response triggers a fallback to the next tier.
    const requireTools =
      (req.metadata as { fortune_require_tools?: boolean } | undefined)?.fortune_require_tools === true;
    if (requireTools) {
      const calls = Array.isArray(raw.tool_calls)
        ? raw.tool_calls
        : Array.isArray(raw.choices?.[0]?.message?.tool_calls)
          ? (raw.choices[0]?.message?.tool_calls ?? [])
          : [];
      if (calls.length === 0) {
        throw new Error(
          "Workers AI returned no tool calls though tools were declared and " +
            "fortune_require_tools=true. Falling through to next tier.",
        );
      }
    }

    if (wantsStream) {
      return synthesizeToolStream(raw, req.model || model);
    }
    const body = workersAiToAnthropicMessage(raw, req.model || model, nonStreamInput);
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  const raw = await ai.run(model, input, options);
  if (wantsStream) {
    if (!(raw instanceof ReadableStream)) {
      throw new Error(`Workers AI did not return a stream for model ${model}`);
    }
    return streamWorkersAiAsAnthropic(raw, model);
  }
  const result = raw as WorkersAiChatResponse;
  const body = workersAiToAnthropicMessage(result, req.model || model, input);
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/**
 * Pull text + tool calls from a Workers AI response, regardless of which
 * response shape the model uses. Newer models (Gemma 4 26B, GPT-OSS,
 * Llama 4) return the OpenAI-compatible `{choices: [{message: {content,
 * tool_calls}}]}` shape. Older models (Llama 3.x, Mistral, classic chat)
 * return the simpler `{response, tool_calls}` shape. Both are valid;
 * the gateway must handle either.
 */
function extractWorkersAiContent(result: WorkersAiChatResponse): {
  text: string;
  toolCalls: OpenAIToolCall[];
} {
  // Legacy shape first — fast path for older models still in rotation.
  const legacyText = typeof result.response === "string" ? result.response : "";
  const legacyCalls = Array.isArray(result.tool_calls) ? result.tool_calls : [];
  if (legacyText || legacyCalls.length > 0) {
    return { text: legacyText, toolCalls: legacyCalls };
  }
  // Newer OpenAI-compatible shape — Gemma 4 and friends.
  const msg = result.choices?.[0]?.message;
  if (msg) {
    return {
      text: typeof msg.content === "string" ? msg.content : "",
      toolCalls: Array.isArray(msg.tool_calls) ? msg.tool_calls : [],
    };
  }
  return { text: "", toolCalls: [] };
}

export function workersAiToAnthropicMessage(
  result: WorkersAiChatResponse,
  modelLabel: string,
  input: WorkersAiChatRequest,
): AnthropicMessageResponse {
  const { text, toolCalls } = extractWorkersAiContent(result);

  const content = toAnthropicContent(text, toolCalls);

  const validToolCalls = toolCalls.filter((tc) => tc.function);
  const finishReason = result.choices?.[0]?.finish_reason;
  let stopReason: AnthropicStopReason;
  if (validToolCalls.length > 0) {
    stopReason = "tool_use";
  } else if (finishReason === "length") {
    stopReason = "max_tokens";
  } else {
    stopReason = "end_turn";
  }
  const promptTokens = result.usage?.prompt_tokens ?? estimatePromptTokens(input);
  const completionTokens =
    result.usage?.completion_tokens ??
    Math.ceil(
      (text.length +
        validToolCalls.reduce((acc, tc) => acc + tc.function.name.length + (tc.function.arguments?.length ?? 0), 0)) /
        4,
    );

  return {
    id: `msg_wa_${randomId()}`,
    type: "message",
    role: "assistant",
    model: modelLabel,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
    },
  };
}

// Builds Anthropic content blocks from an OpenAI-shaped reply's text and
// tool calls, shared by every OpenAI-compatible translator.
export function toAnthropicContent(
  text: string,
  toolCalls: OpenAIToolCall[],
): AnthropicResponseContentBlock[] {
  const content: AnthropicResponseContentBlock[] = [];
  if (text) {
    content.push({ type: "text", text });
  }
  for (const tc of toolCalls) {
    // Guard against malformed upstream responses where `function` is absent.
    if (!tc.function) continue;
    content.push({
      type: "tool_use",
      id: tc.id,
      name: tc.function.name,
      input: safeJsonObject(tc.function.arguments),
    });
  }
  // Anthropic SDK requires at least one content block even on empty responses.
  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }
  return content;
}

function safeJsonObject(s: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(s);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
}

function estimatePromptTokens(input: WorkersAiChatRequest): number {
  const chars = input.messages.reduce((acc, m) => acc + (m.content?.length ?? 0), 0);
  return Math.ceil(chars / 4);
}

/**
 * Synthesize an Anthropic SSE stream from a fully-buffered Workers AI
 * tool-call response. We emit text deltas (if any) followed by one
 * tool_use content block per tool call, with the full args as a single
 * input_json_delta. Same event vocabulary the SDK expects.
 */
function synthesizeToolStream(
  result: WorkersAiChatResponse,
  modelLabel: string,
): Response {
  const messageId = `msg_wa_${randomId()}`;
  const { text, toolCalls } = extractWorkersAiContent(result);

  const downstream = new ReadableStream<Uint8Array>({
    start(controller) {
      const writeEvent = sseEventWriter(controller);

      writeEvent(
        "message_start",
        messageStartEvent(messageId, modelLabel, result.usage?.prompt_tokens ?? 0),
      );

      let blockIndex = 0;
      if (text) {
        writeEvent("content_block_start", {
          type: "content_block_start",
          index: blockIndex,
          content_block: { type: "text", text: "" },
        });
        writeEvent("content_block_delta", {
          type: "content_block_delta",
          index: blockIndex,
          delta: { type: "text_delta", text },
        });
        writeEvent("content_block_stop", { type: "content_block_stop", index: blockIndex });
        blockIndex += 1;
      }

      const validToolCalls = toolCalls.filter((tc) => tc.function);
      for (const tc of validToolCalls) {
        writeEvent("content_block_start", {
          type: "content_block_start",
          index: blockIndex,
          content_block: {
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: {},
          },
        });
        writeEvent("content_block_delta", {
          type: "content_block_delta",
          index: blockIndex,
          delta: { type: "input_json_delta", partial_json: tc.function.arguments ?? "" },
        });
        writeEvent("content_block_stop", { type: "content_block_stop", index: blockIndex });
        blockIndex += 1;
      }

      writeEvent("message_delta", {
        type: "message_delta",
        delta: {
          stop_reason: validToolCalls.length > 0 ? "tool_use" : "end_turn",
          stop_sequence: null,
        },
        usage: {
          output_tokens: result.usage?.completion_tokens ?? 0,
        },
      });
      writeEvent("message_stop", { type: "message_stop" });
      controller.close();
    },
  });

  return sseResponse(downstream);
}

export function streamWorkersAiAsAnthropic(
  upstream: ReadableStream,
  modelLabel: string,
): Response {
  const messageId = `msg_wa_${randomId()}`;

  const downstream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const writeEvent = sseEventWriter(controller);

      writeEvent("message_start", messageStartEvent(messageId, modelLabel));
      writeEvent("content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      });

      let outputTokenCount = 0;
      let streamError = false;
      const processFrame = (frame: string) => {
        const payload = parseSseDataLine(frame);
        if (!payload || payload === "[DONE]") return;
        const delta = extractWorkersAiDelta(payload);
        if (!delta) return;
        outputTokenCount += Math.max(1, Math.ceil(delta.length / 4));
        writeEvent("content_block_delta", {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: delta },
        });
      };
      try {
        await readSseFrames(upstream, processFrame);
      } catch (err) {
        streamError = true;
        console.error("fortune-llm: Workers AI stream failed:", err);
        writeEvent("error", {
          type: "error",
          error: {
            type: "api_error",
            message: "Workers AI stream failed",
          },
        });
      } finally {
        if (!streamError) {
          writeEvent("content_block_stop", { type: "content_block_stop", index: 0 });
          writeEvent("message_delta", {
            type: "message_delta",
            delta: { stop_reason: "end_turn", stop_sequence: null },
            usage: { output_tokens: outputTokenCount },
          });
          writeEvent("message_stop", { type: "message_stop" });
        }
        try {
          controller.close();
        } catch {
          // already closed
        }
      }
    },
  });

  return sseResponse(downstream);
}

/**
 * The text in one streamed Workers AI frame. Older models send
 * `{"response": "..."}`; newer ones (Gemma 4, the default) send OpenAI-style
 * `{"choices": [{"delta": {"content": "..."}}]}`, the streamed twin of the
 * `choices` shape `WorkersAiChatResponse` already reads. `delta.reasoning` is
 * the model's private thinking and is never passed on as reply text.
 */
export function extractWorkersAiDelta(jsonPayload: string): string | null {
  if (!jsonPayload) return null;
  try {
    const obj = JSON.parse(jsonPayload) as {
      response?: unknown;
      choices?: Array<{ delta?: { content?: unknown } }>;
    };
    if (typeof obj.response === "string") return obj.response;
    const content = obj.choices?.[0]?.delta?.content;
    if (typeof content === "string") return content;
  } catch {
    // ignore
  }
  return null;
}
