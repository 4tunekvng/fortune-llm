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
import { ghostThoughtFilter, stripGhostThought } from "./ghost-thought.js";
import { extractOutputSchema, outputSchemaToOpenAIResponseFormat } from "./structured-output.js";
import {
  messageStartEvent,
  parseSseDataLine,
  readSseFrames,
  randomId,
  keepAliveWriter,
  sseEventWriter,
  saysSomething,
  sseResponse,
  synthesizeAnthropicSSE,
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
  // Gemma 4 thinks before it answers unless told not to: on a 6K-token
  // prompt it wrote about 3,000 characters of hidden reasoning, took 29 s,
  // and sometimes spent all of max_tokens before a word of the answer.
  // With thinking off the same question took 1.8 s. Anthropic-style
  // requests ask for no thinking, so this matches the Gemini tier, which
  // turns it off on its 2.5 models for the same reason (gemini.ts).
  const input: WorkersAiChatRequest = {
    ...buildWorkersAiInput(req),
    chat_template_kwargs: { enable_thinking: false },
  };
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

    const body = workersAiToAnthropicMessage(
      raw,
      req.model || model,
      nonStreamInput,
      model,
    );
    // index.ts rejects an empty reply only when it is JSON, and this one may
    // go back as a stream, so reject it here.
    if (!saysSomething(body)) throw new Error("Workers AI returned an empty reply");
    if (wantsStream) return synthesizeAnthropicSSE(body);
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
  const body = workersAiToAnthropicMessage(result, req.model || model, input, model);
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
  /** The Workers AI model that wrote the reply. */
  model: string,
): AnthropicMessageResponse {
  const { text, toolCalls } = extractWorkersAiContent(result);

  const content = toAnthropicContent(text, toolCalls, model);

  const validToolCalls = toolCalls.filter((tc) => tc.function);
  const finishReason = result.choices?.[0]?.finish_reason;
  const stopReason = openAIStopReason(finishReason, validToolCalls.length);
  if (stopReason === null) throw unfinishedReplyError("Workers AI", finishReason);
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

// Finish reasons that mean the model ended its reply itself. Any other
// (content_filter, OpenRouter's error, Mistral's model_length) means the
// provider cut the reply off.
const FINISHED_REASONS = new Set(["stop", "length", "tool_calls", "function_call"]);

/**
 * The Anthropic stop reason for an OpenAI-shaped reply, or null when the
 * reply did not finish: the provider cut it off, or it promised a tool call
 * that never came. Shared by every OpenAI-compatible translator, so a cut-off
 * reply fails the tier instead of reaching the caller as a normal end. A
 * missing reason still counts as finished (fortune-llm#52).
 */
export function openAIStopReason(
  finishReason: string | null | undefined,
  toolCallCount: number,
): AnthropicStopReason | null {
  if (finishReason && !FINISHED_REASONS.has(finishReason)) return null;
  // Checked before tool calls: a call cut off mid-arguments is no call.
  if (finishReason === "length") return "max_tokens";
  if (toolCallCount > 0) return "tool_use";
  if (finishReason === "tool_calls" || finishReason === "function_call") return null;
  return "end_turn";
}

/** Thrown when a whole reply did not finish, so the chain tries the next tier. */
export function unfinishedReplyError(
  label: string,
  finishReason: string | null | undefined,
): Error {
  return new Error(
    `${label} did not finish its reply (${finishReason ?? "no finish reason"}). ` +
      "Falling through to next tier.",
  );
}

// Builds Anthropic content blocks from an OpenAI-shaped reply's text and
// tool calls, shared by every OpenAI-compatible translator. `model` is the
// one that wrote the text, which decides what counts as a ghost thought.
export function toAnthropicContent(
  text: string,
  toolCalls: OpenAIToolCall[],
  model: string,
): AnthropicResponseContentBlock[] {
  const content: AnthropicResponseContentBlock[] = [];
  const shown = stripGhostThought(text, model);
  if (shown) {
    content.push({ type: "text", text: shown });
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

export function streamWorkersAiAsAnthropic(
  upstream: ReadableStream,
  modelLabel: string,
): Response {
  const messageId = `msg_wa_${randomId()}`;

  const downstream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const { writeEvent, sawFrame } = keepAliveWriter(sseEventWriter(controller));

      writeEvent("message_start", messageStartEvent(messageId, modelLabel));
      writeEvent("content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      });

      let outputTokenCount = 0;
      let streamError = false;
      const ghost = ghostThoughtFilter(modelLabel);
      const writeText = (text: string) => {
        if (!text) return;
        writeEvent("content_block_delta", {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text },
        });
      };
      let finishReason: string | undefined;
      // A stream that ends with neither [DONE] nor a finish reason was cut.
      let sawDone = false;
      const processFrame = (frame: string) => {
        const payload = parseSseDataLine(frame);
        if (payload === "[DONE]") sawDone = true;
        if (!payload || payload === "[DONE]") return;
        const { text: delta, finishReason: reason } = extractWorkersAiDelta(payload);
        if (reason) finishReason = reason;
        if (!delta) return;
        outputTokenCount += Math.max(1, Math.ceil(delta.length / 4));
        writeText(ghost.push(delta));
      };
      try {
        await readSseFrames(upstream, (frame) => {
          processFrame(frame);
          sawFrame();
        });
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
        const cut = !sawDone && finishReason === undefined;
        const stopReason = cut ? null : openAIStopReason(finishReason, 0);
        if (!streamError && stopReason === null) {
          // Already streaming, so no other tier can take over: end on an
          // error the caller sees and retries, not on a clean stop.
          writeEvent("error", {
            type: "error",
            error: {
              type: "api_error",
              message: `Workers AI did not finish its reply (${
                finishReason ?? "stream cut off"
              })`,
            },
          });
        } else if (!streamError) {
          writeText(ghost.flush());
          writeEvent("content_block_stop", { type: "content_block_stop", index: 0 });
          writeEvent("message_delta", {
            type: "message_delta",
            delta: { stop_reason: stopReason, stop_sequence: null },
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
 * The text and finish reason in one streamed Workers AI frame. Older models
 * send `{"response": "..."}`; newer ones (Gemma 4, the default) send
 * OpenAI-style `{"choices": [{"delta": {"content": "..."}, "finish_reason"}]}`,
 * the streamed twin of the `choices` shape `WorkersAiChatResponse` already
 * reads. `delta.reasoning` is the model's private thinking and is never
 * passed on as reply text.
 */
export function extractWorkersAiDelta(jsonPayload: string): {
  text: string | null;
  finishReason?: string;
} {
  if (!jsonPayload) return { text: null };
  try {
    const obj = JSON.parse(jsonPayload) as {
      response?: unknown;
      choices?: Array<{
        delta?: { content?: unknown };
        finish_reason?: unknown;
      }>;
    };
    const reason = obj.choices?.[0]?.finish_reason;
    const finishReason = typeof reason === "string" ? reason : undefined;
    if (typeof obj.response === "string") {
      return { text: obj.response, finishReason };
    }
    const content = obj.choices?.[0]?.delta?.content;
    return { text: typeof content === "string" ? content : null, finishReason };
  } catch {
    return { text: null };
  }
}
