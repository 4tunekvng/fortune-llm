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
 *
 * What this still does NOT cover (router routes these to Gemini):
 *   - images (vision) — Gemma 4 supports vision but image-block translation
 *     from Anthropic format is not yet implemented here
 */

import type {
  AnthropicMessagesRequest,
  AnthropicMessageResponse,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicResponseContentBlock,
  AnthropicStopReason,
  AnthropicTool,
  OpenAITool,
  OpenAIToolCall,
  WorkersAiChatRequest,
  WorkersAiChatMessage,
  WorkersAiChatResponse,
} from "./types.js";
import { extractOutputSchema, outputSchemaToOpenAIResponseFormat } from "./structured-output.js";

export interface AiBinding {
  run(model: string, input: unknown): Promise<unknown>;
}

export function flattenContent(content: AnthropicMessage["content"]): string {
  if (typeof content === "string") return content;
  return content
    .map((block) => {
      if (block.type === "text" && typeof (block as { text?: unknown }).text === "string") {
        return (block as { text: string }).text;
      }
      if (block.type === "tool_result") {
        const tr = block as { content?: unknown };
        if (typeof tr.content === "string") return tr.content;
        if (Array.isArray(tr.content)) {
          return tr.content
            .map((c) =>
              typeof c === "object" && c && "text" in c && typeof (c as { text: unknown }).text === "string"
                ? (c as { text: string }).text
                : "",
            )
            .join("\n");
        }
      }
      return "";
    })
    .filter(Boolean)
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
 */
function translateMessage(m: AnthropicMessage): WorkersAiChatMessage[] {
  if (typeof m.content === "string") {
    return [{ role: m.role, content: m.content }];
  }

  const out: WorkersAiChatMessage[] = [];
  const textBuf: string[] = [];
  const toolCalls: OpenAIToolCall[] = [];
  const toolMessages: WorkersAiChatMessage[] = [];

  for (const block of m.content) {
    if (block.type === "text" && typeof (block as { text?: unknown }).text === "string") {
      textBuf.push((block as { text: string }).text);
    } else if (block.type === "tool_use" && m.role === "assistant") {
      const tu = block as { id: string; name: string; input: unknown };
      toolCalls.push({
        id: tu.id,
        type: "function",
        function: {
          name: tu.name,
          arguments: typeof tu.input === "string" ? tu.input : JSON.stringify(tu.input ?? {}),
        },
      });
    } else if (block.type === "tool_result" && m.role === "user") {
      const tr = block as { tool_use_id: string; content: unknown };
      let body = "";
      if (typeof tr.content === "string") {
        body = tr.content;
      } else if (Array.isArray(tr.content)) {
        body = tr.content
          .map((c) =>
            typeof c === "object" && c && "text" in c && typeof (c as { text: unknown }).text === "string"
              ? (c as { text: string }).text
              : "",
          )
          .join("\n");
      } else if (tr.content != null) {
        body = JSON.stringify(tr.content);
      }
      toolMessages.push({
        role: "tool",
        tool_call_id: tr.tool_use_id,
        content: body,
      });
    }
  }

  const textJoined = textBuf.join("\n");

  if (m.role === "assistant") {
    if (textJoined || toolCalls.length > 0) {
      out.push({
        role: "assistant",
        content: textJoined,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
    }
  } else {
    if (textJoined) {
      out.push({ role: "user", content: textJoined });
    }
    out.push(...toolMessages);
  }

  return out;
}

function clampMaxTokens(requested: number): number {
  if (!Number.isFinite(requested) || requested <= 0) return 1024;
  return requested;
}

export async function callWorkersAi(
  ai: AiBinding,
  model: string,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  const input = buildWorkersAiInput(req);
  const wantsStream = Boolean(input.stream);
  const hasTools = Array.isArray(input.tools) && input.tools.length > 0;

  // Tool-using calls are always run non-streamed and then synthesized
  // into SSE if the caller asked for it. This keeps the streaming
  // re-encoder simple and avoids partial-JSON parsing in tool args.
  if (hasTools) {
    const nonStreamInput = { ...input, stream: false };
    const raw = (await ai.run(model, nonStreamInput)) as WorkersAiChatResponse;

    // Tool-silence detection. Opt-in via metadata.fortune_require_tools=true.
    // Gemma 4 26B A4B supports native tool calling, but consumers that need
    // a guaranteed tool call (e.g. strict agentic loops) can set this flag
    // so a silent plain-text response triggers a fallback to the next tier.
    const requireTools =
      (req.metadata as { fortune_require_tools?: boolean } | undefined)?.fortune_require_tools === true;
    if (requireTools) {
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
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

  const raw = await ai.run(model, input);
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
export function extractWorkersAiContent(result: WorkersAiChatResponse): {
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
      input: safeJsonParse(tc.function.arguments) ?? {},
    });
  }
  // Anthropic SDK requires at least one content block even on empty responses.
  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }

  const validToolCalls = toolCalls.filter((tc) => tc.function);
  const stopReason: AnthropicStopReason = validToolCalls.length > 0 ? "tool_use" : "end_turn";
  const promptTokens = result.usage?.prompt_tokens ?? estimatePromptTokens(input);
  const completionTokens =
    result.usage?.completion_tokens ??
    Math.ceil(
      (text.length +
        validToolCalls.reduce((acc, tc) => acc + tc.function.name.length + tc.function.arguments.length, 0)) /
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

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function estimatePromptTokens(input: WorkersAiChatRequest): number {
  const chars = input.messages.reduce((acc, m) => acc + (m.content?.length ?? 0), 0);
  return Math.ceil(chars / 4);
}

function randomId(): string {
  const buf = new Uint8Array(6);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Synthesize an Anthropic SSE stream from a fully-buffered Workers AI
 * tool-call response. We emit text deltas (if any) followed by one
 * tool_use content block per tool call, with the full args as a single
 * input_json_delta. Same event vocabulary the SDK expects.
 */
export function synthesizeToolStream(
  result: WorkersAiChatResponse,
  modelLabel: string,
): Response {
  const messageId = `msg_wa_${randomId()}`;
  const { text, toolCalls } = extractWorkersAiContent(result);

  const encoder = new TextEncoder();
  const downstream = new ReadableStream<Uint8Array>({
    start(controller) {
      const writeEvent = (event: string, data: unknown) => {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
        );
      };

      writeEvent("message_start", {
        type: "message_start",
        message: {
          id: messageId,
          type: "message",
          role: "assistant",
          model: modelLabel,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: result.usage?.prompt_tokens ?? 0,
            output_tokens: 0,
          },
        },
      });

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
          delta: { type: "input_json_delta", partial_json: tc.function.arguments },
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

  return new Response(downstream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}

export function streamWorkersAiAsAnthropic(
  upstream: ReadableStream,
  modelLabel: string,
): Response {
  const messageId = `msg_wa_${randomId()}`;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const downstream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const writeEvent = (event: string, data: unknown) => {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
        );
      };

      writeEvent("message_start", {
        type: "message_start",
        message: {
          id: messageId,
          type: "message",
          role: "assistant",
          model: modelLabel,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      });
      writeEvent("content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      });

      let outputTokenCount = 0;
      let buffer = "";
      let streamError = false;
      const reader = upstream.getReader();
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            const tail = decoder.decode();
            if (tail) buffer += tail;
            let cut: number;
            while ((cut = buffer.indexOf("\n\n")) !== -1) {
              const frame = buffer.slice(0, cut);
              buffer = buffer.slice(cut + 2);
              const payload = parseSseFrame(frame);
              if (!payload || payload === "[DONE]") continue;
              const delta = extractWorkersAiDelta(payload);
              if (!delta) continue;
              outputTokenCount += Math.max(1, Math.ceil(delta.length / 4));
              writeEvent("content_block_delta", {
                type: "content_block_delta",
                index: 0,
                delta: { type: "text_delta", text: delta },
              });
            }
            if (buffer.trim()) {
              const payload = parseSseFrame(buffer.trim());
              if (payload && payload !== "[DONE]") {
                const delta = extractWorkersAiDelta(payload);
                if (delta) {
                  outputTokenCount += Math.max(1, Math.ceil(delta.length / 4));
                  writeEvent("content_block_delta", {
                    type: "content_block_delta",
                    index: 0,
                    delta: { type: "text_delta", text: delta },
                  });
                }
              }
            }
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          let cut: number;
          while ((cut = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, cut);
            buffer = buffer.slice(cut + 2);
            const payload = parseSseFrame(frame);
            if (!payload) continue;
            if (payload === "[DONE]") continue;
            const delta = extractWorkersAiDelta(payload);
            if (!delta) continue;
            outputTokenCount += Math.max(1, Math.ceil(delta.length / 4));
            writeEvent("content_block_delta", {
              type: "content_block_delta",
              index: 0,
              delta: { type: "text_delta", text: delta },
            });
          }
        }
      } catch (err) {
        streamError = true;
        writeEvent("error", {
          type: "error",
          error: {
            type: "api_error",
            message: err instanceof Error ? err.message : "Workers AI stream failed",
          },
        });
      } finally {
        reader.releaseLock();
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

  return new Response(downstream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}

function parseSseFrame(frame: string): string | null {
  const lines = frame.split("\n");
  for (const line of lines) {
    if (line.startsWith("data:")) return line.slice(5).trim();
  }
  return null;
}

export function extractWorkersAiDelta(jsonPayload: string): string | null {
  if (!jsonPayload) return null;
  try {
    const obj = JSON.parse(jsonPayload) as { response?: unknown };
    if (typeof obj.response === "string") return obj.response;
  } catch {
    // ignore
  }
  return null;
}
