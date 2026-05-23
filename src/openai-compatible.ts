/**
 * Generic OpenAI-compatible chat-completions adapter. Used by every
 * backend that speaks the canonical OpenAI `/v1/chat/completions` shape
 * over HTTP — Groq, OpenRouter, Cerebras, GitHub Models, Together,
 * Fireworks, etc.
 *
 * The Anthropic→OpenAI *request* translation is shared with the Workers
 * AI path via `buildWorkersAiInput` (the wire shape is identical for the
 * request body). The *response* translation is provider-specific because
 * the canonical OpenAI HTTP response has `choices: [{message, finish_reason}]`,
 * whereas the Workers AI binding returns a flat `{response, tool_calls}`.
 *
 * Streaming is the canonical OpenAI SSE protocol:
 *   data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}
 *   data: {"choices":[{"index":0,"delta":{"tool_calls":[{...}]}}]}
 *   data: [DONE]
 *
 * Tool-call deltas are tricky: OpenAI emits `tool_calls[i].function.arguments`
 * as a stream of *string fragments* (partial JSON), with the `id` and
 * `name` arriving only on the first chunk for that index. We accumulate
 * per-index state and re-emit as Anthropic `input_json_delta` events on
 * the corresponding tool_use content block.
 */

import type {
  AnthropicMessagesRequest,
  AnthropicMessageResponse,
  AnthropicResponseContentBlock,
  AnthropicStopReason,
  OpenAIToolCall,
} from "./types.js";
import { buildWorkersAiInput } from "./workers-ai.js";

/**
 * Static config for an OpenAI-compatible backend. The headers map gets
 * merged into the request — OpenRouter wants `HTTP-Referer` and
 * `X-Title`, GitHub Models wants an `api-version`, etc.
 */
export interface OpenAICompatibleConfig {
  /** Display label used in `model:` on the response and in errors. */
  label: string;
  /** Full URL of the chat-completions endpoint. e.g. https://api.groq.com/openai/v1/chat/completions */
  url: string;
  apiKey: string;
  model: string;
  /** Optional extra headers (auth scheme is `Authorization: Bearer <key>` by default). */
  extraHeaders?: Record<string, string>;
}

interface OpenAIChoice {
  index: number;
  message: {
    role: "assistant";
    content: string | null;
    tool_calls?: OpenAIToolCall[];
  };
  finish_reason?: string;
}

interface OpenAIChatResponse {
  id?: string;
  choices?: OpenAIChoice[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  error?: { message?: string; type?: string; code?: string };
}

interface OpenAIStreamDelta {
  role?: string;
  content?: string | null;
  tool_calls?: Array<{
    index: number;
    id?: string;
    type?: string;
    function?: { name?: string; arguments?: string };
  }>;
}

interface OpenAIStreamChunk {
  id?: string;
  choices?: Array<{
    index?: number;
    delta?: OpenAIStreamDelta;
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

/**
 * Run the request against the configured OpenAI-compatible endpoint and
 * return an Anthropic-shaped Response (JSON message body or SSE stream).
 */
export async function callOpenAICompatible(
  config: OpenAICompatibleConfig,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  // Reuse the Anthropic→OpenAI request translator from the Workers AI
  // path. It produces the canonical chat-completions body shape.
  const body = buildWorkersAiInput(req);
  // The body needs the model name — Workers AI takes that as a separate
  // arg to the binding, but HTTP endpoints expect it inline.
  const httpBody = { ...body, model: config.model };

  const wantsStream = Boolean(body.stream);

  const headers: Record<string, string> = {
    "content-type": "application/json",
    authorization: `Bearer ${config.apiKey}`,
    ...config.extraHeaders,
  };

  const upstream = await fetch(config.url, {
    method: "POST",
    headers,
    body: JSON.stringify(httpBody),
  });

  if (!upstream.ok) {
    const errText = await upstream.text().catch(() => "");
    throw new Error(`${config.label} ${upstream.status}: ${errText.slice(0, 400)}`);
  }

  if (wantsStream) {
    if (!upstream.body) {
      throw new Error(`${config.label} stream had no body`);
    }
    return streamOpenAIAsAnthropic(upstream.body, req.model || config.model, config.label);
  }

  const parsed = (await upstream.json()) as OpenAIChatResponse;
  if (parsed.error) {
    throw new Error(`${config.label} error: ${parsed.error.message ?? JSON.stringify(parsed.error)}`);
  }
  const anthropicShape = openAIToAnthropicMessage(parsed, req.model || config.model);

  // Tool-silence detection. Matches the same opt-in semantics workers-ai
  // and gemini use: callers that require a tool call set
  // metadata.fortune_require_tools=true; a text-only response then
  // throws so the dispatcher falls through to the next tier.
  const requireTools =
    Array.isArray(req.tools) &&
    req.tools.length > 0 &&
    (req.metadata as { fortune_require_tools?: boolean } | undefined)?.fortune_require_tools === true;
  if (requireTools && anthropicShape.stop_reason !== "tool_use") {
    throw new Error(
      `${config.label} returned no tool calls though tools were declared and fortune_require_tools=true. ` +
        "Falling through to next tier.",
    );
  }

  return new Response(JSON.stringify(anthropicShape), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

/**
 * Translate a non-streaming OpenAI chat-completions response into the
 * Anthropic message shape consumers expect.
 */
export function openAIToAnthropicMessage(
  resp: OpenAIChatResponse,
  modelLabel: string,
): AnthropicMessageResponse {
  const choice = resp.choices?.[0];
  const msg = choice?.message;
  const text = typeof msg?.content === "string" ? msg.content : "";
  const toolCalls = Array.isArray(msg?.tool_calls) ? msg.tool_calls : [];

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
  if (content.length === 0) {
    content.push({ type: "text", text: "" });
  }

  const validToolCalls = toolCalls.filter((tc) => tc.function);
  let stopReason: AnthropicStopReason = "end_turn";
  if (validToolCalls.length > 0) {
    stopReason = "tool_use";
  } else if (choice?.finish_reason === "length") {
    stopReason = "max_tokens";
  } else if (choice?.finish_reason === "stop" || choice?.finish_reason === undefined) {
    stopReason = "end_turn";
  } else if (choice?.finish_reason === "tool_calls" || choice?.finish_reason === "function_call") {
    stopReason = "tool_use";
  }

  return {
    id: resp.id ?? `msg_oai_${randomId()}`,
    type: "message",
    role: "assistant",
    model: modelLabel,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: resp.usage?.prompt_tokens ?? 0,
      output_tokens: resp.usage?.completion_tokens ?? 0,
    },
  };
}

/**
 * Translate an OpenAI SSE stream into the Anthropic SSE stream shape.
 *
 * Tool-call streaming: OpenAI emits the same tool call across multiple
 * chunks. The first chunk for a given `index` carries `id` and
 * `function.name`; subsequent chunks add `function.arguments` fragments.
 * We accumulate per-index state and re-emit the cleaned-up shape as
 * Anthropic `content_block_*` events.
 */
export function streamOpenAIAsAnthropic(
  upstream: ReadableStream<Uint8Array>,
  modelLabel: string,
  providerLabel: string,
): Response {
  const messageId = `msg_oai_${randomId()}`;
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

      // Block layout: text block (if any) is index 0; each tool_call gets
      // its own block. We open the text block lazily on first content,
      // and we open tool_use blocks lazily on first delta for that
      // tool_call index. OpenAI tool_call indices are tool-call ordinals
      // within the message, not Anthropic content_block indices, so we
      // maintain a separate map.
      let textBlockIndex: number | null = null;
      let nextBlockIndex = 0;
      let emittedToolUse = false;
      let finishReason: string | undefined;
      let inputTokens = 0;
      let outputTokens = 0;
      let streamError = false;
      let buffer = "";

      interface ToolCallState {
        blockIndex: number;
        id: string;
        name: string;
        argsBuf: string;
      }
      const toolByIndex = new Map<number, ToolCallState>();

      const openTextBlock = () => {
        if (textBlockIndex !== null) return;
        textBlockIndex = nextBlockIndex++;
        writeEvent("content_block_start", {
          type: "content_block_start",
          index: textBlockIndex,
          content_block: { type: "text", text: "" },
        });
      };

      const closeTextBlock = () => {
        if (textBlockIndex === null) return;
        writeEvent("content_block_stop", { type: "content_block_stop", index: textBlockIndex });
        textBlockIndex = null;
      };

      const ensureToolBlock = (
        oaiIndex: number,
        meta: { id?: string; name?: string },
      ): ToolCallState => {
        let state = toolByIndex.get(oaiIndex);
        if (state) {
          // Update id/name if they arrived in a later chunk (rare but
          // some providers split the metadata across multiple deltas).
          if (meta.id && !state.id) state.id = meta.id;
          if (meta.name && !state.name) state.name = meta.name;
          return state;
        }
        // Close the text block before opening the first tool block to
        // keep Anthropic's content_block ordering well-formed.
        closeTextBlock();
        const id = meta.id ?? `toolu_${randomId()}`;
        const name = meta.name ?? "";
        state = { blockIndex: nextBlockIndex++, id, name, argsBuf: "" };
        toolByIndex.set(oaiIndex, state);
        emittedToolUse = true;
        writeEvent("content_block_start", {
          type: "content_block_start",
          index: state.blockIndex,
          content_block: { type: "tool_use", id, name, input: {} },
        });
        return state;
      };

      const reader = upstream.getReader();
      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            const tail = decoder.decode();
            if (tail) buffer += tail;
            // Flush any remaining frames.
            let cut: number;
            while ((cut = buffer.indexOf("\n\n")) !== -1) {
              processFrame(buffer.slice(0, cut));
              buffer = buffer.slice(cut + 2);
            }
            if (buffer.trim()) processFrame(buffer.trim());
            buffer = "";
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          let cut: number;
          while ((cut = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, cut);
            buffer = buffer.slice(cut + 2);
            processFrame(frame);
          }
        }
      } catch (err) {
        streamError = true;
        writeEvent("error", {
          type: "error",
          error: {
            type: "api_error",
            message: err instanceof Error ? err.message : `${providerLabel} stream failed`,
          },
        });
      } finally {
        reader.releaseLock();
        if (!streamError) {
          // Close any open blocks.
          for (const state of toolByIndex.values()) {
            writeEvent("content_block_stop", { type: "content_block_stop", index: state.blockIndex });
          }
          closeTextBlock();

          let stopReason: AnthropicStopReason = "end_turn";
          if (emittedToolUse) {
            stopReason = "tool_use";
          } else if (finishReason === "length") {
            stopReason = "max_tokens";
          } else if (finishReason === "tool_calls" || finishReason === "function_call") {
            stopReason = "tool_use";
          }
          writeEvent("message_delta", {
            type: "message_delta",
            delta: { stop_reason: stopReason, stop_sequence: null },
            usage: { output_tokens: outputTokens, input_tokens: inputTokens },
          });
          writeEvent("message_stop", { type: "message_stop" });
        }
        try {
          controller.close();
        } catch {
          // already closed
        }
      }

      function processFrame(frame: string) {
        const payload = parseSseDataLine(frame);
        if (!payload) return;
        if (payload === "[DONE]") return;
        let parsed: OpenAIStreamChunk;
        try {
          parsed = JSON.parse(payload) as OpenAIStreamChunk;
        } catch {
          return;
        }

        // Some providers (OpenRouter) emit usage on the final chunk in a
        // top-level `usage` field; others put it on a separate trailing
        // chunk with empty choices. Both are handled.
        if (parsed.usage) {
          if (typeof parsed.usage.prompt_tokens === "number") inputTokens = parsed.usage.prompt_tokens;
          if (typeof parsed.usage.completion_tokens === "number") outputTokens = parsed.usage.completion_tokens;
        }

        const choice = parsed.choices?.[0];
        if (!choice) return;
        if (choice.finish_reason) finishReason = choice.finish_reason;

        const delta = choice.delta;
        if (!delta) return;

        if (typeof delta.content === "string" && delta.content.length > 0) {
          openTextBlock();
          // textBlockIndex is non-null after openTextBlock.
          const idx = textBlockIndex as number;
          outputTokens += Math.max(1, Math.ceil(delta.content.length / 4));
          writeEvent("content_block_delta", {
            type: "content_block_delta",
            index: idx,
            delta: { type: "text_delta", text: delta.content },
          });
        }

        if (Array.isArray(delta.tool_calls)) {
          for (const tc of delta.tool_calls) {
            const oaiIndex = typeof tc.index === "number" ? tc.index : 0;
            const state = ensureToolBlock(oaiIndex, {
              id: tc.id,
              name: tc.function?.name,
            });
            const argsFragment = tc.function?.arguments;
            if (typeof argsFragment === "string" && argsFragment.length > 0) {
              state.argsBuf += argsFragment;
              writeEvent("content_block_delta", {
                type: "content_block_delta",
                index: state.blockIndex,
                delta: { type: "input_json_delta", partial_json: argsFragment },
              });
            }
          }
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

function parseSseDataLine(frame: string): string | null {
  const lines = frame.split("\n");
  for (const line of lines) {
    if (line.startsWith("data:")) return line.slice(5).trim();
  }
  return null;
}

function safeJsonParse(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function randomId(): string {
  const buf = new Uint8Array(6);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}
