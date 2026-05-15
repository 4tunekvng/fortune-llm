/**
 * Workers AI path. Translates an Anthropic Messages request into a
 * Llama-style chat-completions call, then translates the response (or
 * the streamed chunks) back into Anthropic's message / SSE format so
 * @anthropic-ai/sdk on the consumer side parses it transparently.
 *
 * Limitations the router enforces *before* we get here:
 *   - no tools (Llama tool-call shape is not 1:1 Anthropic's)
 *   - no images (text-only path)
 *   - moderate context length
 *
 * This file therefore only handles the text-in / text-out shape.
 */

import type {
  AnthropicMessagesRequest,
  AnthropicMessageResponse,
  AnthropicMessage,
  AnthropicContentBlock,
  WorkersAiChatRequest,
  WorkersAiChatResponse,
} from "./types.js";

/**
 * Loosely-typed view of the Cloudflare Workers AI binding. The real
 * binding type ships from `@cloudflare/workers-types` but its return
 * type is `unknown` per-model, so we narrow here.
 */
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
        // We won't see tool_use blocks here (router excludes them) but a
        // user message *can* still include a tool_result that references
        // a prior assistant — which we drop to text on a best-effort basis.
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

export function buildWorkersAiInput(req: AnthropicMessagesRequest): WorkersAiChatRequest {
  const messages: WorkersAiChatRequest["messages"] = [];

  // Anthropic's `system` is a top-level string (or content blocks). Llama
  // expects a system message at the head.
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
    messages.push({ role: m.role, content: flattenContent(m.content) });
  }

  return {
    messages,
    max_tokens: clampMaxTokens(req.max_tokens),
    temperature: typeof req.temperature === "number" ? req.temperature : undefined,
    top_p: typeof req.top_p === "number" ? req.top_p : undefined,
    top_k: typeof req.top_k === "number" ? req.top_k : undefined,
    stream: Boolean(req.stream),
  };
}

function clampMaxTokens(requested: number): number {
  // Workers AI Llama 4 Scout supports up to 8192 output tokens; let the
  // platform surface its own error for over-cap requests but provide a
  // sane default if the caller didn't ask for anything specific.
  if (!Number.isFinite(requested) || requested <= 0) return 1024;
  return requested;
}

export async function callWorkersAi(
  ai: AiBinding,
  model: string,
  req: AnthropicMessagesRequest,
): Promise<Response> {
  const input = buildWorkersAiInput(req);
  const raw = await ai.run(model, input);
  if (input.stream) {
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

export function workersAiToAnthropicMessage(
  result: WorkersAiChatResponse,
  modelLabel: string,
  input: WorkersAiChatRequest,
): AnthropicMessageResponse {
  const text = typeof result.response === "string" ? result.response : "";
  const promptTokens = result.usage?.prompt_tokens ?? estimatePromptTokens(input);
  const completionTokens = result.usage?.completion_tokens ?? Math.ceil(text.length / 4);
  return {
    id: `msg_wa_${randomId()}`,
    type: "message",
    role: "assistant",
    model: modelLabel,
    content: [{ type: "text", text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
    },
  };
}

function estimatePromptTokens(input: WorkersAiChatRequest): number {
  const chars = input.messages.reduce((acc, m) => acc + m.content.length, 0);
  return Math.ceil(chars / 4);
}

function randomId(): string {
  // 12 hex chars is plenty for a request-scoped id.
  const buf = new Uint8Array(6);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Translate a Workers AI streaming response (newline-delimited SSE-ish
 * chunks of `data: {"response": "..."}`) into a real Anthropic SSE
 * stream. The @anthropic-ai/sdk parser expects this exact event sequence:
 *
 *   message_start
 *   content_block_start (index 0, text)
 *   content_block_delta (deltas)+
 *   content_block_stop
 *   message_delta (with stop_reason)
 *   message_stop
 */
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
            // Flush any remaining bytes held by the stateful TextDecoder
            // (important for multi-byte UTF-8 characters split across chunks).
            const tail = decoder.decode();
            if (tail) buffer += tail;
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          // Workers AI emits "data: {...}\n\n" frames (SSE-style). Split
          // on the blank-line frame boundary and process each frame.
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
        controller.close();
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
    // Frame wasn't JSON — ignore.
  }
  return null;
}
