/**
 * Anthropic SSE synthesizer. Given a complete `AnthropicMessageResponse`
 * (already in JSON shape), emit the equivalent Anthropic SSE event
 * stream. Used by the cache layer to serve streaming requests from
 * non-streaming cached entries, and by any future code path that has a
 * full message but needs to deliver it as a stream.
 *
 * Why this exists: the Anthropic SDK's `.stream()` method requires the
 * SSE protocol; it can't consume a raw JSON message. Most consumer
 * traffic is `.stream()` (the SDK's default for chat UIs). Without
 * this synthesizer, the cache would only serve non-streaming requests
 * — missing the bulk of traffic.
 *
 * Event vocabulary (exact match of Anthropic's wire format):
 *   message_start         { message: { id, role, model, content:[], stop_reason:null, ... } }
 *   content_block_start   { index, content_block: { type:"text"|"tool_use", ... } }
 *   content_block_delta   { index, delta: { type:"text_delta", text } | { type:"input_json_delta", partial_json } }
 *   content_block_stop    { index }
 *   message_delta         { delta: { stop_reason, stop_sequence }, usage: {...} }
 *   message_stop          {}
 */

import type { AnthropicMessageResponse } from "./types.js";

const encoder = new TextEncoder();

/** Returns a writer that emits one named SSE event onto the stream. */
export function sseEventWriter(
  controller: ReadableStreamDefaultController<Uint8Array>,
): (event: string, data: unknown) => void {
  return (event, data) => {
    controller.enqueue(
      encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
    );
  };
}

/**
 * message_start: the envelope without content. The Anthropic SDK's stream
 * parser uses it to initialize the message that later events fill in.
 * output_tokens starts at 0; the final count arrives via message_delta.
 */
export function messageStartEvent(
  id: string,
  model: string,
  inputTokens = 0,
): unknown {
  return {
    type: "message_start",
    message: {
      id,
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    },
  };
}

/** Wrap an SSE byte stream in a response with no-buffering headers. */
export function sseResponse(body: ReadableStream<Uint8Array>): Response {
  return new Response(body, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}

/** The payload of the first `data:` line in one SSE frame, if any. */
export function parseSseDataLine(frame: string): string | null {
  const lines = frame.split("\n");
  for (const line of lines) {
    if (line.startsWith("data:")) return line.slice(5).trim();
  }
  return null;
}

/** 12 random hex characters for synthesized message and tool ids. */
export function randomId(): string {
  const buf = new Uint8Array(6);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => b.toString(16).padStart(2, "0")).join("");
}

export function synthesizeAnthropicSSE(message: AnthropicMessageResponse): Response {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const writeEvent = sseEventWriter(controller);

      writeEvent(
        "message_start",
        messageStartEvent(message.id, message.model, message.usage.input_tokens),
      );

      // Per-block: open, emit content as a single delta, close. We don't
      // chunk the text into multiple small deltas because (a) the cache
      // hit case is microsecond-fast anyway so there's no UX benefit to
      // artificial chunking, and (b) splitting unicode safely adds
      // surface area without benefit.
      message.content.forEach((block, index) => {
        if (block.type === "text") {
          writeEvent("content_block_start", {
            type: "content_block_start",
            index,
            content_block: { type: "text", text: "" },
          });
          if (block.text) {
            writeEvent("content_block_delta", {
              type: "content_block_delta",
              index,
              delta: { type: "text_delta", text: block.text },
            });
          }
          writeEvent("content_block_stop", { type: "content_block_stop", index });
        } else if (block.type === "tool_use") {
          writeEvent("content_block_start", {
            type: "content_block_start",
            index,
            content_block: { type: "tool_use", id: block.id, name: block.name, input: {} },
          });
          const argsJson = JSON.stringify(block.input ?? {});
          writeEvent("content_block_delta", {
            type: "content_block_delta",
            index,
            delta: { type: "input_json_delta", partial_json: argsJson },
          });
          writeEvent("content_block_stop", { type: "content_block_stop", index });
        }
      });

      writeEvent("message_delta", {
        type: "message_delta",
        delta: {
          stop_reason: message.stop_reason,
          stop_sequence: message.stop_sequence,
        },
        usage: {
          output_tokens: message.usage.output_tokens,
          input_tokens: message.usage.input_tokens,
        },
      });

      writeEvent("message_stop", { type: "message_stop" });

      controller.close();
    },
  });

  return sseResponse(stream);
}
