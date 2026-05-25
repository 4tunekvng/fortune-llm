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

export function synthesizeAnthropicSSE(message: AnthropicMessageResponse): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const writeEvent = (event: string, data: unknown): void => {
        controller.enqueue(
          encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
        );
      };

      // message_start: envelope without content. The Anthropic SDK's
      // stream parser uses this to initialize the message object that
      // later events accumulate into.
      writeEvent("message_start", {
        type: "message_start",
        message: {
          id: message.id,
          type: "message",
          role: "assistant",
          model: message.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: {
            input_tokens: message.usage.input_tokens,
            output_tokens: 0, // output_tokens accumulates as deltas arrive; final value comes via message_delta
          },
        },
      });

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

  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-cache, no-transform",
      "x-accel-buffering": "no",
    },
  });
}
