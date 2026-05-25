import { describe, it, expect } from "vitest";
import { synthesizeAnthropicSSE } from "../src/sse.js";
import type { AnthropicMessageResponse } from "../src/types.js";

/**
 * Helper: drain the synthesized SSE stream into the ordered list of
 * (event, data) pairs the consumer's SDK parser would see.
 */
async function drainSSE(resp: Response): Promise<Array<{ event: string; data: unknown }>> {
  const text = await resp.text();
  const out: Array<{ event: string; data: unknown }> = [];
  const frames = text.split("\n\n").filter((f) => f.trim());
  for (const frame of frames) {
    const lines = frame.split("\n");
    let event = "";
    let dataStr = "";
    for (const line of lines) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
    }
    if (event && dataStr) {
      out.push({ event, data: JSON.parse(dataStr) });
    }
  }
  return out;
}

const plainText: AnthropicMessageResponse = {
  id: "msg_abc",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: [{ type: "text", text: "Hello, world." }],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage: { input_tokens: 10, output_tokens: 5 },
};

const toolUse: AnthropicMessageResponse = {
  id: "msg_t1",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: [
    { type: "text", text: "Let me search." },
    { type: "tool_use", id: "toolu_xyz", name: "search", input: { q: "weather" } },
  ],
  stop_reason: "tool_use",
  stop_sequence: null,
  usage: { input_tokens: 20, output_tokens: 12 },
};

describe("synthesizeAnthropicSSE", () => {
  it("emits the canonical Anthropic stream event sequence for plain text", async () => {
    const events = await drainSSE(synthesizeAnthropicSSE(plainText));
    const eventNames = events.map((e) => e.event);
    expect(eventNames).toEqual([
      "message_start",
      "content_block_start",
      "content_block_delta",
      "content_block_stop",
      "message_delta",
      "message_stop",
    ]);
  });

  it("message_start carries the message envelope with empty content", async () => {
    const events = await drainSSE(synthesizeAnthropicSSE(plainText));
    const ms = events[0]?.data as { message: { id: string; model: string; content: unknown[]; usage: { input_tokens: number } } };
    expect(ms.message.id).toBe("msg_abc");
    expect(ms.message.model).toBe("claude-sonnet-4-6");
    expect(ms.message.content).toEqual([]);
    expect(ms.message.usage.input_tokens).toBe(10);
  });

  it("emits a single text_delta with the full text for a text block", async () => {
    const events = await drainSSE(synthesizeAnthropicSSE(plainText));
    const delta = events.find((e) => e.event === "content_block_delta")?.data as {
      delta: { type: string; text: string };
    };
    expect(delta.delta).toEqual({ type: "text_delta", text: "Hello, world." });
  });

  it("emits final message_delta with the right stop_reason and usage", async () => {
    const events = await drainSSE(synthesizeAnthropicSSE(plainText));
    const md = events.find((e) => e.event === "message_delta")?.data as {
      delta: { stop_reason: string };
      usage: { output_tokens: number };
    };
    expect(md.delta.stop_reason).toBe("end_turn");
    expect(md.usage.output_tokens).toBe(5);
  });

  it("synthesizes text + tool_use blocks in order with proper indices", async () => {
    const events = await drainSSE(synthesizeAnthropicSSE(toolUse));
    // Block 0 = text, Block 1 = tool_use
    const starts = events.filter((e) => e.event === "content_block_start");
    expect(starts).toHaveLength(2);
    expect((starts[0]?.data as { content_block: { type: string } }).content_block.type).toBe("text");
    expect((starts[1]?.data as { content_block: { type: string; id?: string; name?: string } }).content_block).toMatchObject({
      type: "tool_use",
      id: "toolu_xyz",
      name: "search",
    });
  });

  it("tool_use args arrive as input_json_delta with full JSON", async () => {
    const events = await drainSSE(synthesizeAnthropicSSE(toolUse));
    const toolDeltas = events.filter(
      (e) =>
        e.event === "content_block_delta" &&
        (e.data as { delta: { type: string } }).delta.type === "input_json_delta",
    );
    expect(toolDeltas).toHaveLength(1);
    expect((toolDeltas[0]?.data as { delta: { partial_json: string } }).delta.partial_json).toBe(
      JSON.stringify({ q: "weather" }),
    );
  });

  it("propagates tool_use stop_reason from the original message", async () => {
    const events = await drainSSE(synthesizeAnthropicSSE(toolUse));
    const md = events.find((e) => e.event === "message_delta")?.data as { delta: { stop_reason: string } };
    expect(md.delta.stop_reason).toBe("tool_use");
  });

  it("omits text_delta when the text block is empty (Anthropic edge case)", async () => {
    const empty: AnthropicMessageResponse = {
      ...plainText,
      content: [{ type: "text", text: "" }],
    };
    const events = await drainSSE(synthesizeAnthropicSSE(empty));
    const deltas = events.filter((e) => e.event === "content_block_delta");
    expect(deltas).toHaveLength(0);
    // But start/stop bookends are still emitted so consumer parsers
    // see a valid (empty) content block.
    const starts = events.filter((e) => e.event === "content_block_start");
    const stops = events.filter((e) => e.event === "content_block_stop");
    expect(starts).toHaveLength(1);
    expect(stops).toHaveLength(1);
  });

  it("sets the right Content-Type and disables proxy buffering", () => {
    const resp = synthesizeAnthropicSSE(plainText);
    expect(resp.headers.get("content-type")).toMatch(/event-stream/);
    expect(resp.headers.get("cache-control")).toContain("no-cache");
    expect(resp.headers.get("x-accel-buffering")).toBe("no");
  });
});
