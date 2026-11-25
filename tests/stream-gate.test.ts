import { describe, expect, it } from "vitest";
import { holdUntilFirstWords } from "../src/stream-gate.js";

const encoder = new TextEncoder();

const event = (data: Record<string, unknown>) =>
  `event: ${String(data.type)}\ndata: ${JSON.stringify(data)}\n\n`;
const start = event({ type: "message_start", message: { content: [] } });
const words = (text: string) =>
  event({
    type: "content_block_delta",
    index: 0,
    delta: { type: "text_delta", text },
  });
const stop = event({ type: "message_stop" });
const failed = event({
  type: "error",
  error: { type: "api_error", message: "boom" },
});

/** A stream that sends `chunks` in order, then ends (or, with `hang`, stalls). */
function streamOf(chunks: string[], hang = false): Response {
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(c));
      if (!hang) controller.close();
    },
  });
  return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

const opts = (stopped: string[] = []) => ({
  idleMs: 20,
  signal: new AbortController().signal,
  stopUpstream: () => stopped.push("stopped"),
});

describe("holdUntilFirstWords", () => {
  it("replays every byte once the first words arrive", async () => {
    const sent = [start, words("Hello"), stop];

    const resp = await holdUntilFirstWords(streamOf(sent), "groq", opts());

    expect(await resp.text()).toBe(sent.join(""));
  });

  it("fails the tier on an error before its first words", async () => {
    const stopped: string[] = [];

    const upstream = streamOf([start, failed]);

    const held = holdUntilFirstWords(upstream, "gemini", opts(stopped));

    await expect(held).rejects.toThrow(/gemini stream failed before its first/);
    expect(stopped).toEqual(["stopped"]);
  });

  it("falls through when an error follows words in the same read", async () => {
    const upstream = streamOf([start + words("Let me") + failed]);

    const held = holdUntilFirstWords(upstream, "groq", opts());

    await expect(held).rejects.toThrow(/failed before its words were sent: boom/);
  });

  it("fails the tier when the stream ends before its first words", async () => {
    const held = holdUntilFirstWords(streamOf([start]), "groq", opts());

    await expect(held).rejects.toThrow(/ended before its first words/);
  });

  it("fails the tier on a reply that finishes blank", async () => {
    const upstream = streamOf([start, words("  "), stop]);

    const held = holdUntilFirstWords(upstream, "groq", opts());

    await expect(held).rejects.toThrow(/groq returned an empty reply/);
  });

  it("counts a tool call as first words", async () => {
    const tool = event({
      type: "content_block_start",
      index: 0,
      content_block: { type: "tool_use", id: "t1", name: "search", input: {} },
    });

    const upstream = streamOf([start, tool, stop]);

    const resp = await holdUntilFirstWords(upstream, "groq", opts());

    expect(await resp.text()).toContain('"name":"search"');
  });

  it("ends on an error and stops the upstream when quiet after words", async () => {
    const stopped: string[] = [];
    const resp = await holdUntilFirstWords(
      streamOf([start, words("Let me")], true),
      "workers-ai",
      opts(stopped),
    );

    const text = await resp.text();

    expect(text).toContain("Let me");
    expect(text).toContain("workers-ai went quiet for 20 ms");
    expect(stopped).toEqual(["stopped"]);
  });

  it("stops waiting when the tier's time limit fires while held", async () => {
    const limit = new AbortController();
    const held = holdUntilFirstWords(streamOf([start], true), "gemini", {
      ...opts(),
      signal: limit.signal,
    });

    limit.abort();

    await expect(held).rejects.toThrow(/ended before its first words/);
  });
});
