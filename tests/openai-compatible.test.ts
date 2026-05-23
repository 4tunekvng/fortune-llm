import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  callOpenAICompatible,
  openAIToAnthropicMessage,
  streamOpenAIAsAnthropic,
} from "../src/openai-compatible.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

const baseReq = (overrides: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest => ({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "hi" }],
  ...overrides,
});

describe("openAIToAnthropicMessage", () => {
  it("translates a plain text response", () => {
    const out = openAIToAnthropicMessage(
      {
        id: "chatcmpl-123",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "hello there" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 7, completion_tokens: 3 },
      },
      "llama-3.3-70b-versatile",
    );
    expect(out.content).toEqual([{ type: "text", text: "hello there" }]);
    expect(out.stop_reason).toBe("end_turn");
    expect(out.model).toBe("llama-3.3-70b-versatile");
    expect(out.usage).toEqual({ input_tokens: 7, output_tokens: 3 });
    expect(out.id).toBe("chatcmpl-123");
  });

  it("translates a tool-use response", () => {
    const out = openAIToAnthropicMessage(
      {
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_abc",
                  type: "function",
                  function: { name: "search", arguments: '{"q":"hi"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      },
      "llama-3.3-70b-versatile",
    );
    expect(out.stop_reason).toBe("tool_use");
    expect(out.content).toEqual([
      { type: "tool_use", id: "call_abc", name: "search", input: { q: "hi" } },
    ]);
  });

  it("emits an empty text block when the response is empty (Anthropic SDK requirement)", () => {
    const out = openAIToAnthropicMessage(
      { choices: [{ index: 0, message: { role: "assistant", content: "" }, finish_reason: "stop" }] },
      "llama-3.3-70b-versatile",
    );
    expect(out.content).toEqual([{ type: "text", text: "" }]);
  });

  it("maps finish_reason=length to stop_reason=max_tokens", () => {
    const out = openAIToAnthropicMessage(
      { choices: [{ index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "length" }] },
      "test",
    );
    expect(out.stop_reason).toBe("max_tokens");
  });

  it("emits both text and tool_use blocks in order", () => {
    const out = openAIToAnthropicMessage(
      {
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Let me check.",
              tool_calls: [
                { id: "t1", type: "function", function: { name: "search", arguments: "{}" } },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      },
      "test",
    );
    expect(out.content).toEqual([
      { type: "text", text: "Let me check." },
      { type: "tool_use", id: "t1", name: "search", input: {} },
    ]);
    expect(out.stop_reason).toBe("tool_use");
  });
});

describe("callOpenAICompatible", () => {
  // Use `any` here because vi.spyOn over the CF-typed global fetch
  // produces an intersection type the strict TS config can't narrow.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let fetchSpy: any;
  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });
  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("posts the OpenAI shape and returns Anthropic JSON for a non-streaming response", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: "chatcmpl-1",
          choices: [
            { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
        { status: 200, headers: { "content-type": "application/json" } },
      ),
    );

    const resp = await callOpenAICompatible(
      {
        label: "Groq",
        url: "https://api.groq.com/openai/v1/chat/completions",
        apiKey: "test-key",
        model: "llama-3.3-70b-versatile",
      },
      baseReq(),
    );
    expect(resp.status).toBe(200);

    const call = fetchSpy.mock.calls[0]!;
    const [reqUrl, init] = call as [string, RequestInit];
    expect(reqUrl).toBe("https://api.groq.com/openai/v1/chat/completions");
    expect((init.headers as Record<string, string>).authorization).toBe("Bearer test-key");
    const body = JSON.parse(init.body as string);
    expect(body.model).toBe("llama-3.3-70b-versatile");
    expect(body.messages).toEqual([{ role: "user", content: "hi" }]);

    const json = (await resp.json()) as { content: unknown };
    expect(json.content).toEqual([{ type: "text", text: "ok" }]);
  });

  it("merges extra headers (used by OpenRouter for HTTP-Referer/X-Title)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          choices: [
            { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
          ],
        }),
        { status: 200 },
      ),
    );
    await callOpenAICompatible(
      {
        label: "OpenRouter",
        url: "https://openrouter.ai/api/v1/chat/completions",
        apiKey: "or-key",
        model: "meta-llama/llama-3.3-70b-instruct:free",
        extraHeaders: { "HTTP-Referer": "https://example", "X-Title": "test" },
      },
      baseReq(),
    );
    const init = fetchSpy.mock.calls[0]![1] as RequestInit;
    const headers = init.headers as Record<string, string>;
    expect(headers["HTTP-Referer"]).toBe("https://example");
    expect(headers["X-Title"]).toBe("test");
    expect(headers.authorization).toBe("Bearer or-key");
  });

  it("throws with the upstream status when the provider returns non-2xx", async () => {
    fetchSpy.mockResolvedValueOnce(new Response("rate limited", { status: 429 }));
    await expect(
      callOpenAICompatible(
        { label: "Groq", url: "https://api.groq.com/openai/v1/chat/completions", apiKey: "k", model: "m" },
        baseReq(),
      ),
    ).rejects.toThrow(/Groq 429/);
  });

  it("treats {error:{...}} JSON bodies on 200 as errors (some providers use that shape)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ error: { message: "model not found" } }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    await expect(
      callOpenAICompatible(
        { label: "Groq", url: "https://api.groq.com/openai/v1/chat/completions", apiKey: "k", model: "m" },
        baseReq(),
      ),
    ).rejects.toThrow(/Groq error/);
  });

  it("falls through (throws) when fortune_require_tools=true and the response is text-only", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          choices: [
            { index: 0, message: { role: "assistant", content: "just talking" }, finish_reason: "stop" },
          ],
        }),
        { status: 200 },
      ),
    );
    await expect(
      callOpenAICompatible(
        { label: "Groq", url: "https://x", apiKey: "k", model: "m" },
        baseReq({
          tools: [{ name: "search", input_schema: { type: "object" } }],
          metadata: { fortune_require_tools: true },
        }),
      ),
    ).rejects.toThrow(/no tool calls/);
  });

  it("returns SSE for streaming requests", async () => {
    const sseLines = [
      'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}',
      'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}',
      'data: {"choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}',
      "data: [DONE]",
    ];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const encoder = new TextEncoder();
        for (const line of sseLines) {
          controller.enqueue(encoder.encode(line + "\n\n"));
        }
        controller.close();
      },
    });
    fetchSpy.mockResolvedValueOnce(
      new Response(stream, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      }),
    );

    const resp = await callOpenAICompatible(
      { label: "Groq", url: "https://x", apiKey: "k", model: "m" },
      baseReq({ stream: true }),
    );
    expect(resp.headers.get("content-type")).toMatch(/event-stream/);

    const text = await resp.text();
    expect(text).toContain('event: message_start');
    expect(text).toContain('event: content_block_start');
    expect(text).toContain('"text":"hello"');
    expect(text).toContain('"text":" world"');
    expect(text).toContain('event: content_block_stop');
    expect(text).toContain('event: message_delta');
    expect(text).toContain('"stop_reason":"end_turn"');
    expect(text).toContain('event: message_stop');
  });
});

describe("streamOpenAIAsAnthropic", () => {
  it("reconstructs tool-use deltas split across multiple SSE frames", async () => {
    // OpenAI emits tool_calls with id+name on the first chunk for that
    // index, then function.arguments fragments on subsequent chunks. The
    // stream translator has to accumulate them into one Anthropic
    // content_block_start + N input_json_delta + content_block_stop.
    const frames = [
      'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}',
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_xyz","type":"function","function":{"name":"search","arguments":""}}]}}]}',
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"q"}}]}}]}',
      'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\":\\"hi\\"}"}}]}}]}',
      'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
      "data: [DONE]",
    ];
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const encoder = new TextEncoder();
        for (const line of frames) controller.enqueue(encoder.encode(line + "\n\n"));
        controller.close();
      },
    });
    const resp = streamOpenAIAsAnthropic(stream, "llama-3.3-70b", "Groq");
    const text = await resp.text();

    // The tool_use block should be opened once with id+name, and the
    // arguments should arrive as input_json_delta fragments.
    expect(text).toMatch(/content_block_start[^\n]*"type":"tool_use"[^\n]*"id":"call_xyz"[^\n]*"name":"search"/);
    expect(text).toContain('"partial_json":"{\\"q"');
    expect(text).toContain('"partial_json":"\\":\\"hi\\"}"');
    expect(text).toContain('"stop_reason":"tool_use"');
  });

  it("handles content split across frame boundaries (partial SSE frame mid-chunk)", async () => {
    // Some upstream proxies cut SSE bytes mid-frame. The decoder must
    // buffer and reassemble.
    const encoder = new TextEncoder();
    const frame1 = 'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n';
    const frame2partial = '\ndata: {"choices":[{"index":0,"delta":{"content":" world"';
    const frame2rest = '},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(frame1));
        controller.enqueue(encoder.encode(frame2partial));
        controller.enqueue(encoder.encode(frame2rest));
        controller.close();
      },
    });
    const resp = streamOpenAIAsAnthropic(stream, "test", "test");
    const text = await resp.text();
    expect(text).toContain('"text":"hello"');
    expect(text).toContain('"text":" world"');
  });
});
