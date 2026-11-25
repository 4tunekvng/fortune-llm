import { describe, it, expect } from "vitest";
import {
  buildWorkersAiInput,
  flattenContent,
  workersAiToAnthropicMessage,
  extractWorkersAiDelta,
  translateTools,
  translateToolChoice,
  streamWorkersAiAsAnthropic,
  callWorkersAi,
} from "../src/workers-ai.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

const WA_MODEL = "@cf/meta/llama-3.1-8b-instruct";
const WA_GEMMA = "@cf/google/gemma-4-26b-a4b-it";

describe("flattenContent", () => {
  it("returns string content unchanged", () => {
    expect(flattenContent("hello")).toBe("hello");
  });

  it("joins text blocks with newlines", () => {
    expect(
      flattenContent([
        { type: "text", text: "one" },
        { type: "text", text: "two" },
      ]),
    ).toBe("one\ntwo");
  });

  it("flattens tool_result blocks with string content", () => {
    expect(
      flattenContent([
        { type: "tool_result", tool_use_id: "abc", content: "result body" },
      ]),
    ).toBe("result body");
  });

  it("flattens tool_result blocks with array content", () => {
    expect(
      flattenContent([
        {
          type: "tool_result",
          tool_use_id: "abc",
          content: [{ type: "text", text: "a" }, { type: "text", text: "b" }],
        },
      ]),
    ).toBe("a\nb");
  });

  it("drops unknown blocks gracefully", () => {
    expect(
      flattenContent([
        { type: "text", text: "real text" },
        { type: "unknown_thing", payload: 42 },
      ]),
    ).toBe("real text");
  });
});

describe("buildWorkersAiInput", () => {
  it("prefixes a system message when system is a string", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: 100,
      system: "you are helpful",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(input.messages[0]).toEqual({ role: "system", content: "you are helpful" });
    expect(input.messages[1]).toEqual({ role: "user", content: "hi" });
  });

  it("flattens system content blocks", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: 100,
      system: [
        { type: "text", text: "line1" },
        { type: "text", text: "line2" },
      ],
      messages: [{ role: "user", content: "hi" }],
    } as AnthropicMessagesRequest);
    expect(input.messages[0]).toEqual({ role: "system", content: "line1\nline2" });
  });

  it("forwards stream + sampling params", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: 256,
      messages: [{ role: "user", content: "hi" }],
      temperature: 0.7,
      top_p: 0.95,
      top_k: 40,
      stream: true,
    });
    expect(input).toMatchObject({
      max_tokens: 256,
      temperature: 0.7,
      top_p: 0.95,
      top_k: 40,
      stream: true,
    });
  });

  it("forwards stop_sequences as the OpenAI stop field", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: 256,
      messages: [{ role: "user", content: "hi" }],
      stop_sequences: ["<END>", "\n\nHuman:"],
    });
    expect(input.stop).toEqual(["<END>", "\n\nHuman:"]);
  });

  it("omits stop when stop_sequences is empty", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: 256,
      messages: [{ role: "user", content: "hi" }],
      stop_sequences: [],
    });
    expect(input.stop).toBeUndefined();
  });

  it("clamps non-finite max_tokens to a sane default", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: Number.NaN,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(input.max_tokens).toBe(1024);
  });

  it("translates Anthropic tool_use blocks on an assistant turn to OpenAI tool_calls", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: 100,
      messages: [
        { role: "user", content: "search the web for cats" },
        {
          role: "assistant",
          content: [
            { type: "text", text: "Let me search." },
            { type: "tool_use", id: "toolu_abc", name: "web_search", input: { query: "cats" } },
          ],
        },
        {
          role: "user",
          content: [
            { type: "tool_result", tool_use_id: "toolu_abc", content: "found 1000 results" },
          ],
        },
      ],
    });
    const assistantMsg = input.messages.find((m) => m.role === "assistant");
    expect(assistantMsg).toBeDefined();
    expect(assistantMsg?.content).toBe("Let me search.");
    expect(assistantMsg?.tool_calls).toEqual([
      {
        id: "toolu_abc",
        type: "function",
        function: { name: "web_search", arguments: JSON.stringify({ query: "cats" }) },
      },
    ]);
    const toolMsg = input.messages.find((m) => m.role === "tool");
    expect(toolMsg).toEqual({
      role: "tool",
      tool_call_id: "toolu_abc",
      content: "found 1000 results",
    });
  });

  it("translates request-level tools[] declarations", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
      tools: [
        {
          name: "get_weather",
          description: "Get the weather in a city",
          input_schema: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      ],
    });
    expect(input.tools).toEqual([
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get the weather in a city",
          parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"],
          },
        },
      },
    ]);
  });
});

describe("translateTools", () => {
  it("returns undefined for empty / missing input", () => {
    expect(translateTools(undefined)).toBeUndefined();
    expect(translateTools([])).toBeUndefined();
  });

  it("fills a placeholder parameters object when input_schema is missing", () => {
    const out = translateTools([{ name: "noop", input_schema: undefined as unknown as Record<string, unknown> }]);
    expect(out?.[0]?.function.parameters).toEqual({ type: "object", properties: {} });
  });
});

describe("translateToolChoice", () => {
  it("maps Anthropic's tool_choice shapes to OpenAI", () => {
    expect(translateToolChoice({ type: "auto" })).toBe("auto");
    expect(translateToolChoice({ type: "none" })).toBe("none");
    expect(translateToolChoice({ type: "any" })).toBe("required");
    expect(translateToolChoice({ type: "tool", name: "search" })).toEqual({
      type: "function",
      function: { name: "search" },
    });
  });
});

describe("workersAiToAnthropicMessage", () => {
  it("wraps text response in an Anthropic Message shape", () => {
    const msg = workersAiToAnthropicMessage(
      { response: "hello world", usage: { prompt_tokens: 5, completion_tokens: 2 } },
      "claude-sonnet-5",
      { messages: [{ role: "user", content: "hi" }], max_tokens: 100 },
      WA_MODEL,
    );
    expect(msg.type).toBe("message");
    expect(msg.role).toBe("assistant");
    expect(msg.model).toBe("claude-sonnet-5");
    expect(msg.content).toEqual([{ type: "text", text: "hello world" }]);
    expect(msg.stop_reason).toBe("end_turn");
    expect(msg.usage).toEqual({ input_tokens: 5, output_tokens: 2 });
  });

  it("converts Workers AI tool_calls to Anthropic tool_use content blocks with stop_reason=tool_use", () => {
    const msg = workersAiToAnthropicMessage(
      {
        response: "Calling tool.",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "get_weather", arguments: JSON.stringify({ city: "Tokyo" }) },
          },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      },
      "claude-sonnet-5",
      { messages: [{ role: "user", content: "weather in Tokyo?" }], max_tokens: 100 },
      WA_MODEL,
    );
    expect(msg.stop_reason).toBe("tool_use");
    expect(msg.content).toEqual([
      { type: "text", text: "Calling tool." },
      { type: "tool_use", id: "call_1", name: "get_weather", input: { city: "Tokyo" } },
    ]);
  });

  it("coerces non-object tool_calls arguments to an empty object", () => {
    const msg = workersAiToAnthropicMessage(
      {
        tool_calls: [
          { id: "call_str", type: "function", function: { name: "f", arguments: '"hello"' } },
          { id: "call_arr", type: "function", function: { name: "g", arguments: "[1, 2]" } },
          { id: "call_num", type: "function", function: { name: "h", arguments: "42" } },
        ],
      },
      "claude-sonnet-5",
      { messages: [{ role: "user", content: "call" }], max_tokens: 100 },
      WA_MODEL,
    );
    expect(msg.content).toEqual([
      { type: "tool_use", id: "call_str", name: "f", input: {} },
      { type: "tool_use", id: "call_arr", name: "g", input: {} },
      { type: "tool_use", id: "call_num", name: "h", input: {} },
    ]);
  });

  it("estimates tokens when Workers AI omits usage", () => {
    const msg = workersAiToAnthropicMessage(
      { response: "abcd".repeat(10) },
      "claude-sonnet-5",
      { messages: [{ role: "user", content: "x".repeat(20) }], max_tokens: 100 },
      WA_MODEL,
    );
    expect(msg.usage.input_tokens).toBeGreaterThan(0);
    expect(msg.usage.output_tokens).toBe(10);
  });

  it("ensures at least one content block even for empty Workers AI responses", () => {
    const msg = workersAiToAnthropicMessage(
      {},
      "claude-sonnet-5",
      { messages: [{ role: "user", content: "hi" }], max_tokens: 100 },
      WA_MODEL,
    );
    expect(msg.content).toHaveLength(1);
    expect(msg.content[0]).toEqual({ type: "text", text: "" });
  });

  // Regression: gemma-4-26b returns OpenAI-compat shape. Without this
  // path the translator emitted empty text + non-zero output_tokens.
  it("handles the OpenAI-compat {choices[].message.content} shape (Gemma 4 et al)", () => {
    const msg = workersAiToAnthropicMessage(
      {
        choices: [
          {
            message: { role: "assistant", content: "hello from gemma" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 7, completion_tokens: 3 },
      },
      "claude-sonnet-5",
      { messages: [{ role: "user", content: "hi" }], max_tokens: 100 },
      WA_MODEL,
    );
    expect(msg.content).toEqual([{ type: "text", text: "hello from gemma" }]);
    expect(msg.stop_reason).toBe("end_turn");
    expect(msg.usage).toEqual({ input_tokens: 7, output_tokens: 3 });
  });

  it("extracts tool_calls from the OpenAI-compat choices shape", () => {
    const msg = workersAiToAnthropicMessage(
      {
        choices: [
          {
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_42",
                  type: "function",
                  function: { name: "get_weather", arguments: JSON.stringify({ city: "Tokyo" }) },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 12, completion_tokens: 4 },
      },
      "claude-sonnet-5",
      { messages: [{ role: "user", content: "weather in Tokyo?" }], max_tokens: 100 },
      WA_MODEL,
    );
    expect(msg.stop_reason).toBe("tool_use");
    expect(msg.content).toEqual([
      { type: "tool_use", id: "call_42", name: "get_weather", input: { city: "Tokyo" } },
    ]);
  });
});

describe("workersAiToAnthropicMessage on a reply the model did not finish", () => {
  const input = buildWorkersAiInput({
    model: "m",
    max_tokens: 64,
    messages: [{ role: "user", content: "hi" }],
  });

  const replyEnding = (content: string, finish: string) => ({
    choices: [{ message: { role: "assistant", content }, finish_reason: finish }],
  });

  it("throws on an unknown finish reason so the next tier answers", () => {
    const result = replyEnding("half", "content_filter");

    expect(() => workersAiToAnthropicMessage(result, "m", input, WA_MODEL)).toThrow(
      /Workers AI did not finish its reply \(content_filter\)/,
    );
  });

  it("throws when it promised a tool call and sent none", () => {
    const result = replyEnding("", "tool_calls");

    expect(() => workersAiToAnthropicMessage(result, "m", input, WA_MODEL)).toThrow(
      /did not finish/,
    );
  });
});

describe("callWorkersAi with tools", () => {
  const toolReq: AnthropicMessagesRequest = {
    model: "m",
    max_tokens: 64,
    stream: true,
    messages: [{ role: "user", content: "hi" }],
    tools: [{ name: "search", input_schema: { type: "object" } }],
  };
  const aiReturning = (result: unknown) => ({ run: async () => result });
  const replyEnding = (content: string, finish: string) => ({
    choices: [{ message: { role: "assistant", content }, finish_reason: finish }],
  });

  it("throws on a cut-off reply before anything is streamed", async () => {
    const ai = aiReturning(replyEnding("half", "content_filter"));

    await expect(callWorkersAi(ai, "m", toolReq)).rejects.toThrow(/did not finish/);
  });

  it("throws on a blank reply before anything is streamed", async () => {
    const ai = aiReturning(replyEnding("", "stop"));

    await expect(callWorkersAi(ai, "m", toolReq)).rejects.toThrow(/empty reply/);
  });

  it("drops Gemma 4's ghost thought line before a tool call", async () => {
    const ai = aiReturning({
      choices: [{
        message: {
          role: "assistant",
          content: "thought\nDrafting it now.",
          tool_calls: [{
            id: "c1",
            type: "function",
            function: { name: "search", arguments: "{}" },
          }],
        },
        finish_reason: "tool_calls",
      }],
    });

    const text = await (await callWorkersAi(ai, WA_GEMMA, toolReq)).text();

    expect(text).toContain('"text":"Drafting it now."');
    expect(text).not.toContain("thought");
  });

  it("falls through when the reply is only the ghost thought line", async () => {
    const ai = aiReturning(replyEnding("thought\n", "stop"));

    const reply = callWorkersAi(ai, WA_GEMMA, toolReq);

    await expect(reply).rejects.toThrow(/empty reply/);
  });

  it("streams a finished tool call back as tool_use", async () => {
    const ai = aiReturning({
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "search", arguments: '{"q":"a"}' },
        },
      ],
    });

    const text = await (await callWorkersAi(ai, "m", toolReq)).text();

    expect(text).toContain('"partial_json":"{\\"q\\":\\"a\\"}"');
    expect(text).toContain('"stop_reason":"tool_use"');
  });
});

describe("extractWorkersAiDelta", () => {
  it("returns the response field for normal frames", () => {
    expect(extractWorkersAiDelta('{"response":"abc"}').text).toBe("abc");
  });

  it("returns null for malformed JSON", () => {
    expect(extractWorkersAiDelta("not json").text).toBe(null);
  });

  it("returns null when response is not a string", () => {
    expect(extractWorkersAiDelta('{"response": 42}').text).toBe(null);
  });

  it("reads the OpenAI-style delta that Gemma 4 streams", () => {
    const frame =
      '{"object":"chat.completion.chunk","choices":[{"index":0,' +
      '"delta":{"content":"Pick one thing."},"finish_reason":null}]}';

    expect(extractWorkersAiDelta(frame).text).toBe("Pick one thing.");
  });

  it("never passes on the model's reasoning as reply text", () => {
    const frame = '{"choices":[{"index":0,"delta":{"reasoning":"   Goal"}}]}';

    expect(extractWorkersAiDelta(frame).text).toBe(null);
  });
});

describe("buildWorkersAiInput — image block translation", () => {
  it("translates base64 image blocks to OpenAI data-URL vision format", () => {
    const req: AnthropicMessagesRequest = {
      model: "claude-sonnet-5",
      max_tokens: 256,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "describe this" },
            {
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: "abc123",
              },
            },
          ],
        },
      ],
    };
    const built = buildWorkersAiInput(req);
    expect(built.messages).toHaveLength(1);
    const msg = built.messages[0]!;
    expect(Array.isArray(msg.content)).toBe(true);
    const parts = msg.content as Array<{ type: string; text?: string;
      image_url?: { url: string } }>;
    expect(parts[0]).toEqual({ type: "text", text: "describe this" });
    expect(parts[1]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,abc123" },
    });
  });

  it("translates URL image sources as a plain image_url reference", () => {
    const req: AnthropicMessagesRequest = {
      model: "claude-sonnet-5",
      max_tokens: 256,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image",
              source: {
                type: "url",
                url: "https://example.com/photo.jpg",
              },
            },
          ],
        },
      ],
    };
    const built = buildWorkersAiInput(req);
    const parts = built.messages[0]!.content as Array<{
      type: string; image_url?: { url: string } }>;
    expect(parts).toHaveLength(1);
    expect(parts[0]).toEqual({
      type: "image_url",
      image_url: { url: "https://example.com/photo.jpg" },
    });
  });

  it("produces a plain string for text-only user messages (no regression)", () => {
    const req: AnthropicMessagesRequest = {
      model: "claude-sonnet-5",
      max_tokens: 256,
      messages: [{ role: "user", content: "hello" }],
    };
    const built = buildWorkersAiInput(req);
    expect(typeof built.messages[0]!.content).toBe("string");
  });
});

describe("streamWorkersAiAsAnthropic", () => {
  const upstreamOf = (chunks: string[]) => {
    const encoder = new TextEncoder();
    return new ReadableStream<Uint8Array>({
      start(controller) {
        for (const c of chunks) controller.enqueue(encoder.encode(c));
        controller.close();
      },
    });
  };

  it("emits one text block with a delta per response frame", async () => {
    const upstream = upstreamOf([
      'data: {"response":"hello"}\n\n',
      'data: {"response":" world"}\n\ndata: [DONE]\n\n',
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text.match(/event: content_block_start/g)).toHaveLength(1);
    expect(text).toContain('"text":"hello"');
    expect(text).toContain('"text":" world"');
    expect(text).toContain('"stop_reason":"end_turn"');
    expect(text).toContain('"usage":{"output_tokens":4}');
    expect(text.trimEnd().endsWith('data: {"type":"message_stop"}')).toBe(true);
  });

  it("streams the text of a Gemma 4 reply, skipping its reasoning", async () => {
    // Frames as @cf/google/gemma-4-26b-a4b-it sent them on 2026-11-19.
    const upstream = upstreamOf([
      'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n',
      'data: {"choices":[{"index":0,"delta":{"reasoning":"   Goal"}}]}\n\n',
      'data: {"choices":[{"index":0,"delta":{"content":"Hello there"}}]}\n\n',
      "data: [DONE]\n\n",
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text.match(/event: content_block_delta/g)).toHaveLength(1);
    expect(text).toContain('"text":"Hello there"');
    expect(text).not.toContain("Goal");
  });

  it("drops the ghost thought line Gemma 4 can open a stream with", async () => {
    const upstream = upstreamOf([
      'data: {"choices":[{"index":0,"delta":{"content":"thought"}}]}\n\n',
      'data: {"choices":[{"index":0,"delta":{"content":"\\nHello"}}]}\n\n',
      'data: {"choices":[{"index":0,"delta":{"content":" there"},' +
        '"finish_reason":"stop"}]}\n\n',
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, WA_GEMMA).text();

    expect(text).toContain('"text":"Hello"');
    expect(text).toContain('"text":" there"');
    expect(text).not.toContain("thought");
  });

  it("sends a held opening once the stream ends", async () => {
    const upstream = upstreamOf([
      'data: {"choices":[{"index":0,"delta":{"content":"thou"},' +
        '"finish_reason":"stop"}]}\n\n',
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, WA_GEMMA).text();

    expect(text).toContain('"text":"thou"');
    const stopAt = text.indexOf("content_block_stop");
    expect(text.indexOf('"text":"thou"')).toBeLessThan(stopAt);
  });

  it("ends on an error when a Gemma 4 stream is cut off", async () => {
    const upstream = upstreamOf([
      'data: {"choices":[{"index":0,"delta":{"content":"Let me"}}]}\n\n',
      'data: {"choices":[{"index":0,"delta":{},' +
        '"finish_reason":"content_filter"}]}\n\n',
      "data: [DONE]\n\n",
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text).toContain("Workers AI did not finish its reply (content_filter)");
    expect(text).not.toContain("message_stop");
  });

  it("ends on an error when cut with no finish reason or [DONE]", async () => {
    const upstream = upstreamOf(['data: {"response":"Let me"}\n\n']);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text).toContain("Workers AI did not finish its reply (stream cut off)");
    expect(text).not.toContain("message_stop");
  });

  it("reports a stream that ran out of tokens as max_tokens", async () => {
    const upstream = upstreamOf([
      'data: {"choices":[{"index":0,"delta":{"content":"long"},' +
        '"finish_reason":"length"}]}\n\n',
      "data: [DONE]\n\n",
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text).toContain('"stop_reason":"max_tokens"');
  });

  it("reassembles a frame split across chunks", async () => {
    const upstream = upstreamOf(['data: {"resp', 'onse":"joined"}\n\n']);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text).toContain('"text":"joined"');
  });

  it("keeps a final frame that lacks the closing blank line", async () => {
    const upstream = upstreamOf([
      'data: {"response":"a"}\n\ndata: {"response":"tail"}',
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text).toContain('"text":"tail"');
  });

  it("skips frames with no response text", async () => {
    const upstream = upstreamOf([
      'data: {"usage":{}}\n\n',
      ": ping\n\n",
      "data: [DONE]\n\n",
    ]);

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text).not.toContain("content_block_delta");
    expect(text).toContain('"usage":{"output_tokens":0}');
  });

  it("reports an upstream read failure as a fixed-text error event", async () => {
    const upstream = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.error(new Error("socket reset"));
      },
    });

    const text = await streamWorkersAiAsAnthropic(upstream, "m").text();

    expect(text).toContain("event: error");
    expect(text).toContain('"message":"Workers AI stream failed"');
    expect(text).not.toContain("socket reset");
    expect(text).not.toContain("message_stop");
  });
});

describe("buildWorkersAiInput — message translation edge cases", () => {
  const build = (messages: AnthropicMessagesRequest["messages"]) =>
    buildWorkersAiInput({ model: "claude-sonnet-5", max_tokens: 64, messages })
      .messages;

  it("joins array tool_result text parts and drops non-text parts", () => {
    const out = build([
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "t1",
            content: [{ type: "text", text: "a" }, { type: "image" }, { text: "b" }],
          },
        ],
      },
    ]);

    expect(out).toEqual([{ role: "tool", tool_call_id: "t1", content: "a\n\nb" }]);
  });

  it("serializes object tool_result content and empties missing content", () => {
    const out = build([
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: { ok: true } },
          { type: "tool_result", tool_use_id: "t2", content: null },
        ],
      },
    ]);

    expect(out).toEqual([
      { role: "tool", tool_call_id: "t1", content: '{"ok":true}' },
      { role: "tool", tool_call_id: "t2", content: "" },
    ]);
  });

  it("puts the user text before the tool messages of the same turn", () => {
    const out = build([
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: "r" },
          { type: "text", text: "next" },
        ],
      },
    ]);

    expect(out).toEqual([
      { role: "user", content: "next" },
      { role: "tool", tool_call_id: "t1", content: "r" },
    ]);
  });

  it("keeps string tool_use input and defaults missing input to {}", () => {
    const out = build([
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "a", name: "f", input: '{"x":1}' },
          { type: "tool_use", id: "b", name: "g", input: undefined },
        ],
      },
    ]);

    expect(out).toEqual([
      {
        role: "assistant",
        content: "",
        tool_calls: [
          { id: "a", type: "function", function: { name: "f", arguments: '{"x":1}' } },
          { id: "b", type: "function", function: { name: "g", arguments: "{}" } },
        ],
      },
    ]);
  });

  it("drops an assistant turn with no text and no tool calls", () => {
    const out = build([{ role: "assistant", content: [{ type: "text", text: "" }] }]);

    expect(out).toEqual([]);
  });

  it("ignores role-mismatched blocks", () => {
    const out = build([
      {
        role: "assistant",
        content: [
          { type: "text", text: "hi" },
          { type: "image", source: { type: "base64", data: "x" } },
          { type: "tool_result", tool_use_id: "t", content: "r" },
        ],
      },
      {
        role: "user",
        content: [{ type: "tool_use", id: "a", name: "f", input: {} }],
      },
    ]);

    expect(out).toEqual([{ role: "assistant", content: "hi" }]);
  });

  it("defaults image media type to jpeg and skips unusable image sources", () => {
    const out = build([
      {
        role: "user",
        content: [
          { type: "image", source: { type: "base64", data: "x" } },
          { type: "image", source: { type: "image_url", url: "https://e.com/p" } },
          { type: "image", source: { type: "url" } },
          { type: "image", source: { type: "file", data: "y" } },
        ],
      },
    ]);

    expect(out).toEqual([
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: "data:image/jpeg;base64,x" } },
          { type: "image_url", image_url: { url: "https://e.com/p" } },
        ],
      },
    ]);
  });
});

describe("callWorkersAi thinking", () => {
  const req: AnthropicMessagesRequest = {
    model: "m",
    max_tokens: 64,
    messages: [{ role: "user", content: "hi" }],
  };
  const recordingAi = (seen: unknown[]) => ({
    run: async (_model: string, input: unknown) => {
      seen.push(input);
      return { choices: [{ message: { role: "assistant", content: "hello" }, finish_reason: "stop" }] };
    },
  });

  it("turns the model's thinking off for a plain reply", async () => {
    const seen: unknown[] = [];

    await callWorkersAi(recordingAi(seen), "m", req);

    expect(seen[0]).toMatchObject({ chat_template_kwargs: { enable_thinking: false } });
  });

  it("turns the model's thinking off when tools are declared", async () => {
    const seen: unknown[] = [];
    const withTools = { ...req, tools: [{ name: "search", input_schema: { type: "object" } }] };

    await callWorkersAi(recordingAi(seen), "m", withTools);

    expect(seen[0]).toMatchObject({ chat_template_kwargs: { enable_thinking: false } });
  });
});
