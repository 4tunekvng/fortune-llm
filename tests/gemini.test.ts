import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  buildGeminiInput,
  translateToolsToGemini,
  translateToolChoiceToGemini,
  sanitizeJsonSchemaForGemini,
  geminiToAnthropicMessage,
  requestHasImage,
  streamGeminiAsAnthropic,
  resolveGeminiKeys,
  callGeminiWithRotation,
} from "../src/gemini.js";
import { isQuotaError } from "../src/circuit-breaker.js";
import type { AnthropicMessagesRequest, GeminiResponse } from "../src/types.js";

const baseReq = (overrides: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest => ({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "hi" }],
  ...overrides,
});

describe("buildGeminiInput", () => {
  it("translates a plain user turn", () => {
    const out = buildGeminiInput(baseReq());
    expect(out.contents).toEqual([{ role: "user", parts: [{ text: "hi" }] }]);
    expect(out.generationConfig?.maxOutputTokens).toBe(256);
  });

  it("translates an assistant turn to role=model", () => {
    const out = buildGeminiInput(
      baseReq({
        messages: [
          { role: "user", content: "ping" },
          { role: "assistant", content: "pong" },
        ],
      }),
    );
    expect(out.contents).toHaveLength(2);
    expect(out.contents[1]?.role).toBe("model");
  });

  it("lifts the system prompt into systemInstruction", () => {
    const out = buildGeminiInput(
      baseReq({ system: "you are helpful", messages: [{ role: "user", content: "hi" }] }),
    );
    expect(out.systemInstruction).toEqual({ parts: [{ text: "you are helpful" }] });
  });

  it("translates an Anthropic image block to inlineData", () => {
    const out = buildGeminiInput(
      baseReq({
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "describe this" },
              {
                type: "image",
                source: { type: "base64", media_type: "image/png", data: "AAAA" },
              },
            ],
          },
        ],
      }),
    );
    const userContent = out.contents[0];
    expect(userContent?.parts).toEqual([
      { text: "describe this" },
      { inlineData: { mimeType: "image/png", data: "AAAA" } },
    ]);
  });

  it("translates tools[] to functionDeclarations", () => {
    const out = buildGeminiInput(
      baseReq({
        tools: [
          {
            name: "get_weather",
            description: "Get the weather",
            input_schema: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
      }),
    );
    expect(out.tools).toEqual([
      {
        functionDeclarations: [
          {
            name: "get_weather",
            description: "Get the weather",
            parameters: {
              type: "object",
              properties: { city: { type: "string" } },
              required: ["city"],
            },
          },
        ],
      },
    ]);
  });

  it("translates an assistant tool_use block to a functionCall part", () => {
    const out = buildGeminiInput(
      baseReq({
        messages: [
          { role: "user", content: "weather?" },
          {
            role: "assistant",
            content: [
              { type: "text", text: "calling" },
              { type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "Paris" } },
            ],
          },
          {
            role: "user",
            content: [
              { type: "tool_result", tool_use_id: "toolu_1", content: "Sunny, 22C" },
            ],
          },
        ],
      }),
    );
    const assistantMsg = out.contents[1];
    expect(assistantMsg?.role).toBe("model");
    expect(assistantMsg?.parts).toContainEqual({
      functionCall: { name: "get_weather", args: { city: "Paris" } },
    });
    const toolResultMsg = out.contents[2];
    // functionResponse.name must be the tool name (looked up from the
    // prior assistant tool_use), not the Anthropic tool_use_id.
    expect(toolResultMsg?.parts[0]).toMatchObject({
      functionResponse: { name: "get_weather", response: { result: "Sunny, 22C" } },
    });
  });
});

describe("sanitizeJsonSchemaForGemini", () => {
  it("strips $schema, additionalProperties, $defs etc. (legacy blocklist cases)", () => {
    const out = sanitizeJsonSchemaForGemini({
      $schema: "http://json-schema.org/draft-07/schema#",
      type: "object",
      additionalProperties: false,
      $defs: { foo: { type: "string" } },
      properties: {
        city: { type: "string", $ref: "#/$defs/foo" },
      },
    });
    expect(out).not.toHaveProperty("$schema");
    expect(out).not.toHaveProperty("additionalProperties");
    expect(out).not.toHaveProperty("$defs");
    expect((out.properties as Record<string, Record<string, unknown>>).city).not.toHaveProperty("$ref");
  });

  it("strips exclusiveMinimum / exclusiveMaximum / propertyNames / patternProperties (Claude-Code tool schemas)", () => {
    const out = sanitizeJsonSchemaForGemini({
      type: "object",
      properties: {
        line: {
          type: "integer",
          exclusiveMinimum: 0,
          exclusiveMaximum: 10000,
          description: "Line number, 1-indexed.",
        },
        glob: {
          type: "object",
          propertyNames: { pattern: "^[A-Z_]+$" },
          patternProperties: { "^.*$": { type: "string" } },
        },
      },
      required: ["line"],
    });
    const lineSchema = (out.properties as Record<string, Record<string, unknown>>).line!;
    expect(lineSchema).not.toHaveProperty("exclusiveMinimum");
    expect(lineSchema).not.toHaveProperty("exclusiveMaximum");
    // Allowed fields preserved.
    expect(lineSchema.type).toBe("integer");
    expect(lineSchema.description).toBe("Line number, 1-indexed.");
    const globSchema = (out.properties as Record<string, Record<string, unknown>>).glob!;
    expect(globSchema).not.toHaveProperty("propertyNames");
    expect(globSchema).not.toHaveProperty("patternProperties");
    // required preserved at the top level
    expect(out.required).toEqual(["line"]);
  });

  it("preserves the Gemini-allowed schema fields (type, items, enum, etc.)", () => {
    const out = sanitizeJsonSchemaForGemini({
      type: "object",
      properties: {
        mode: { type: "string", enum: ["read", "write"] },
        path: { type: "string", pattern: "^/.*", minLength: 1 },
        bytes: { type: "integer", minimum: 0, maximum: 1_000_000 },
        files: { type: "array", items: { type: "string" }, minItems: 1, maxItems: 50 },
      },
      required: ["mode", "path"],
    });
    expect(out.type).toBe("object");
    expect(out.required).toEqual(["mode", "path"]);
    const props = out.properties as Record<string, Record<string, unknown>>;
    expect(props.mode).toEqual({ type: "string", enum: ["read", "write"] });
    expect(props.path).toEqual({ type: "string", pattern: "^/.*", minLength: 1 });
    expect(props.bytes).toEqual({ type: "integer", minimum: 0, maximum: 1_000_000 });
    expect(props.files).toEqual({ type: "array", items: { type: "string" }, minItems: 1, maxItems: 50 });
  });
});

describe("translateToolsToGemini", () => {
  it("returns the function declarations with sanitized parameters", () => {
    const out = translateToolsToGemini([
      {
        name: "search",
        description: "do search",
        input_schema: { $schema: "x", type: "object", properties: { q: { type: "string" } } },
      },
    ]);
    expect(out[0]?.name).toBe("search");
    expect(out[0]?.description).toBe("do search");
    expect(out[0]?.parameters).not.toHaveProperty("$schema");
    expect(out[0]?.parameters).toHaveProperty("properties");
  });
});

describe("translateToolChoiceToGemini", () => {
  it("maps auto → AUTO", () => {
    expect(translateToolChoiceToGemini({ type: "auto" })).toEqual({
      functionCallingConfig: { mode: "AUTO" },
    });
  });

  it("maps any → ANY", () => {
    expect(translateToolChoiceToGemini({ type: "any" })).toEqual({
      functionCallingConfig: { mode: "ANY" },
    });
  });

  it("maps none → NONE", () => {
    expect(translateToolChoiceToGemini({ type: "none" })).toEqual({
      functionCallingConfig: { mode: "NONE" },
    });
  });

  it("maps tool:{name} → ANY with allowedFunctionNames", () => {
    expect(translateToolChoiceToGemini({ type: "tool", name: "search" })).toEqual({
      functionCallingConfig: { mode: "ANY", allowedFunctionNames: ["search"] },
    });
  });

  it("returns null when choice is undefined", () => {
    expect(translateToolChoiceToGemini(undefined)).toBeNull();
  });

  it("buildGeminiInput sets toolConfig when tools + tool_choice are present", () => {
    const out = buildGeminiInput(
      baseReq({
        tools: [{ name: "search", input_schema: { type: "object" } }],
        tool_choice: { type: "any" },
      }),
    );
    expect(out.toolConfig).toEqual({ functionCallingConfig: { mode: "ANY" } });
  });

  it("buildGeminiInput omits toolConfig when tools are present but tool_choice is not set", () => {
    const out = buildGeminiInput(
      baseReq({
        tools: [{ name: "search", input_schema: { type: "object" } }],
      }),
    );
    expect(out.toolConfig).toBeUndefined();
  });
});

describe("geminiToAnthropicMessage", () => {
  it("converts a text-only Gemini candidate to an Anthropic Message", () => {
    const gem: GeminiResponse = {
      candidates: [
        {
          content: { parts: [{ text: "hello" }, { text: " world" }] },
          finishReason: "STOP",
        },
      ],
      usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 2 },
    };
    const msg = geminiToAnthropicMessage(gem, "claude-sonnet-4-6");
    expect(msg.content).toEqual([{ type: "text", text: "hello world" }]);
    expect(msg.stop_reason).toBe("end_turn");
    expect(msg.usage).toEqual({ input_tokens: 5, output_tokens: 2 });
  });

  it("converts a functionCall part to a tool_use content block with stop_reason=tool_use", () => {
    const gem: GeminiResponse = {
      candidates: [
        {
          content: {
            parts: [
              { text: "Looking up." },
              { functionCall: { name: "get_weather", args: { city: "Tokyo" } } },
            ],
          },
          finishReason: "STOP",
        },
      ],
    };
    const msg = geminiToAnthropicMessage(gem, "claude-sonnet-4-6");
    expect(msg.stop_reason).toBe("tool_use");
    expect(msg.content).toHaveLength(2);
    expect(msg.content[0]).toEqual({ type: "text", text: "Looking up." });
    expect(msg.content[1]).toMatchObject({
      type: "tool_use",
      name: "get_weather",
      input: { city: "Tokyo" },
    });
  });

  it("maps Gemini MAX_TOKENS to Anthropic max_tokens stop_reason", () => {
    const gem: GeminiResponse = {
      candidates: [
        {
          content: { parts: [{ text: "truncated..." }] },
          finishReason: "MAX_TOKENS",
        },
      ],
    };
    const msg = geminiToAnthropicMessage(gem, "claude-sonnet-4-6");
    expect(msg.stop_reason).toBe("max_tokens");
  });

  it("maps Gemini STOP_SEQUENCE to Anthropic stop_sequence stop_reason", () => {
    const gem: GeminiResponse = {
      candidates: [
        {
          content: { parts: [{ text: "partial answer" }] },
          finishReason: "STOP_SEQUENCE",
        },
      ],
    };
    const msg = geminiToAnthropicMessage(gem, "claude-sonnet-4-6");
    expect(msg.stop_reason).toBe("stop_sequence");
  });

  it("provides an empty content block when Gemini returns no parts", () => {
    const msg = geminiToAnthropicMessage({ candidates: [{ content: { parts: [] } }] }, "x");
    expect(msg.content).toEqual([{ type: "text", text: "" }]);
  });
});

describe("callGemini tool-silence detection", () => {
  // Mock fetch so we can drive the response shape without hitting Google.
  const realFetch = globalThis.fetch;
  afterEachInstall();

  function afterEachInstall() {
    // vitest's afterEach via dynamic import to avoid changing the existing imports.
  }

  function withFetchOnce(body: object) {
    (globalThis as { fetch: typeof fetch }).fetch = (async () =>
      new Response(JSON.stringify(body), { status: 200 })) as typeof fetch;
  }

  // restore fetch after each test
  it("throws when fortune_require_tools=true and Gemini returned no functionCall", async () => {
    const { callGemini } = await import("../src/gemini.js");
    withFetchOnce({
      candidates: [
        {
          content: { parts: [{ text: "Sure, I can help with that." }] },
          finishReason: "STOP",
        },
      ],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 12 },
    });

    const req: AnthropicMessagesRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      messages: [{ role: "user", content: "do the thing" }],
      tools: [
        {
          name: "do_thing",
          description: "do the thing",
          input_schema: { type: "object", properties: {} },
        },
      ],
      metadata: { fortune_require_tools: true } as { fortune_require_tools: boolean },
    };

    await expect(callGemini({ apiKey: "test", model: "gemini-2.5-flash" }, req)).rejects.toThrow(
      /no tool calls/i,
    );

    globalThis.fetch = realFetch;
  });

  it("does NOT throw when fortune_require_tools is missing (default behaviour)", async () => {
    const { callGemini } = await import("../src/gemini.js");
    withFetchOnce({
      candidates: [
        { content: { parts: [{ text: "Sure thing." }] }, finishReason: "STOP" },
      ],
      usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 4 },
    });

    const req: AnthropicMessagesRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      messages: [{ role: "user", content: "hi" }],
      tools: [
        {
          name: "do_thing",
          description: "do the thing",
          input_schema: { type: "object", properties: {} },
        },
      ],
    };

    const resp = await callGemini({ apiKey: "test", model: "gemini-2.5-flash" }, req);
    expect(resp.status).toBe(200);

    globalThis.fetch = realFetch;
  });

  it("does NOT throw when fortune_require_tools=true and Gemini DID call a tool", async () => {
    const { callGemini } = await import("../src/gemini.js");
    withFetchOnce({
      candidates: [
        {
          content: {
            parts: [{ functionCall: { name: "do_thing", args: {} } }],
          },
          finishReason: "STOP",
        },
      ],
      usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 4 },
    });

    const req: AnthropicMessagesRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      messages: [{ role: "user", content: "do it" }],
      tools: [
        {
          name: "do_thing",
          description: "do the thing",
          input_schema: { type: "object", properties: {} },
        },
      ],
      metadata: { fortune_require_tools: true } as { fortune_require_tools: boolean },
    };

    const resp = await callGemini({ apiKey: "test", model: "gemini-2.5-flash" }, req);
    expect(resp.status).toBe(200);

    globalThis.fetch = realFetch;
  });
});

describe("requestHasImage", () => {
  it("returns true when any user message contains an image block", () => {
    expect(
      requestHasImage(
        baseReq({
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "look" },
                { type: "image", source: { type: "base64", data: "AAAA" } },
              ],
            },
          ],
        }),
      ),
    ).toBe(true);
  });

  it("returns false for text-only requests", () => {
    expect(requestHasImage(baseReq())).toBe(false);
  });
});

describe("streamGeminiAsAnthropic — stop_reason in the final message_delta", () => {
  /**
   * Build a Gemini SSE upstream that emits a single chunk shaped like
   * what generativelanguage.googleapis.com sends in streamGenerateContent.
   * `parts` controls whether the candidate carries a text part, a
   * functionCall part, or both.
   */
  function makeUpstream(parts: unknown[], finishReason = "STOP"): ReadableStream<Uint8Array> {
    const payload = {
      candidates: [{ content: { parts }, finishReason }],
      usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 },
    };
    const sse = `data: ${JSON.stringify(payload)}\n\n`;
    return new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(sse));
        controller.close();
      },
    });
  }

  async function consumeDownstream(resp: Response): Promise<Record<string, unknown>[]> {
    const reader = resp.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const events: Record<string, unknown>[] = [];
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 2);
        let data = "";
        for (const line of frame.split("\n")) {
          if (line.startsWith("data:")) {
            data = line.slice(5).trim();
            break;
          }
        }
        if (data) {
          try {
            events.push(JSON.parse(data));
          } catch {
            /* ignore malformed */
          }
        }
      }
    }
    return events;
  }

  it("emits stop_reason=tool_use when the stream contained a functionCall part", async () => {
    const upstream = makeUpstream([
      { functionCall: { name: "who_am_i", args: {} } },
    ]);
    const resp = streamGeminiAsAnthropic(upstream, "gemini-2.5-flash");
    const events = await consumeDownstream(resp);
    const delta = events.find((e) => e.type === "message_delta") as
      | { delta?: { stop_reason?: string } }
      | undefined;
    expect(delta?.delta?.stop_reason).toBe("tool_use");
  });

  it("emits stop_reason=end_turn for a pure-text response", async () => {
    const upstream = makeUpstream([{ text: "hi there" }]);
    const resp = streamGeminiAsAnthropic(upstream, "gemini-2.5-flash");
    const events = await consumeDownstream(resp);
    const delta = events.find((e) => e.type === "message_delta") as
      | { delta?: { stop_reason?: string } }
      | undefined;
    expect(delta?.delta?.stop_reason).toBe("end_turn");
  });

  it("emits stop_reason=tool_use when text+functionCall coexist in one candidate", async () => {
    const upstream = makeUpstream([
      { text: "let me check" },
      { functionCall: { name: "who_am_i", args: {} } },
    ]);
    const resp = streamGeminiAsAnthropic(upstream, "gemini-2.5-flash");
    const events = await consumeDownstream(resp);
    const delta = events.find((e) => e.type === "message_delta") as
      | { delta?: { stop_reason?: string } }
      | undefined;
    expect(delta?.delta?.stop_reason).toBe("tool_use");
  });

  it("emits stop_reason=max_tokens when finishReason is MAX_TOKENS and no tool was called", async () => {
    const upstream = makeUpstream([{ text: "lorem ipsum" }], "MAX_TOKENS");
    const resp = streamGeminiAsAnthropic(upstream, "gemini-2.5-flash");
    const events = await consumeDownstream(resp);
    const delta = events.find((e) => e.type === "message_delta") as
      | { delta?: { stop_reason?: string } }
      | undefined;
    expect(delta?.delta?.stop_reason).toBe("max_tokens");
  });

  it("tool_use takes precedence over MAX_TOKENS when both signals are present", async () => {
    const upstream = makeUpstream(
      [{ functionCall: { name: "who_am_i", args: {} } }],
      "MAX_TOKENS",
    );
    const resp = streamGeminiAsAnthropic(upstream, "gemini-2.5-flash");
    const events = await consumeDownstream(resp);
    const delta = events.find((e) => e.type === "message_delta") as
      | { delta?: { stop_reason?: string } }
      | undefined;
    expect(delta?.delta?.stop_reason).toBe("tool_use");
  });

  it("emits stop_reason=stop_sequence when finishReason is STOP_SEQUENCE", async () => {
    const upstream = makeUpstream([{ text: "partial response" }], "STOP_SEQUENCE");
    const resp = streamGeminiAsAnthropic(upstream, "gemini-2.5-flash");
    const events = await consumeDownstream(resp);
    const delta = events.find((e) => e.type === "message_delta") as
      | { delta?: { stop_reason?: string } }
      | undefined;
    expect(delta?.delta?.stop_reason).toBe("stop_sequence");
  });
});

describe("resolveGeminiKeys", () => {
  it("returns the singular key when only it is set", () => {
    expect(resolveGeminiKeys(undefined, "single-key")).toEqual(["single-key"]);
  });

  it("splits comma-separated plural", () => {
    expect(resolveGeminiKeys("a,b,c", undefined)).toEqual(["a", "b", "c"]);
  });

  it("trims whitespace and skips empties", () => {
    expect(resolveGeminiKeys(" a , , b ,c", undefined)).toEqual(["a", "b", "c"]);
  });

  it("merges singular + plural without duplicating shared keys", () => {
    expect(resolveGeminiKeys("a,b", "b")).toEqual(["a", "b"]);
  });

  it("returns an empty array when neither is set", () => {
    expect(resolveGeminiKeys(undefined, undefined)).toEqual([]);
  });

  it("ignores empty strings", () => {
    expect(resolveGeminiKeys("", "")).toEqual([]);
  });
});

describe("callGeminiWithRotation", () => {
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

  const okResponse = () =>
    new Response(
      JSON.stringify({
        candidates: [
          { content: { role: "model", parts: [{ text: "ok" }] }, finishReason: "STOP" },
        ],
        usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1 },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );

  it("throws synchronously when given zero keys", async () => {
    await expect(
      callGeminiWithRotation([], "gemini-2.5-flash", baseReq(), isQuotaError),
    ).rejects.toThrow(/not configured/);
  });

  it("returns on the first key that succeeds", async () => {
    fetchSpy.mockResolvedValueOnce(okResponse());
    const resp = await callGeminiWithRotation(["k1"], "gemini-2.5-flash", baseReq(), isQuotaError);
    expect(resp.status).toBe(200);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("falls through quota errors to the next key", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("RESOURCE_EXHAUSTED: quota exceeded for free tier", { status: 429 }),
    );
    fetchSpy.mockResolvedValueOnce(okResponse());
    const resp = await callGeminiWithRotation(
      ["k1", "k2"],
      "gemini-2.5-flash",
      baseReq(),
      isQuotaError,
    );
    expect(resp.status).toBe(200);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });

  it("throws on the first non-quota error without trying remaining keys", async () => {
    // 400 with no quota signal — likely a request-shape problem that
    // every key would hit equally. No point burning the rest.
    fetchSpy.mockResolvedValueOnce(new Response("bad request", { status: 400 }));
    await expect(
      callGeminiWithRotation(["k1", "k2", "k3"], "gemini-2.5-flash", baseReq(), isQuotaError),
    ).rejects.toThrow(/Gemini 400/);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("rethrows the last quota error when every key is exhausted", async () => {
    fetchSpy.mockResolvedValue(
      new Response("RESOURCE_EXHAUSTED: rate limit", { status: 429 }),
    );
    await expect(
      callGeminiWithRotation(["k1", "k2"], "gemini-2.5-flash", baseReq(), isQuotaError),
    ).rejects.toThrow(/RESOURCE_EXHAUSTED|429/);
    expect(fetchSpy).toHaveBeenCalledTimes(2);
  });
});
