import { describe, it, expect } from "vitest";
import {
  buildGeminiInput,
  translateToolsToGemini,
  sanitizeJsonSchemaForGemini,
  geminiToAnthropicMessage,
  requestHasImage,
  streamGeminiAsAnthropic,
} from "../src/gemini.js";
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

  it("provides an empty content block when Gemini returns no parts", () => {
    const msg = geminiToAnthropicMessage({ candidates: [{ content: { parts: [] } }] }, "x");
    expect(msg.content).toEqual([{ type: "text", text: "" }]);
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
});
