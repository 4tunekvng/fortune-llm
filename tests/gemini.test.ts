import { describe, it, expect } from "vitest";
import {
  buildGeminiInput,
  translateToolsToGemini,
  sanitizeJsonSchemaForGemini,
  geminiToAnthropicMessage,
  requestHasImage,
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
  it("strips $schema, additionalProperties, $defs etc.", () => {
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
