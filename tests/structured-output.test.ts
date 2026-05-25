import { describe, it, expect } from "vitest";
import {
  extractOutputSchema,
  getOutputSchemaObject,
  outputSchemaToOpenAIResponseFormat,
  outputSchemaToGeminiGenerationConfig,
} from "../src/structured-output.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

const baseReq = (extra: Partial<AnthropicMessagesRequest> & { output_config?: unknown } = {}): AnthropicMessagesRequest => ({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "hi" }],
  ...(extra as Partial<AnthropicMessagesRequest>),
});

const objectSchema = {
  type: "object",
  properties: { title: { type: "string" }, count: { type: "number" } },
  required: ["title"],
  additionalProperties: false,
};

describe("extractOutputSchema", () => {
  it("returns null when output_config is absent", () => {
    expect(extractOutputSchema(baseReq())).toBeNull();
  });

  it("returns null when format type is not json_schema", () => {
    const r = baseReq({ output_config: { format: { type: "other", schema: objectSchema } } } as never);
    expect(extractOutputSchema(r)).toBeNull();
  });

  it("returns null when schema is missing", () => {
    const r = baseReq({ output_config: { format: { type: "json_schema" } } } as never);
    expect(extractOutputSchema(r)).toBeNull();
  });

  it("returns the format when json_schema + schema are present", () => {
    const r = baseReq({ output_config: { format: { type: "json_schema", schema: objectSchema } } } as never);
    const out = extractOutputSchema(r);
    expect(out?.type).toBe("json_schema");
    expect(out?.schema).toEqual(objectSchema);
  });
});

describe("getOutputSchemaObject", () => {
  it("returns just the schema object", () => {
    const r = baseReq({ output_config: { format: { type: "json_schema", schema: objectSchema } } } as never);
    expect(getOutputSchemaObject(r)).toEqual(objectSchema);
  });

  it("returns null when no output_config", () => {
    expect(getOutputSchemaObject(baseReq())).toBeNull();
  });
});

describe("outputSchemaToOpenAIResponseFormat", () => {
  it("builds the canonical OpenAI strict response_format", () => {
    const out = outputSchemaToOpenAIResponseFormat({ type: "json_schema", schema: objectSchema });
    expect(out).toEqual({
      response_format: {
        type: "json_schema",
        json_schema: {
          name: "response",
          schema: objectSchema,
          strict: true,
        },
      },
    });
  });

  it("uses the format name when provided and safe", () => {
    const out = outputSchemaToOpenAIResponseFormat({
      type: "json_schema",
      schema: objectSchema,
      name: "MyOutput_1",
    });
    expect(out.response_format.json_schema.name).toBe("MyOutput_1");
  });

  it("falls back to 'response' when name has unsafe characters", () => {
    const out = outputSchemaToOpenAIResponseFormat({
      type: "json_schema",
      schema: objectSchema,
      name: "My Name With Spaces!",
    });
    expect(out.response_format.json_schema.name).toBe("response");
  });

  it("caps name length to 64 chars", () => {
    const longName = "a".repeat(200);
    const out = outputSchemaToOpenAIResponseFormat({
      type: "json_schema",
      schema: objectSchema,
      name: longName,
    });
    expect(out.response_format.json_schema.name.length).toBeLessThanOrEqual(64);
  });
});

describe("outputSchemaToGeminiGenerationConfig", () => {
  it("returns responseMimeType + responseSchema", () => {
    const out = outputSchemaToGeminiGenerationConfig(objectSchema);
    expect(out).toEqual({
      responseMimeType: "application/json",
      responseSchema: objectSchema,
    });
  });

  it("passes the schema through unchanged (the Gemini-side sanitizer in gemini.ts handles cleanup)", () => {
    const richSchema = {
      type: "object",
      $schema: "https://json-schema.org/draft/2020-12/schema",
      additionalProperties: false,
      properties: { x: { type: "string" } },
    };
    const out = outputSchemaToGeminiGenerationConfig(richSchema);
    expect(out.responseSchema).toBe(richSchema);
  });
});
