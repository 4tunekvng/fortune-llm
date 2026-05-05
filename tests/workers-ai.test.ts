import { describe, it, expect } from "vitest";
import {
  buildWorkersAiInput,
  flattenContent,
  workersAiToAnthropicMessage,
  extractWorkersAiDelta,
} from "../src/workers-ai.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

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
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      system: "you are helpful",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(input.messages[0]).toEqual({ role: "system", content: "you are helpful" });
    expect(input.messages[1]).toEqual({ role: "user", content: "hi" });
  });

  it("flattens system content blocks", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-4-6",
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
      model: "claude-sonnet-4-6",
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

  it("clamps non-finite max_tokens to a sane default", () => {
    const input = buildWorkersAiInput({
      model: "claude-sonnet-4-6",
      max_tokens: Number.NaN,
      messages: [{ role: "user", content: "hi" }],
    });
    expect(input.max_tokens).toBe(1024);
  });
});

describe("workersAiToAnthropicMessage", () => {
  it("wraps text response in an Anthropic Message shape", () => {
    const msg = workersAiToAnthropicMessage(
      { response: "hello world", usage: { prompt_tokens: 5, completion_tokens: 2 } },
      "claude-sonnet-4-6",
      { messages: [{ role: "user", content: "hi" }], max_tokens: 100 },
    );
    expect(msg.type).toBe("message");
    expect(msg.role).toBe("assistant");
    expect(msg.model).toBe("claude-sonnet-4-6");
    expect(msg.content).toEqual([{ type: "text", text: "hello world" }]);
    expect(msg.stop_reason).toBe("end_turn");
    expect(msg.usage).toEqual({ input_tokens: 5, output_tokens: 2 });
  });

  it("estimates tokens when Workers AI omits usage", () => {
    const msg = workersAiToAnthropicMessage(
      { response: "abcd".repeat(10) },
      "claude-sonnet-4-6",
      { messages: [{ role: "user", content: "x".repeat(20) }], max_tokens: 100 },
    );
    expect(msg.usage.input_tokens).toBeGreaterThan(0);
    expect(msg.usage.output_tokens).toBe(10);
  });
});

describe("extractWorkersAiDelta", () => {
  it("returns the response field for normal frames", () => {
    expect(extractWorkersAiDelta('{"response":"abc"}')).toBe("abc");
  });

  it("returns null for malformed JSON", () => {
    expect(extractWorkersAiDelta("not json")).toBe(null);
  });

  it("returns null when response is not a string", () => {
    expect(extractWorkersAiDelta('{"response": 42}')).toBe(null);
  });
});
