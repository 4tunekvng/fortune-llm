import { describe, it, expect } from "vitest";
import { decideRoute, estimateInputTokens } from "../src/route.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

const baseReq = (overrides: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest => ({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "hi" }],
  ...overrides,
});

describe("decideRoute", () => {
  it("defaults to the free chain [workers-ai, gemini] for plain text chat", () => {
    const d = decideRoute(baseReq());
    expect(d.tiers).toEqual(["workers-ai", "gemini"]);
    expect(d.tiers).not.toContain("anthropic");
  });

  it("routes tools[]-bearing requests gemini-first (Llama 4 Scout bounces Claude-Code-style agent prompts)", () => {
    const d = decideRoute(baseReq({ tools: [{ name: "search", input_schema: { type: "object" } }] }));
    expect(d.tiers).toEqual(["gemini", "workers-ai"]);
    expect(d.tiers).not.toContain("anthropic");
    expect(d.reason).toMatch(/tools=1/);
  });

  it("routes vision-bearing requests to gemini only (workers-ai has no vision)", () => {
    const d = decideRoute(
      baseReq({
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "what's in this?" },
              { type: "image", source: { type: "base64", media_type: "image/png", data: "..." } },
            ],
          },
        ],
      }),
    );
    expect(d.tiers).toEqual(["gemini"]);
    expect(d.tiers).not.toContain("anthropic");
  });

  it("prefers gemini for very long context but keeps workers-ai as fallback", () => {
    const huge = "x".repeat(420_000); // ~105k tokens > 100k threshold
    const d = decideRoute(baseReq({ messages: [{ role: "user", content: huge }] }));
    expect(d.tiers[0]).toBe("gemini");
    expect(d.tiers).toContain("workers-ai");
    expect(d.tiers).not.toContain("anthropic");
  });

  it("honors metadata.fortune_route=anthropic as the explicit escape valve", () => {
    const d = decideRoute(baseReq({ metadata: { fortune_route: "anthropic" } }));
    expect(d.tiers).toEqual(["anthropic"]);
    expect(d.reason).toContain("explicit");
  });

  it("honors metadata.fortune_route=workers-ai (even with tools)", () => {
    const d = decideRoute(
      baseReq({
        tools: [{ name: "x", input_schema: { type: "object" } }],
        metadata: { fortune_route: "workers-ai" },
      }),
    );
    expect(d.tiers).toEqual(["workers-ai"]);
  });

  it("honors metadata.fortune_route=gemini", () => {
    const d = decideRoute(baseReq({ metadata: { fortune_route: "gemini" } }));
    expect(d.tiers).toEqual(["gemini"]);
  });

  it("empty tools array does not change routing", () => {
    const d = decideRoute(baseReq({ tools: [] }));
    expect(d.tiers).toEqual(["workers-ai", "gemini"]);
  });
});

describe("estimateInputTokens", () => {
  it("counts string content", () => {
    const n = estimateInputTokens(baseReq({ messages: [{ role: "user", content: "abcd".repeat(40) }] }));
    expect(n).toBe(40); // 160 chars / 4
  });

  it("counts text blocks", () => {
    const n = estimateInputTokens(
      baseReq({
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: "abcd".repeat(20) }],
          },
        ],
      }),
    );
    expect(n).toBe(20);
  });

  it("counts the system prompt", () => {
    const n = estimateInputTokens(
      baseReq({ system: "x".repeat(40), messages: [{ role: "user", content: "y" }] }),
    );
    expect(n).toBe(11); // ceil((40 + 1) / 4)
  });
});
