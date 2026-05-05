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
  it("defaults to workers-ai for plain text chat", () => {
    const d = decideRoute(baseReq());
    expect(d.kind).toBe("workers-ai");
  });

  it("routes to anthropic when tools is non-empty", () => {
    const d = decideRoute(baseReq({ tools: [{ name: "search" }] }));
    expect(d.kind).toBe("anthropic");
    expect(d.reason).toContain("tools=1");
  });

  it("does NOT route to anthropic when tools is an empty array", () => {
    const d = decideRoute(baseReq({ tools: [] }));
    expect(d.kind).toBe("workers-ai");
  });

  it("routes to anthropic on any image content block", () => {
    const d = decideRoute(
      baseReq({
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: "what's in this?" },
              { type: "image", source: { type: "base64", data: "..." } },
            ],
          },
        ],
      }),
    );
    expect(d.kind).toBe("anthropic");
    expect(d.reason).toContain("image");
  });

  it("routes to anthropic when input is very long", () => {
    const huge = "x".repeat(80_000); // ~20k tokens
    const d = decideRoute(baseReq({ messages: [{ role: "user", content: huge }] }));
    expect(d.kind).toBe("anthropic");
    expect(d.reason).toMatch(/input tokens/);
  });

  it("honors metadata.fortune_route override forcing anthropic", () => {
    const d = decideRoute(baseReq({ metadata: { fortune_route: "anthropic" } }));
    expect(d.kind).toBe("anthropic");
    expect(d.reason).toContain("explicit");
  });

  it("honors metadata.fortune_route override forcing workers-ai (even with tools)", () => {
    const d = decideRoute(
      baseReq({
        tools: [{ name: "x" }],
        metadata: { fortune_route: "workers-ai" },
      }),
    );
    expect(d.kind).toBe("workers-ai");
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
