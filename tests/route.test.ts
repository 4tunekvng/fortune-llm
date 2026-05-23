import { describe, it, expect } from "vitest";
import { decideRoute, estimateInputTokens } from "../src/route.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

const baseReq = (overrides: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest => ({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "hi" }],
  ...overrides,
});

// The default free chain stacks every independent free-quota pool. Centralized
// here so future additions (Cerebras, GitHub Models, …) update one place.
const DEFAULT_FREE_CHAIN = ["groq", "workers-ai", "gemini", "openrouter"] as const;
const LONG_CONTEXT_FREE_CHAIN = ["gemini", "openrouter", "workers-ai"] as const;
const VISION_FREE_CHAIN = ["gemini", "openrouter"] as const;

describe("decideRoute", () => {
  it("defaults to the multi-provider free chain for plain text chat", () => {
    const d = decideRoute(baseReq());
    expect(d.tiers).toEqual([...DEFAULT_FREE_CHAIN]);
    expect(d.tiers).not.toContain("anthropic");
  });

  it("routes tools[]-bearing requests through the same multi-provider chain (groq first)", () => {
    const d = decideRoute(baseReq({ tools: [{ name: "search", input_schema: { type: "object" } }] }));
    expect(d.tiers).toEqual([...DEFAULT_FREE_CHAIN]);
    expect(d.tiers[0]).toBe("groq");
    expect(d.tiers).not.toContain("anthropic");
    expect(d.reason).toMatch(/tools=1/);
  });

  it("routes vision-bearing requests to gemini→openrouter (workers-ai has no vision adapter)", () => {
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
    expect(d.tiers).toEqual([...VISION_FREE_CHAIN]);
    expect(d.tiers).not.toContain("anthropic");
  });

  it("prefers gemini for very long context, openrouter Llama 4 next, workers-ai last", () => {
    const huge = "x".repeat(420_000); // ~105k tokens > 100k threshold
    const d = decideRoute(baseReq({ messages: [{ role: "user", content: huge }] }));
    expect(d.tiers).toEqual([...LONG_CONTEXT_FREE_CHAIN]);
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

  it("honors metadata.fortune_route=groq", () => {
    const d = decideRoute(baseReq({ metadata: { fortune_route: "groq" } }));
    expect(d.tiers).toEqual(["groq"]);
  });

  it("honors metadata.fortune_route=openrouter", () => {
    const d = decideRoute(baseReq({ metadata: { fortune_route: "openrouter" } }));
    expect(d.tiers).toEqual(["openrouter"]);
  });

  it("empty tools array does not change routing", () => {
    const d = decideRoute(baseReq({ tools: [] }));
    expect(d.tiers).toEqual([...DEFAULT_FREE_CHAIN]);
  });
});

describe("decideRoute — anthropic auto-fallback", () => {
  it("appends anthropic to a plain-text default chain when configured", () => {
    const d = decideRoute(baseReq(), { anthropicFallback: true });
    expect(d.tiers).toEqual([...DEFAULT_FREE_CHAIN, "anthropic"]);
    expect(d.reason).toMatch(/anthropic appended as last-resort/);
  });

  it("appends anthropic to a tools-bearing default chain", () => {
    const d = decideRoute(
      baseReq({ tools: [{ name: "search", input_schema: { type: "object" } }] }),
      { anthropicFallback: true },
    );
    expect(d.tiers).toEqual([...DEFAULT_FREE_CHAIN, "anthropic"]);
  });

  it("appends anthropic to a vision-only default chain", () => {
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
      { anthropicFallback: true },
    );
    expect(d.tiers).toEqual([...VISION_FREE_CHAIN, "anthropic"]);
  });

  it("does NOT append anthropic when an explicit override pins to a free tier", () => {
    const d = decideRoute(
      baseReq({ metadata: { fortune_route: "workers-ai" } }),
      { anthropicFallback: true },
    );
    expect(d.tiers).toEqual(["workers-ai"]);
  });

  it("respects metadata.fortune_route=free as an explicit opt-out of paid fallback", () => {
    const d = decideRoute(
      baseReq({ metadata: { fortune_route: "free" } }),
      { anthropicFallback: true },
    );
    expect(d.tiers).toEqual([...DEFAULT_FREE_CHAIN]);
    expect(d.tiers).not.toContain("anthropic");
    expect(d.reason).toContain("no paid fallback");
  });

  it("metadata.fortune_route=anthropic still works when anthropic also happens to be configured", () => {
    const d = decideRoute(
      baseReq({ metadata: { fortune_route: "anthropic" } }),
      { anthropicFallback: true },
    );
    expect(d.tiers).toEqual(["anthropic"]);
  });

  it("does not append anthropic twice if it somehow appeared in the free chain", () => {
    // Defensive: the free chain builder never returns anthropic, but if
    // future code did, we shouldn't duplicate it.
    const d = decideRoute(baseReq(), { anthropicFallback: false });
    expect(d.tiers.filter((t) => t === "anthropic")).toHaveLength(0);
  });

  it("routes requests with output_config (json_schema) directly to anthropic when configured", () => {
    // messages.parse() helper from the SDK sets output_config.format.type=json_schema.
    // Free backends don't understand this field and return code-fenced JSON which
    // the SDK's text-mode parser then chokes on. Skip the free tiers entirely.
    const req = {
      ...baseReq(),
      output_config: {
        format: { type: "json_schema", schema: { type: "object", properties: { title: { type: "string" } } } },
      },
    } as unknown as AnthropicMessagesRequest;
    const d = decideRoute(req, { anthropicFallback: true });
    expect(d.tiers).toEqual(["anthropic"]);
    expect(d.reason).toMatch(/output_config/);
  });

  it("falls through to free chain when output_config is present but anthropic not configured (fails loud)", () => {
    const req = {
      ...baseReq(),
      output_config: {
        format: { type: "json_schema", schema: {} },
      },
    } as unknown as AnthropicMessagesRequest;
    const d = decideRoute(req, { anthropicFallback: false });
    expect(d.tiers).toEqual([...DEFAULT_FREE_CHAIN]);
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
