import { afterEach, describe, expect, it, vi } from "vitest";
vi.mock("cloudflare:workers", () => ({ DurableObject: class {} }));

import worker from "../src/index.js";

const ctx = { waitUntil() {}, passThroughOnException() {} };

// crypto.subtle.timingSafeEqual is a Workers runtime extension that Node lacks.
const subtle = crypto.subtle as unknown as Record<string, unknown>;
subtle.timingSafeEqual ??= (a: Uint8Array, b: Uint8Array) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

function messagesRequest(): Request {
  return new Request("https://gw.test/v1/messages", {
    method: "POST",
    headers: { "x-api-key": "gateway-token", "content-type": "application/json" },
    body: JSON.stringify({
      model: "claude-sonnet-5-5",
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
    }),
  });
}

describe("fetch handler fallback chain", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps a later tier's success after a multi-line error", async () => {
    // Arrange: Groq fails with pretty-printed JSON (newlines), Workers AI answers.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        const body =
          '{\n  "error": {\n    "message": "org_secret123 over limit"\n  }\n}';
        return new Response(body, { status: 500 });
      }),
    );
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GROQ_API_KEY: "gsk_test_key_123",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "hello from workers ai" }) },
    };

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    const priorErrors = resp.headers.get("x-fortune-llm-prior-errors");
    expect(priorErrors).toMatch(/^groq:Groq 500 \| /);
    expect(priorErrors).not.toContain("org_secret123");
  });
});

describe("fetch handler with the paid fallback configured", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("keeps the first free tier's success", async () => {
    // Arrange: an Anthropic key appends the paid tier to the chain, and the
    // chain's reason (sent as a header) must stay a valid header value.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("unexpected upstream call", { status: 500 })),
    );
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      ANTHROPIC_API_KEY: "sk-ant-test",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "hello from workers ai" }) },
    };

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("x-fortune-llm-reason")).toMatch(/last-resort/);
  });
});
