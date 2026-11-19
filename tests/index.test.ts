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

describe("fetch handler with an empty reply", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("moves past a tier that answers with no text", async () => {
    // Arrange: Groq answers 200 with blank content, Workers AI answers.
    const blank = {
      id: "chatcmpl-1",
      choices: [{ message: { role: "assistant", content: "" } }],
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json(blank)),
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
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("x-fortune-llm-prior-errors")).toMatch(
      /^groq:groq returned an empty reply/,
    );
    expect(await resp.json()).toMatchObject({
      content: [{ type: "text", text: "hello from workers ai" }],
    });
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

describe("fetch handler with malformed message entries", () => {
  const env = {
    GATEWAY_TOKEN: "gateway-token",
    CACHE_TTL_SECONDS: "0",
    RATE_LIMIT_PER_MIN: "0",
    AI: { run: async () => ({ response: "unexpected model call" }) },
  };
  const send = (body: unknown) =>
    worker.fetch(
      new Request("https://gw.test/v1/messages", {
        method: "POST",
        headers: { "x-api-key": "gateway-token" },
        body: JSON.stringify(body),
      }),
      env as never,
      ctx as never,
    );

  it("answers 400 for a null message instead of throwing", async () => {
    // Act
    const resp = await send({ model: "m", max_tokens: 8, messages: [null] });

    // Assert
    expect(resp.status).toBe(400);
  });

  it("answers 400 for a null content block", async () => {
    // Act
    const resp = await send({
      model: "m",
      max_tokens: 8,
      messages: [{ role: "user", content: [null] }],
    });

    // Assert
    expect(resp.status).toBe(400);
  });

  it("answers 400 for a message with no content", async () => {
    // Act
    const resp = await send({
      model: "m",
      max_tokens: 8,
      messages: [{ role: "user" }],
    });

    // Assert
    expect(resp.status).toBe(400);
  });

  it("answers 400 for a null system block", async () => {
    // Act
    const resp = await send({
      model: "m",
      max_tokens: 8,
      system: [null],
      messages: [{ role: "user", content: "hi" }],
    });

    // Assert
    expect(resp.status).toBe(400);
  });

  it("still serves block-array content", async () => {
    // Act
    const resp = await send({
      model: "m",
      max_tokens: 8,
      system: [{ type: "text", text: "be brief" }],
      messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
    });

    // Assert
    expect(resp.status).toBe(200);
  });
});

describe("fetch handler with the development token", () => {
  const env = {
    GATEWAY_TOKEN: "gateway-token",
    GATEWAY_DEV_TOKEN: "dev-token",
    ANTHROPIC_API_KEY: "sk-ant-test",
    CACHE_TTL_SECONDS: "0",
    RATE_LIMIT_PER_MIN: "0",
    AI: { run: async () => ({ response: "hello from workers ai" }) },
  };

  function requestWith(token: string, route?: string): Request {
    return new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": token, "content-type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
        ...(route ? { metadata: { fortune_route: route } } : {}),
      }),
    });
  }

  it("answers on the free route", async () => {
    const resp = await worker.fetch(requestWith("dev-token"), env as never, ctx as never);

    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
  });

  it("is refused for paid Anthropic", async () => {
    const resp = await worker.fetch(
      requestWith("dev-token", "anthropic"),
      env as never,
      ctx as never,
    );

    expect(resp.status).toBe(403);
    expect(await resp.text()).toContain("development token");
  });

  it("leaves an unknown token refused", async () => {
    const resp = await worker.fetch(requestWith("other"), env as never, ctx as never);

    expect(resp.status).toBe(403);
  });

  it("never falls back to paid Anthropic when free tiers fail", async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "m1", content: [{ type: "text", text: "paid" }] }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchSpy);
    try {
      const noFreeEnv = { ...env, AI: undefined };
      const resp = await worker.fetch(
        requestWith("dev-token"),
        noFreeEnv as never,
        ctx as never,
      );

      expect(resp.status).toBe(503);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
