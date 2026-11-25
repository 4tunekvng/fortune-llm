import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
vi.mock("cloudflare:workers", () => ({ DurableObject: class {} }));

import worker, { unlabelledConsumer } from "../src/index.js";

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
    expect(priorErrors).toMatch(/^gemini:[^|]+\| groq:Groq 500 \| /);
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
      /^gemini:GEMINI_API_KEY not configured \| groq:groq returned an empty reply/,
    );
    expect(await resp.json()).toMatchObject({
      content: [{ type: "text", text: "hello from workers ai" }],
    });
  });
});

describe("fetch handler with a structured request", () => {
  beforeEach(() => {
    vi.spyOn(console, "error").mockImplementation(() => {});
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  const schema = {
    type: "object",
    properties: { docType: { enum: ["eob", "denial"] } },
    required: ["docType"],
    additionalProperties: false,
  };

  function structuredRequest(): Request {
    return new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "gateway-token", "content-type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "read this letter" }],
        output_config: { format: { type: "json_schema", schema } },
      }),
    });
  }

  it("moves past a tier whose reply misses the schema", async () => {
    // Arrange: Groq answers an enum value outside the list (issue #28),
    // Workers AI answers in the schema.
    const offSchema = {
      id: "chatcmpl-1",
      choices: [
        { message: { role: "assistant", content: '{"docType":"Denial Letter"}' } },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(async () => Response.json(offSchema)));
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GROQ_API_KEY: "gsk_test_key_123",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: '{"docType":"denial"}' }) },
    };

    // Act
    const resp = await worker.fetch(structuredRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("x-fortune-llm-prior-errors")).toMatch(
      /^gemini:GEMINI_API_KEY not configured \| groq:groq reply off schema \| /,
    );
    expect(console.error).toHaveBeenCalledWith(
      "fortune-llm: tier groq failed:",
      "groq reply off schema: $.docType is not one of the allowed values " +
        "(stop_reason end_turn)",
    );
    expect(await resp.json()).toMatchObject({
      content: [{ type: "text", text: '{"docType":"denial"}' }],
    });
  });

  it("moves past a tier whose reply is cut off mid-JSON", async () => {
    // Arrange: Groq hits max_tokens inside the object.
    const cut = {
      id: "chatcmpl-2",
      choices: [
        {
          message: { role: "assistant", content: '{"docType":"den' },
          finish_reason: "length",
        },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(async () => Response.json(cut)));
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GROQ_API_KEY: "gsk_test_key_123",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: '{"docType":"eob"}' }) },
    };

    // Act
    const resp = await worker.fetch(structuredRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(console.error).toHaveBeenCalledWith(
      "fortune-llm: tier groq failed:",
      "groq reply off schema: it is not JSON (stop_reason max_tokens)",
    );
  });

  it("accepts JSON in a code fence, as lenient readers like Knox do", async () => {
    // Arrange
    const fenced = {
      id: "chatcmpl-5",
      choices: [
        {
          message: { role: "assistant", content: '```json\n{"docType":"eob"}\n```' },
        },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(async () => Response.json(fenced)));
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GROQ_API_KEY: "gsk_test_key_123",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "unused" }) },
    };

    // Act
    const resp = await worker.fetch(structuredRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("groq");
  });

  it("never shuts a tier over a quota word in a field name", async () => {
    // Arrange: the reply misses a field called quotaLeft.
    const store = new Map<string, string>();
    const kv = {
      get: async (k: string) => store.get(k) ?? null,
      put: async (k: string, v: string) => void store.set(k, v),
    };
    const quotaSchema = {
      type: "object",
      properties: { quotaLeft: { type: "integer" } },
      required: ["quotaLeft"],
    };
    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "gateway-token", "content-type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "how much is left" }],
        output_config: { format: { type: "json_schema", schema: quotaSchema } },
      }),
    });
    const missing = {
      id: "chatcmpl-6",
      choices: [{ message: { role: "assistant", content: "{}" } }],
    };
    vi.stubGlobal("fetch", vi.fn(async () => Response.json(missing)));
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GROQ_API_KEY: "gsk_test_key_123",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      CIRCUIT: kv,
      AI: { run: async () => ({ response: '{"quotaLeft":3}' }) },
    };

    // Act
    const resp = await worker.fetch(req, env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(store.has("circuit:groq")).toBe(false);
  });

  it("keeps a reply that only calls a tool", async () => {
    // Arrange: the SDK leaves parsed_output null for a tool-only reply and
    // does not throw, so the gateway must pass it on.
    const toolOnly = {
      id: "chatcmpl-4",
      choices: [
        {
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "lookup", arguments: "{}" },
              },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    };
    vi.stubGlobal("fetch", vi.fn(async () => Response.json(toolOnly)));
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GROQ_API_KEY: "gsk_test_key_123",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "unused" }) },
    };

    // Act
    const resp = await worker.fetch(structuredRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("groq");
    expect(await resp.json()).toMatchObject({
      content: [{ type: "tool_use", name: "lookup" }],
    });
  });

  it("keeps the first tier when its reply matches", async () => {
    // Arrange
    const inSchema = {
      id: "chatcmpl-3",
      choices: [{ message: { role: "assistant", content: '{"docType":"eob"}' } }],
    };
    vi.stubGlobal("fetch", vi.fn(async () => Response.json(inSchema)));
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GROQ_API_KEY: "gsk_test_key_123",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "unused" }) },
    };

    // Act
    const resp = await worker.fetch(structuredRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("groq");
    expect(resp.headers.get("x-fortune-llm-prior-errors")).toBe(
      "gemini:GEMINI_API_KEY not configured",
    );
  });
});

describe("fetch handler for a private caller", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function knoxRequest(metadata: Record<string, unknown> = {}): Request {
    return new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": "gateway-token",
        "content-type": "application/json",
        "x-fortune-consumer": "Knox",
      },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "read my letter" }],
        metadata,
      }),
    });
  }

  it("sends a listed caller's prompt to Gemini but never to the other tiers that may keep it", async () => {
    // Fortune, 2026-11-22: "data can be sent to gemini". OpenRouter, Mistral
    // and GitHub Models stay out of a private caller's chain.
    // Arrange: Gemini answers.
    const fetchMock = vi.fn(async () =>
      Response.json({ candidates: [{ content: { parts: [{ text: "gemini" }] } }] }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GEMINI_API_KEY: "gemini-key",
      PRIVATE_CONSUMERS: "knox, other",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "hello from workers ai" }) },
    };

    // Act
    const resp = await worker.fetch(knoxRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("gemini");
    expect(resp.headers.get("x-fortune-llm-chain")).toBe(
      "gemini,groq,cerebras,workers-ai",
    );
  });

  it("sends a listed caller's request pinned to Gemini to Gemini", async () => {
    // Arrange: the caller chose Gemini itself (Knox's backup reader).
    const fetchMock = vi.fn(async () =>
      Response.json({ candidates: [{ content: { parts: [{ text: "gemini" }] } }] }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      GEMINI_API_KEY: "gemini-key",
      PRIVATE_CONSUMERS: "knox",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "unused" }) },
    };

    // Act
    const resp = await worker.fetch(
      knoxRequest({ fortune_route: "gemini" }),
      env as never,
      ctx as never,
    );

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("gemini");
    expect(fetchMock).toHaveBeenCalled();
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

  it("answers 400 when tools is not an array", async () => {
    // Act
    const resp = await send({
      model: "m",
      max_tokens: 8,
      tools: { a: 1 },
      messages: [{ role: "user", content: "hi" }],
    });

    // Assert
    expect(resp.status).toBe(400);
  });

  it("answers 400 for a null tool", async () => {
    // Act
    const resp = await send({
      model: "m",
      max_tokens: 8,
      tools: [null],
      messages: [{ role: "user", content: "hi" }],
    });

    // Assert
    expect(resp.status).toBe(400);
  });

  it("answers 400 for an image block with no source", async () => {
    // Act
    const resp = await send({
      model: "m",
      max_tokens: 8,
      messages: [{ role: "user", content: [{ type: "image" }] }],
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

describe("fetch handler with a max_tokens over the gateway cap", () => {
  it("lowers max_tokens to 65536 instead of refusing the request", async () => {
    // Arrange: Claude Code asks for more output than the gateway allows.
    const run = vi.fn(async () => ({ response: "hello from workers ai" }));
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      CACHE_TTL_SECONDS: "0",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run },
    };
    const request = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "gateway-token" },
      body: JSON.stringify({
        model: "claude-opus-5-5",
        max_tokens: 128000,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    // Act
    const resp = await worker.fetch(request, env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(200);
    expect(run).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ max_tokens: 65536 }),
      expect.anything(),
    );
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

describe("fetch handler when CACHE_DO RPC fails", () => {
  it("falls back to an uncached upstream response on read/write error", async () => {
    const failingStub = {
      read: vi.fn().mockRejectedValue(new Error("DO reset")),
      write: vi.fn().mockRejectedValue(new Error("DO write failed")),
    };
    const env = {
      GATEWAY_TOKEN: "gateway-token",
      CACHE_TTL_SECONDS: "60",
      RATE_LIMIT_PER_MIN: "0",
      AI: { run: async () => ({ response: "ok despite cache error" }) },
      CACHE_DO: {
        idFromName: () => "id",
        get: () => failingStub,
      },
    };

    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "gateway-token", "content-type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
        metadata: { fortune_cache: true },
      }),
    });
    const resp = await worker.fetch(req, env as never, ctx as never);

    expect(resp.status).toBe(200);
    const body = (await resp.json()) as { content: Array<{ text: string }> };
    expect(body.content[0]?.text).toBe("ok despite cache error");
    expect(failingStub.read).toHaveBeenCalledTimes(1);
    expect(failingStub.write).toHaveBeenCalledTimes(1);
  });
});

describe("fetch handler routing before the chain", () => {
  const env = {
    GATEWAY_TOKEN: "gateway-token",
    GATEWAY_DEV_TOKEN: "dev-token",
    CACHE_TTL_SECONDS: "0",
    RATE_LIMIT_PER_MIN: "0",
    AI: { run: async () => ({ response: "hello from workers ai" }) },
  };

  function call(path: string, init: RequestInit, extra = {}) {
    const req = new Request(`https://gw.test${path}`, init);
    return worker.fetch(req, { ...env, ...extra } as never, ctx as never);
  }

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("answers a preflight with 204", async () => {
    const resp = await call("/v1/messages", { method: "OPTIONS" });

    expect(resp.status).toBe(204);
  });

  it("answers 404 outside /v1/", async () => {
    const resp = await call("/nope", { method: "GET" });

    expect(resp.status).toBe(404);
    expect(await resp.json()).toMatchObject({
      error: { type: "not_found", message: "No route for GET /nope" },
    });
  });

  it("answers 401 without a token", async () => {
    const resp = await call("/v1/messages", { method: "POST", body: "{}" });

    expect(resp.status).toBe(401);
  });

  it("answers 501 for an Anthropic-only endpoint with no paid key", async () => {
    const resp = await call("/v1/messages/count_tokens", {
      method: "POST",
      headers: { "x-api-key": "gateway-token" },
      body: "{}",
    });

    expect(resp.status).toBe(501);
    expect(await resp.json()).toMatchObject({
      error: { type: "not_implemented" },
    });
  });

  it("refuses the development token on an Anthropic-only endpoint", async () => {
    const resp = await call(
      "/v1/messages/count_tokens",
      { method: "POST", headers: { "x-api-key": "dev-token" }, body: "{}" },
      { ANTHROPIC_API_KEY: "sk-ant-test" },
    );

    expect(resp.status).toBe(403);
  });

  it("forwards an Anthropic-only endpoint when the paid key is set", async () => {
    const upstream = vi.fn(async () => new Response('{"input_tokens":3}'));
    vi.stubGlobal("fetch", upstream);

    const resp = await call(
      "/v1/messages/count_tokens",
      { method: "POST", headers: { "x-api-key": "gateway-token" }, body: "{}" },
      { ANTHROPIC_API_KEY: "sk-ant-test" },
    );

    expect(resp.status).toBe(200);
    expect(await resp.json()).toEqual({ input_tokens: 3 });
    expect(resp.headers.get("access-control-allow-origin")).toBe("*");
  });

  it("caps max_tokens in each batch item it sends to Anthropic", async () => {
    const upstream = vi.fn(async (_url: string, _init: RequestInit) =>
      Response.json({ id: "msgbatch_1" }),
    );
    vi.stubGlobal("fetch", upstream);
    const batch = {
      requests: [
        {
          custom_id: "a",
          params: { model: "claude-opus-5-5", max_tokens: 128000, messages: [] },
        },
      ],
    };

    await call(
      "/v1/messages/batches",
      {
        method: "POST",
        headers: { "x-api-key": "gateway-token" },
        body: JSON.stringify(batch),
      },
      { ANTHROPIC_API_KEY: "sk-ant-test" },
    );
    const sent = JSON.parse(String(upstream.mock.calls[0]?.[1]?.body));

    expect(upstream.mock.calls[0]?.[0]).toBe(
      "https://api.anthropic.com/v1/messages/batches",
    );
    expect(sent.requests[0].params.max_tokens).toBe(65536);
  });

  it("answers 502 when the Anthropic forward throws", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));

    const resp = await call(
      "/v1/messages/count_tokens",
      { method: "POST", headers: { "x-api-key": "gateway-token" }, body: "{}" },
      { ANTHROPIC_API_KEY: "sk-ant-test" },
    );

    expect(resp.status).toBe(502);
    expect(await resp.json()).toMatchObject({
      error: { type: "upstream_error" },
    });
  });

  it("answers 400 for a body that is not JSON", async () => {
    const resp = await call("/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "gateway-token" },
      body: "not json",
    });

    expect(resp.status).toBe(400);
    expect(await resp.json()).toMatchObject({
      error: { type: "invalid_request_error", message: "Body must be valid JSON." },
    });
  });

  it("flushes stats once with the consumer header", async () => {
    const recordEvents = vi.fn(async () => {});
    const waitUntil = vi.fn();
    const statsEnv = {
      ...env,
      STATS_DO: { idFromName: () => "id", get: () => ({ recordEvents }) },
    };
    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "gateway-token", "x-fortune-consumer": "app-a" },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    const resp = await worker.fetch(
      req,
      statsEnv as never,
      { ...ctx, waitUntil } as never,
    );

    expect(resp.status).toBe(200);
    expect(waitUntil).toHaveBeenCalledTimes(1);
    expect(recordEvents).toHaveBeenCalledWith(
      expect.arrayContaining([{ kind: "request" }]),
      "app-a",
    );
  });

  it("names a request with no app label by the site of the Worker sending it", async () => {
    // Arrange
    const recordEvents = vi.fn(async () => {});
    const statsEnv = {
      ...env,
      STATS_DO: { idFromName: () => "id", get: () => ({ recordEvents }) },
    };
    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": "gateway-token",
        "user-agent": "Anthropic/JS 1.10.0",
        "x-stainless-lang": "js",
        "cf-worker": "fortuneking.dev",
      },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    // Act
    await worker.fetch(req, statsEnv as never, ctx as never);

    // Assert
    expect(recordEvents).toHaveBeenCalledWith(
      expect.anything(),
      "unknown-worker-fortuneking-dev",
    );
  });

  it("labels development-token stats apart from the app they name", async () => {
    // Arrange
    const recordEvents = vi.fn(async () => {});
    const statsEnv = {
      ...env,
      GATEWAY_DEV_TOKEN: "dev-token",
      STATS_DO: { idFromName: () => "id", get: () => ({ recordEvents }) },
    };
    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "dev-token", "x-fortune-consumer": "knox" },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    // Act
    await worker.fetch(req, statsEnv as never, ctx as never);

    // Assert
    expect(recordEvents).toHaveBeenCalledWith(expect.anything(), "dev-knox");
  });

  it("guesses a development-token label when no app is named", async () => {
    // Arrange
    const recordEvents = vi.fn(async () => {});
    const statsEnv = {
      ...env,
      GATEWAY_DEV_TOKEN: "dev-token",
      STATS_DO: { idFromName: () => "id", get: () => ({ recordEvents }) },
    };
    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": "dev-token",
        "x-fortune-consumer": "",
        "user-agent": "curl/8.7.1",
      },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    // Act
    await worker.fetch(req, statsEnv as never, ctx as never);

    // Assert
    expect(recordEvents).toHaveBeenCalledWith(expect.anything(), "dev-unknown-curl");
  });

  it("rate-limits the development token by IP when it names an app", async () => {
    // Arrange: knox is registered with room to spare; the IP bucket is full.
    const store = new Map<string, string>([
      [`rate:5.6.7.8:${Math.floor(Date.now() / 60_000)}`, "1"],
    ]);
    const kv = {
      get: async (k: string) => store.get(k) ?? null,
      put: async (k: string, v: string) => void store.set(k, v),
    };
    const limitEnv = {
      ...env,
      GATEWAY_DEV_TOKEN: "dev-token",
      RATE_LIMIT_PER_MIN: "1",
      RATE_LIMIT_PER_MIN_KNOX: "100",
      CIRCUIT: kv,
    };
    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": "dev-token",
        "x-fortune-consumer": "knox",
        "cf-connecting-ip": "5.6.7.8",
      },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "hi" }],
      }),
    });

    // Act
    const resp = await worker.fetch(req, limitEnv as never, ctx as never);

    // Assert
    expect(resp.status).toBe(429);
    expect(resp.headers.get("x-fortune-llm-rate-scope")).toBe("ip:5.6.7.8");
  });
});

describe("unlabelledConsumer", () => {
  it("names Claude Code by its user agent, wherever it runs", () => {
    const headers = new Headers({
      "user-agent": "claude-cli/2.1.3 (external, sdk-cli)",
      "cf-worker": "fortuneking.dev",
    });

    expect(unlabelledConsumer(headers)).toBe("unknown-claude-code");
  });

  it("names a script by its SDK language when no Worker sent it", () => {
    const headers = new Headers({
      "user-agent": "Anthropic/Python 0.75.0",
      "x-stainless-lang": "python",
    });

    expect(unlabelledConsumer(headers)).toBe("unknown-python");
  });

  it("falls back to the first word of the user agent", () => {
    const headers = new Headers({ "user-agent": "curl/8.7.1" });

    expect(unlabelledConsumer(headers)).toBe("unknown-curl");
  });

  it("keeps a long site name inside the 32-character stats label", () => {
    const headers = new Headers({ "cf-worker": "a-very-long-subdomain.example.workers.dev" });

    expect(unlabelledConsumer(headers)).toMatch(/^unknown-worker-[a-z0-9_-]+$/);
    expect(unlabelledConsumer(headers).length).toBeLessThanOrEqual(32);
  });

  it("stays plain unknown when the request carries nothing to go on", () => {
    expect(unlabelledConsumer(new Headers())).toBe("unknown");
  });
});
