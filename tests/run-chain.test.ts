import { afterEach, describe, expect, it, vi } from "vitest";
vi.mock("cloudflare:workers", () => ({ DurableObject: class {} }));

import worker from "../src/index.js";
import { getCircuitState, tripCircuit } from "../src/circuit-breaker.js";

const ctx = { waitUntil() {}, passThroughOnException() {} };

// crypto.subtle.timingSafeEqual is a Workers runtime extension that Node lacks.
const subtle = crypto.subtle as unknown as Record<string, unknown>;
subtle.timingSafeEqual ??= (a: Uint8Array, b: Uint8Array) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

class MapKv {
  readonly store = new Map<string, string>();
  async get(key: string): Promise<string | null> {
    return this.store.get(key) ?? null;
  }
  async put(key: string, value: string): Promise<void> {
    this.store.set(key, value);
  }
}

function messagesRequest(
  extra: Record<string, unknown> = {},
  headers: Record<string, string> = {},
): Request {
  return new Request("https://gw.test/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": "gateway-token",
      "content-type": "application/json",
      ...headers,
    },
    body: JSON.stringify({
      model: "claude-sonnet-5-5",
      max_tokens: 64,
      messages: [{ role: "user", content: "hi" }],
      ...extra,
    }),
  });
}

function makeEnv(kv: MapKv, overrides: Record<string, unknown> = {}) {
  return {
    GATEWAY_TOKEN: "gateway-token",
    GROQ_API_KEY: "gsk_test_key_123",
    CACHE_TTL_SECONDS: "3600",
    RATE_LIMIT_PER_MIN: "0",
    CIRCUIT: kv,
    AI: { run: async () => ({ response: "hello from workers ai" }) },
    ...overrides,
  };
}

function groqFails(status: number, message: string): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => Response.json({ error: { message } }, { status })),
  );
}

function cachedKeys(kv: MapKv): string[] {
  return [...kv.store.keys()].filter((k) => k.startsWith("cache:"));
}

describe("runChain through the fetch handler", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stores a cacheable JSON reply and marks it miss-stored", async () => {
    // Arrange
    groqFails(500, "boom");
    const kv = new MapKv();

    // Act
    const resp = await worker.fetch(
      messagesRequest({ temperature: 0 }),
      makeEnv(kv) as never,
      ctx as never,
    );

    // Assert
    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("x-fortune-llm-cache")).toBe("miss-stored");
    expect(resp.headers.get("access-control-allow-origin")).toBe("*");
    expect(resp.headers.get("x-fortune-llm-chain")).toContain("workers-ai");
    expect(resp.headers.get("x-fortune-llm-reason")).toBeTruthy();
    expect(await resp.json()).toMatchObject({
      content: [{ type: "text", text: "hello from workers ai" }],
    });
    expect(cachedKeys(kv)).toHaveLength(1);
  });

  it("stores a cacheable stream request as JSON and replies with SSE", async () => {
    // Arrange
    groqFails(500, "boom");
    const kv = new MapKv();

    // Act
    const resp = await worker.fetch(
      messagesRequest({ temperature: 0, stream: true }),
      makeEnv(kv) as never,
      ctx as never,
    );

    // Assert
    expect(resp.headers.get("x-fortune-llm-cache")).toBe("miss-stored-stream");
    expect(resp.headers.get("content-type")).toBe("text/event-stream; charset=utf-8");
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    const text = await resp.text();
    expect(text).toContain("event: message_start");
    expect(text).toContain("hello from workers ai");
    expect(cachedKeys(kv)).toHaveLength(1);
  });

  it("does not cache or label a request that is not cacheable", async () => {
    // Arrange
    groqFails(500, "boom");
    const kv = new MapKv();

    // Act
    const resp = await worker.fetch(messagesRequest(), makeEnv(kv) as never, ctx as never);

    // Assert
    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-cache")).toBeNull();
    expect(cachedKeys(kv)).toEqual([]);
  });

  it("skips a tier whose circuit is open and names it", async () => {
    // Arrange
    const fetchSpy = vi.fn(async () => new Response("unexpected", { status: 500 }));
    vi.stubGlobal("fetch", fetchSpy);
    const kv = new MapKv();
    await tripCircuit("groq", kv as never, 60_000, "quota");

    // Act
    const resp = await worker.fetch(messagesRequest(), makeEnv(kv) as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("x-fortune-llm-skipped")).toMatch(/^groq:/);
    expect(resp.headers.get("x-fortune-llm-prior-errors")).not.toContain("groq");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("trips the circuit when a shared key hits its quota", async () => {
    // Arrange
    groqFails(429, "rate limit reached");
    const kv = new MapKv();

    // Act
    await worker.fetch(messagesRequest(), makeEnv(kv) as never, ctx as never);

    // Assert
    expect((await getCircuitState("groq", kv as never)).open).toBe(true);
  });

  it("leaves the circuit closed when one request is too big for a tier (413)", async () => {
    // Arrange: Groq's real per-minute "request too large" reply, in its real shape.
    const message =
      "Request too large for model `llama-3.3-70b-versatile` on tokens per minute " +
      "(TPM): Limit 8000, Requested 9000, please reduce your message size and try " +
      "again. Upgrade at https://console.groq.com/settings/billing";
    const error = { message, type: "tokens", code: "rate_limit_exceeded" };
    vi.stubGlobal("fetch", vi.fn(async () => Response.json({ error }, { status: 413 })));
    const kv = new MapKv();

    // Act
    await worker.fetch(messagesRequest(), makeEnv(kv) as never, ctx as never);

    // Assert
    expect((await getCircuitState("groq", kv as never)).open).toBe(false);
  });

  it("trips for the full window when a tier's account is out of credit (402)", async () => {
    // Arrange
    groqFails(402, "Payment required");
    const kv = new MapKv();

    // Act
    await worker.fetch(messagesRequest(), makeEnv(kv) as never, ctx as never);

    // Assert
    const state = await getCircuitState("groq", kv as never);
    expect((state.until ?? 0) - Date.now()).toBeGreaterThan(50 * 60 * 1000);
  });

  it("trips a 429 for as long as its Retry-After header says", async () => {
    // Arrange
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json(
          { error: { message: "rate limit reached" } },
          { status: 429, headers: { "retry-after": "90" } },
        ),
      ),
    );
    const kv = new MapKv();

    // Act
    await worker.fetch(messagesRequest(), makeEnv(kv) as never, ctx as never);

    // Assert
    const state = await getCircuitState("groq", kv as never);
    expect((state.until ?? 0) - Date.now()).toBeGreaterThan(85_000);
    expect((state.until ?? 0) - Date.now()).toBeLessThan(95_000);
  });

  it("leaves the circuit closed when a caller's own key hits its quota", async () => {
    // Arrange
    groqFails(429, "rate limit reached");
    const kv = new MapKv();

    // Act
    await worker.fetch(
      messagesRequest({}, { "x-fortune-byok-groq": "gsk_caller_key_123" }),
      makeEnv(kv) as never,
      ctx as never,
    );

    // Assert
    expect((await getCircuitState("groq", kv as never)).open).toBe(false);
  });

  it("answers 503 naming every tier once the whole chain fails", async () => {
    // Arrange
    groqFails(500, "boom");
    const kv = new MapKv();
    const env = makeEnv(kv, {
      AI: {
        run: async () => {
          throw new Error("workers ai down");
        },
      },
    });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(503);
    const body = (await resp.json()) as {
      error: { type: string; message: string; detail: string };
    };
    expect(body.error.type).toBe("all_backends_failed");
    expect(body.error.detail).toContain("groq (Groq 500)");
    expect(body.error.detail).toContain("workers-ai (workers ai down)");
  });

  it("says plainly that the models are busy, and not to retry at once", async () => {
    // Arrange
    groqFails(500, "boom");
    const env = makeEnv(new MapKv(), {
      AI: {
        run: async () => {
          throw new Error("workers ai down");
        },
      },
    });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(503);
    expect(resp.headers.get("x-should-retry")).toBe("false");
    expect(resp.headers.get("retry-after")).toBe("60");
    expect(await resp.json()).toMatchObject({
      error: {
        message:
          "Every free model is busy or out right now. Try again in a few minutes.",
        retry_after_s: 60,
      },
    });
  });

  it("keeps each open tier's reopen time and waits for the soonest", async () => {
    // Arrange: every free tier's circuit is open; the soonest in 10 minutes.
    const kv = new MapKv();
    await tripCircuit("groq", kv as never, 10 * 60_000, "Groq 429: per day");
    await tripCircuit("workers-ai", kv as never, 30 * 60_000, "neurons exhausted");
    const env = makeEnv(kv, { GROQ_API_KEY: undefined });
    const req = messagesRequest({ metadata: { fortune_route: "groq" } });

    // Act
    const resp = await worker.fetch(req, env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(503);
    const body = (await resp.json()) as {
      error: { type: string; detail: string; retry_after_s: number };
    };
    expect(body.error.type).toBe("quota_exhausted");
    expect(body.error.detail).toMatch(/groq \(open until \d{4}-\d{2}-\d{2}T/);
    expect(body.error.retry_after_s).toBeGreaterThan(590);
    expect(body.error.retry_after_s).toBeLessThanOrEqual(600);
  });

  it("names the paid cap's end instead of passing its 400 through", async () => {
    // Arrange: every free tier fails, then paid answers with its cap 400.
    const capped = {
      type: "error",
      error: {
        type: "invalid_request_error",
        message:
          "You have reached your specified API usage limits. You will regain " +
          "access on 2026-12-01 at 00:00 UTC.",
      },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.startsWith("https://api.anthropic.com")
          ? Response.json(capped, { status: 400 })
          : Response.json({ error: { message: "boom" } }, { status: 500 }),
      ),
    );
    const env = makeEnv(new MapKv(), {
      ANTHROPIC_API_KEY: "sk-ant-test",
      AI: {
        run: async () => {
          throw new Error("workers ai down");
        },
      },
    });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(503);
    expect(await resp.json()).toMatchObject({
      error: {
        type: "all_backends_failed",
        paid_capped_until: "2026-12-01T00:00:00Z",
      },
    });
  });

  it("still passes on a paid 400 that is the caller's own mistake", async () => {
    // Arrange
    const bad = {
      type: "error",
      error: { type: "invalid_request_error", message: "max_tokens: too large" },
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => Response.json(bad, { status: 400 })),
    );
    const env = makeEnv(new MapKv(), { ANTHROPIC_API_KEY: "sk-ant-test" });
    const req = messagesRequest({ metadata: { fortune_route: "anthropic" } });

    // Act
    const resp = await worker.fetch(req, env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(400);
  });

  it("answers from a later tier after Gemini refuses a non-ASCII field", async () => {
    // Arrange
    groqFails(500, "boom");
    const tool = {
      name: "t",
      description: "pick a size",
      input_schema: {
        type: "object",
        properties: { "\u540d": { type: "number", enum: [1, 2] } },
      },
    };
    const env = makeEnv(new MapKv(), { GEMINI_API_KEY: "gemini-test-key" });

    // Act
    const resp = await worker.fetch(
      messagesRequest({ tools: [tool] }),
      env as never,
      ctx as never,
    );

    // Assert
    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
  });

  it("answers from the next tier when groq cuts its reply off", async () => {
    // Arrange
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Let me" },
              finish_reason: "content_filter",
            },
          ],
        }),
      ),
    );
    const env = makeEnv(new MapKv(), { CACHE_TTL_SECONDS: "0" });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(await resp.text()).toContain("hello from workers ai");
  });

  it("checks a structured stream against its schema, then streams it", async () => {
    // Arrange
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        Response.json({
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "Sure, here you go" },
              finish_reason: "stop",
            },
          ],
        }),
      ),
    );
    const env = makeEnv(new MapKv(), {
      CACHE_TTL_SECONDS: "0",
      AI: { run: async () => ({ response: '{"a":"ok"}' }) },
    });
    const schema = {
      type: "object",
      properties: { a: { type: "string" } },
      required: ["a"],
    };

    // Act
    const resp = await worker.fetch(
      messagesRequest({
        stream: true,
        output_config: { format: { type: "json_schema", schema } },
      }),
      env as never,
      ctx as never,
    );

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("content-type")).toMatch(/event-stream/);
    expect(await resp.text()).toContain('{\\"a\\":\\"ok\\"}');
  });
});
