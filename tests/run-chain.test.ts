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
    const body = (await resp.json()) as { error: { type: string; message: string } };
    expect(body.error.type).toBe("all_backends_failed");
    expect(body.error.message).toContain("groq");
    expect(body.error.message).toContain("workers-ai");
  });
});
