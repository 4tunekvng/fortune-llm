import { afterEach, describe, expect, it, vi } from "vitest";
vi.mock("cloudflare:workers", () => ({ DurableObject: class {} }));

import worker from "../src/index.js";
import { getCircuitState } from "../src/circuit-breaker.js";
import { chainBudgetMs, tierLimitMs } from "../src/time-limit.js";

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
  metadata: Record<string, unknown> = {},
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
      metadata,
    }),
  });
}

/** A provider that never answers: its fetch settles only when aborted. */
function hangingFetch() {
  return vi.fn(
    (_url: string, init?: RequestInit) =>
      new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
      }),
  );
}

/** Workers AI that never answers, so no free tier can win the race. */
function hangingAi(
  _model: string,
  _input: unknown,
  options?: { signal?: AbortSignal },
) {
  return new Promise((_, reject) => {
    options?.signal?.addEventListener("abort", () => reject(options.signal?.reason));
  });
}

function makeEnv(kv: MapKv, overrides: Record<string, unknown> = {}) {
  return {
    GATEWAY_TOKEN: "gateway-token",
    GROQ_API_KEY: "gsk_test_key_123",
    CACHE_TTL_SECONDS: "0",
    RATE_LIMIT_PER_MIN: "0",
    CIRCUIT: kv,
    AI: { run: async () => ({ response: "hello from workers ai" }) },
    ...overrides,
  };
}

describe("chainBudgetMs", () => {
  it("reads the caller's timeout from the SDK headers", () => {
    // Act
    const budgets = [
      chainBudgetMs(undefined, "20"), // TypeScript SDK, whole seconds
      chainBudgetMs(undefined, "600.0"), // Python SDK default
      chainBudgetMs(undefined, "NOT_GIVEN"), // Python SDK, no timeout set
      chainBudgetMs(undefined, null), // curl or fetch
      chainBudgetMs(undefined, "1"), // nearly out of time
      chainBudgetMs("30000", null),
      chainBudgetMs("soon", "abc"),
    ];

    // Assert
    expect(budgets).toEqual([18_000, 80_000, 80_000, 80_000, 5_000, 30_000, 80_000]);
  });
});

describe("tierLimitMs", () => {
  it("bounds the fast tiers and leaves the slow ones to the budget", () => {
    // Act
    const limits = [
      tierLimitMs("gemini", {}),
      tierLimitMs("groq", {}),
      tierLimitMs("openrouter", {}),
      tierLimitMs("workers-ai", {}),
      tierLimitMs("github-models", { TIER_TIME_LIMIT_MS_GITHUB_MODELS: "7000" }),
    ];

    // Assert
    expect(limits).toEqual([20_000, 20_000, undefined, undefined, 7_000]);
  });
});

describe("fetch handler with a slow tier", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("gives up on a hanging tier, cancels it and moves on", async () => {
    // Arrange: Groq hangs; its own limit is shortened for the test.
    const kv = new MapKv();
    const fetchMock = hangingFetch();
    vi.stubGlobal("fetch", fetchMock);
    const env = makeEnv(kv, { TIER_TIME_LIMIT_MS_GROQ: "50" });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("x-fortune-llm-prior-errors")).toMatch(
      /^groq:groq timed out after 50 ms \| /,
    );
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(init.signal?.aborted).toBe(true);
    expect((await getCircuitState("groq", kv as never)).open).toBe(true);
  });

  it("does not shut a tier the chain budget cut short", async () => {
    // Arrange: the whole budget is shorter than Groq's own limit.
    const kv = new MapKv();
    vi.stubGlobal("fetch", hangingFetch());
    const env = makeEnv(kv, { CHAIN_TIME_LIMIT_MS: "50" });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(503);
    expect((await getCircuitState("groq", kv as never)).open).toBe(false);
  });

  it("skips the free tiers left once the budget is spent", async () => {
    // Arrange
    const kv = new MapKv();
    const run = vi.fn(async () => ({ response: "unused" }));
    vi.stubGlobal("fetch", hangingFetch());
    const env = makeEnv(kv, { CHAIN_TIME_LIMIT_MS: "50", AI: { run } });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(503);
    expect(run).not.toHaveBeenCalled();
  });

  it("still runs paid Anthropic after the free budget is spent", async () => {
    // Arrange: Groq hangs past the budget; Anthropic answers.
    const kv = new MapKv();
    const fetchMock = vi.fn((url: string, init?: RequestInit) =>
      url.startsWith("https://api.anthropic.com")
        ? Promise.resolve(
            Response.json({
              id: "msg_1",
              type: "message",
              role: "assistant",
              content: [{ type: "text", text: "paid answer" }],
              stop_reason: "end_turn",
            }),
          )
        : hangingFetch()(url, init),
    );
    vi.stubGlobal("fetch", fetchMock);
    const env = makeEnv(kv, {
      CHAIN_TIME_LIMIT_MS: "50",
      ANTHROPIC_API_KEY: "sk-ant-test",
      AI: { run: hangingAi },
    });

    // Act
    const resp = await worker.fetch(messagesRequest(), env as never, ctx as never);

    // Assert
    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("anthropic");
  });

  it("hands Workers AI the abort signal", async () => {
    // Arrange
    const kv = new MapKv();
    const run = vi.fn(async (..._args: unknown[]) => ({ response: "hello" }));
    const env = makeEnv(kv, { AI: { run } });

    // Act
    await worker.fetch(
      messagesRequest({ fortune_route: "workers-ai" }),
      env as never,
      ctx as never,
    );

    // Assert
    const options = run.mock.calls[0]?.[2] as { signal?: AbortSignal } | undefined;
    expect(options?.signal).toBeInstanceOf(AbortSignal);
  });

  it("cuts the budget to the caller's own timeout", async () => {
    // Arrange: the SDK says the caller waits 7 s, so the budget is 5 s.
    vi.useFakeTimers();
    const kv = new MapKv();
    vi.stubGlobal("fetch", hangingFetch());
    const env = makeEnv(kv);

    // Act
    const pending = worker.fetch(
      messagesRequest({}, { "x-stainless-timeout": "7" }),
      env as never,
      ctx as never,
    );
    await vi.advanceTimersByTimeAsync(5_001);
    const resp = await pending;
    vi.useRealTimers();

    // Assert
    expect(resp.status).toBe(503);
    expect((await getCircuitState("groq", kv as never)).open).toBe(false);
  });

  it("does not start paid Anthropic when the caller is about to leave", async () => {
    // Arrange: the caller waits 20 s; every free tier hangs.
    vi.useFakeTimers();
    const kv = new MapKv();
    const fetchMock = hangingFetch();
    vi.stubGlobal("fetch", fetchMock);
    const env = makeEnv(kv, {
      ANTHROPIC_API_KEY: "sk-ant-test",
      AI: { run: hangingAi },
    });

    // Act
    const pending = worker.fetch(
      messagesRequest({}, { "x-stainless-timeout": "20" }),
      env as never,
      ctx as never,
    );
    await vi.advanceTimersByTimeAsync(18_001);
    const resp = await pending;
    vi.useRealTimers();

    // Assert
    expect(resp.status).toBe(503);
    const paidCalls = fetchMock.mock.calls.filter(([url]) =>
      String(url).startsWith("https://api.anthropic.com"),
    );
    expect(paidCalls).toHaveLength(0);
  });

  it("gives a long-context request no fast-tier limit", async () => {
    // Arrange: a 110K-token prompt; Gemini hangs; its own limit is tiny.
    const kv = new MapKv();
    vi.stubGlobal("fetch", hangingFetch());
    const env = makeEnv(kv, {
      GEMINI_API_KEY: "gemini-key",
      TIER_TIME_LIMIT_MS_GEMINI: "20",
      CHAIN_TIME_LIMIT_MS: "100",
      AI: { run: hangingAi },
    });
    const req = new Request("https://gw.test/v1/messages", {
      method: "POST",
      headers: { "x-api-key": "gateway-token", "content-type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-5-5",
        max_tokens: 64,
        messages: [{ role: "user", content: "x".repeat(440_000) }],
      }),
    });

    // Act
    await worker.fetch(req, env as never, ctx as never);

    // Assert: the budget cut it, not its own limit, so it is not shut.
    expect((await getCircuitState("gemini", kv as never)).open).toBe(false);
  });
});
