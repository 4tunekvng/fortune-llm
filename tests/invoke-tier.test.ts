import { describe, it, expect, afterEach, vi } from "vitest";
import worker from "../src/index.js";

// The Durable Object base class only exists inside the Workers runtime.
vi.mock("cloudflare:workers", () => ({ DurableObject: class {} }));
// Gateway auth calls crypto.subtle.timingSafeEqual, which only Workers has.
vi.mock("../src/auth.js", () => ({ authenticate: () => ({ ok: true }) }));

// Pins the per-tier response headers the dispatcher adds (model name,
// key counts, BYOK flag). Drives the public Worker fetch with one tier
// forced through metadata.fortune_route and the upstream fetch stubbed.

type Env = Parameters<typeof worker.fetch>[1];

const ctx = {
  waitUntil: () => {},
  passThroughOnException: () => {},
} as unknown as ExecutionContext;

const BYOK_KEY = "byok-key-123456";

const OPENAI_REPLY = {
  id: "chatcmpl-1",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "hello" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 1, completion_tokens: 1 },
};

const GEMINI_REPLY = {
  candidates: [
    { content: { parts: [{ text: "hello" }] }, finishReason: "STOP" },
  ],
  usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1 },
};

const ANTHROPIC_REPLY = {
  id: "msg_1",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: [{ type: "text", text: "hello" }],
  stop_reason: "end_turn",
  stop_sequence: null,
  usage: { input_tokens: 1, output_tokens: 1 },
};

function stubUpstream(body: object): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    ),
  );
}

function baseEnv(overrides: Partial<Env> = {}): Env {
  return {
    AI: {
      run: async () => ({ response: "hello" }),
    },
    CACHE_TTL_SECONDS: "0",
    RATE_LIMIT_PER_MIN: "0",
    ...overrides,
  } as Env;
}

function messagesRequest(
  route: string,
  headers: Record<string, string> = {},
): Request {
  return new Request("https://gw.example/v1/messages", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 16,
      messages: [{ role: "user", content: "hi" }],
      metadata: { fortune_route: route },
    }),
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("tier dispatch headers", () => {
  it("workers-ai reports the default model", async () => {
    const resp = await worker.fetch(
      messagesRequest("workers-ai"),
      baseEnv(),
      ctx,
    );

    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("workers-ai");
    expect(resp.headers.get("x-fortune-llm-model")).toBe(
      "@cf/google/gemma-4-26b-a4b-it",
    );
    expect(resp.headers.get("x-fortune-llm-byok")).toBeNull();
  });

  it("workers-ai reports an overridden model", async () => {
    const resp = await worker.fetch(
      messagesRequest("workers-ai"),
      baseEnv({ DEFAULT_OSS_MODEL: "@cf/custom/model" }),
      ctx,
    );

    expect(resp.headers.get("x-fortune-llm-model")).toBe("@cf/custom/model");
  });

  it("gemini with one shared key omits the key-count header", async () => {
    stubUpstream(GEMINI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("gemini"),
      baseEnv({ GEMINI_API_KEY: "shared-gemini" }),
      ctx,
    );

    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-model")).toBe("gemini-2.5-flash");
    expect(resp.headers.get("x-fortune-llm-gemini-keys")).toBeNull();
    expect(resp.headers.get("x-fortune-llm-byok")).toBeNull();
  });

  it("gemini with several shared keys reports the key count", async () => {
    stubUpstream(GEMINI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("gemini"),
      baseEnv({ GEMINI_API_KEYS: "k1,k2,k3" }),
      ctx,
    );

    expect(resp.headers.get("x-fortune-llm-gemini-keys")).toBe("3");
  });

  it("gemini with a caller key uses only that key and flags BYOK", async () => {
    stubUpstream(GEMINI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("gemini", { "x-fortune-byok-gemini": BYOK_KEY }),
      baseEnv({ GEMINI_API_KEYS: "k1,k2,k3" }),
      ctx,
    );

    expect(resp.headers.get("x-fortune-llm-gemini-keys")).toBeNull();
    expect(resp.headers.get("x-fortune-llm-byok")).toBe("true");
  });

  it("groq reports its default model with a shared key", async () => {
    stubUpstream(OPENAI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("groq"),
      baseEnv({ GROQ_API_KEY: "shared-groq" }),
      ctx,
    );

    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-model")).toBe(
      "openai/gpt-oss-120b",
    );
    expect(resp.headers.get("x-fortune-llm-byok")).toBeNull();
  });

  it("groq flags BYOK when the caller sends a key", async () => {
    stubUpstream(OPENAI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("groq", { "x-fortune-byok-groq": BYOK_KEY }),
      baseEnv(),
      ctx,
    );

    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-byok")).toBe("true");
  });

  it("cerebras reports an overridden model", async () => {
    stubUpstream(OPENAI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("cerebras"),
      baseEnv({
        CEREBRAS_API_KEY: "shared-cerebras",
        DEFAULT_CEREBRAS_MODEL: "cerebras-custom",
      }),
      ctx,
    );

    expect(resp.headers.get("x-fortune-llm-model")).toBe("cerebras-custom");
  });

  it("github-models reports an overridden model", async () => {
    stubUpstream(OPENAI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("github-models"),
      baseEnv({
        GITHUB_MODELS_API_KEY: "shared-github",
        DEFAULT_GITHUB_MODELS_MODEL: "github-custom",
      }),
      ctx,
    );

    expect(resp.headers.get("x-fortune-llm-model")).toBe("github-custom");
  });

  it("mistral reports an overridden model", async () => {
    stubUpstream(OPENAI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("mistral"),
      baseEnv({
        MISTRAL_API_KEY: "shared-mistral",
        DEFAULT_MISTRAL_MODEL: "mistral-custom",
      }),
      ctx,
    );

    expect(resp.headers.get("x-fortune-llm-model")).toBe("mistral-custom");
  });

  it("openrouter reports the first model and the model count", async () => {
    stubUpstream(OPENAI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("openrouter"),
      baseEnv({
        OPENROUTER_API_KEY: "shared-openrouter",
        DEFAULT_OPENROUTER_MODELS: "a/one:free,b/two:free",
      }),
      ctx,
    );

    expect(resp.headers.get("x-fortune-llm-model")).toBe("a/one:free");
    expect(resp.headers.get("x-fortune-llm-openrouter-models")).toBe("2");
  });

  it("anthropic with the shared key adds no BYOK flag", async () => {
    stubUpstream(ANTHROPIC_REPLY);

    const resp = await worker.fetch(
      messagesRequest("anthropic"),
      baseEnv({ ANTHROPIC_API_KEY: "shared-anthropic" }),
      ctx,
    );

    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-route")).toBe("anthropic");
    expect(resp.headers.get("x-fortune-llm-byok")).toBeNull();
  });

  it("anthropic with a caller key flags BYOK", async () => {
    stubUpstream(ANTHROPIC_REPLY);

    const resp = await worker.fetch(
      messagesRequest("anthropic", {
        "x-fortune-byok-anthropic": BYOK_KEY,
      }),
      baseEnv(),
      ctx,
    );

    expect(resp.status).toBe(200);
    expect(resp.headers.get("x-fortune-llm-byok")).toBe("true");
  });

  it("a tier with no key fails the chain", async () => {
    stubUpstream(OPENAI_REPLY);

    const resp = await worker.fetch(
      messagesRequest("mistral"),
      baseEnv(),
      ctx,
    );

    expect(resp.status).toBe(503);
    expect(fetch).not.toHaveBeenCalled();
  });
});
