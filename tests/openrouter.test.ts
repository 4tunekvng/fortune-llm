import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  resolveOpenRouterModels,
  callOpenRouter,
  DEFAULT_OPENROUTER_MODELS,
} from "../src/openrouter.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

const baseReq = (overrides: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest => ({
  model: "claude-sonnet-4-6",
  max_tokens: 64,
  messages: [{ role: "user", content: "hi" }],
  ...overrides,
});

describe("resolveOpenRouterModels", () => {
  it("returns the default list when env value is undefined", () => {
    expect(resolveOpenRouterModels(undefined)).toEqual([...DEFAULT_OPENROUTER_MODELS]);
  });

  it("returns the default list when env value is empty", () => {
    expect(resolveOpenRouterModels("")).toEqual([...DEFAULT_OPENROUTER_MODELS]);
  });

  it("splits comma-separated env value, trimming whitespace", () => {
    expect(resolveOpenRouterModels(" a:free , b:free ,c:free")).toEqual([
      "a:free",
      "b:free",
      "c:free",
    ]);
  });

  it("returns the default list when env value parses to empty (e.g. only commas/whitespace)", () => {
    expect(resolveOpenRouterModels(" , , ")).toEqual([...DEFAULT_OPENROUTER_MODELS]);
  });

  it("caps at OpenRouter's max of 3 models (API rejects longer arrays)", () => {
    const out = resolveOpenRouterModels("a:free,b:free,c:free,d:free,e:free");
    expect(out).toHaveLength(3);
    expect(out).toEqual(["a:free", "b:free", "c:free"]);
  });
});

describe("callOpenRouter", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let fetchSpy: any;
  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });
  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("posts model + models[] + provider.sort to the OpenRouter endpoint", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: "or-1",
          choices: [
            { index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" },
          ],
        }),
        { status: 200 },
      ),
    );
    const resp = await callOpenRouter(
      { apiKey: "or-key", models: ["a:free", "b:free", "c:free"] },
      baseReq(),
    );
    expect(resp.status).toBe(200);

    const [reqUrl, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
    expect(reqUrl).toBe("https://openrouter.ai/api/v1/chat/completions");
    const body = JSON.parse(init.body as string);
    expect(body.model).toBe("a:free");
    expect(body.models).toEqual(["a:free", "b:free", "c:free"]);
    expect(body.provider).toMatchObject({ sort: "throughput", allow_fallbacks: true });

    const headers = init.headers as Record<string, string>;
    expect(headers.authorization).toBe("Bearer or-key");
    expect(headers["HTTP-Referer"]).toBeTruthy();
    expect(headers["X-Title"]).toBe("fortune-llm gateway");
  });

  it("throws when configured with an empty model list", async () => {
    await expect(callOpenRouter({ apiKey: "k", models: [] }, baseReq())).rejects.toThrow(
      /no models/i,
    );
  });
});
