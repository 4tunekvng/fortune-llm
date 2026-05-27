import { describe, it, expect, vi } from "vitest";
import {
  computeCacheKey,
  isCacheable,
  readCache,
  writeCache,
  resolveCacheTtlSeconds,
  DEFAULT_CACHE_TTL_SECONDS,
} from "../src/cache.js";
import type { AnthropicMessagesRequest } from "../src/types.js";

const baseReq = (overrides: Partial<AnthropicMessagesRequest> = {}): AnthropicMessagesRequest => ({
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  messages: [{ role: "user", content: "hi" }],
  ...overrides,
});

describe("isCacheable", () => {
  it("caches by default when temperature is 0", () => {
    expect(isCacheable(baseReq({ temperature: 0 }))).toBe(true);
  });

  it("does NOT cache when temperature is unset (Anthropic default = 1.0)", () => {
    expect(isCacheable(baseReq())).toBe(false);
  });

  it("does NOT cache when temperature > 0", () => {
    expect(isCacheable(baseReq({ temperature: 0.7 }))).toBe(false);
  });

  it("caches when fortune_cache=true is set explicitly, even with temperature > 0", () => {
    expect(isCacheable(baseReq({ temperature: 0.7, metadata: { fortune_cache: true } }))).toBe(true);
  });

  it("does NOT cache when fortune_no_cache=true (explicit opt-out wins over fortune_cache)", () => {
    expect(
      isCacheable(
        baseReq({
          temperature: 0,
          metadata: { fortune_cache: true, fortune_no_cache: true },
        }),
      ),
    ).toBe(false);
  });

  it("DOES cache streaming requests now (phase 2.5 stream-from-cache)", () => {
    // We force non-stream upstream, cache the JSON, synthesize SSE on
    // the way back. From isCacheable's perspective, stream:true is fine
    // — the dispatcher handles the upstream-forcing.
    expect(isCacheable(baseReq({ temperature: 0, stream: true }))).toBe(true);
  });

  it("does NOT cache tool-using requests (response is too in-context-sensitive)", () => {
    expect(
      isCacheable(baseReq({ temperature: 0, tools: [{ name: "x", input_schema: { type: "object" } }] })),
    ).toBe(false);
  });

  it("does NOT cache when fortune_require_tools is set", () => {
    expect(
      isCacheable(
        baseReq({
          temperature: 0,
          metadata: { fortune_require_tools: true },
        }),
      ),
    ).toBe(false);
  });
});

describe("computeCacheKey", () => {
  it("produces a stable cache: prefixed key", async () => {
    const k1 = await computeCacheKey(baseReq({ temperature: 0 }));
    const k2 = await computeCacheKey(baseReq({ temperature: 0 }));
    expect(k1).toBe(k2);
    expect(k1.startsWith("cache:")).toBe(true);
    expect(k1.length).toBeLessThanOrEqual(40); // prefix + 32 hex chars
  });

  it("differs when the message content differs", async () => {
    const a = await computeCacheKey(
      baseReq({ temperature: 0, messages: [{ role: "user", content: "hi" }] }),
    );
    const b = await computeCacheKey(
      baseReq({ temperature: 0, messages: [{ role: "user", content: "hello" }] }),
    );
    expect(a).not.toBe(b);
  });

  it("differs when the model differs", async () => {
    const a = await computeCacheKey(baseReq({ temperature: 0, model: "claude-sonnet-4-6" }));
    const b = await computeCacheKey(baseReq({ temperature: 0, model: "claude-opus-4-7" }));
    expect(a).not.toBe(b);
  });

  it("is the SAME when only metadata (cosmetic) differs", async () => {
    const a = await computeCacheKey(
      baseReq({ temperature: 0, metadata: { fortune_cache: true } }),
    );
    const b = await computeCacheKey(
      baseReq({ temperature: 0, metadata: { fortune_cache: true, custom_field: "foo" } }),
    );
    expect(a).toBe(b);
  });

  it("differs when temperature differs", async () => {
    const a = await computeCacheKey(baseReq({ temperature: 0 }));
    const b = await computeCacheKey(baseReq({ temperature: 0.5 }));
    expect(a).not.toBe(b);
  });
});

describe("readCache / writeCache", () => {
  const makeKv = (initial: Map<string, string> = new Map()): KVNamespace => {
    const store = new Map(initial);
    return {
      async get(k: string) {
        return store.get(k) ?? null;
      },
      async put(k: string, v: string) {
        store.set(k, v);
      },
      async delete(k: string) {
        store.delete(k);
      },
      async list() {
        return { keys: [], list_complete: true, cacheStatus: null };
      },
      async getWithMetadata() {
        return { value: null, metadata: null, cacheStatus: null };
      },
    } as unknown as KVNamespace;
  };

  it("readCache returns null when KV is undefined", async () => {
    const out = await readCache("cache:x", undefined);
    expect(out).toBeNull();
  });

  it("readCache returns null on miss", async () => {
    const kv = makeKv();
    const out = await readCache("cache:not-there", kv);
    expect(out).toBeNull();
  });

  it("writes then reads back the same entry", async () => {
    const kv = makeKv();
    await writeCache(
      "cache:x",
      { body: '{"hi":1}', tier: "groq", model: "llama", cachedAt: 1000 },
      300,
      kv,
    );
    const out = await readCache("cache:x", kv);
    expect(out).toEqual({ body: '{"hi":1}', tier: "groq", model: "llama", cachedAt: 1000 });
  });

  it("readCache returns null on malformed JSON without throwing", async () => {
    const kv = makeKv(new Map([["cache:bad", "not-json"]]));
    const out = await readCache("cache:bad", kv);
    expect(out).toBeNull();
  });

  it("readCache returns null on KV read failure", async () => {
    const kv = {
      get: vi.fn().mockRejectedValue(new Error("kv down")),
    } as unknown as KVNamespace;
    const out = await readCache("cache:x", kv);
    expect(out).toBeNull();
  });

  it("writeCache swallows KV write failures silently", async () => {
    const kv = {
      put: vi.fn().mockRejectedValue(new Error("kv down")),
    } as unknown as KVNamespace;
    await expect(
      writeCache("cache:x", { body: "{}", tier: "x", model: "y", cachedAt: 0 }, 60, kv),
    ).resolves.toBeUndefined();
  });
});

describe("resolveCacheTtlSeconds", () => {
  it("returns the default when env value is undefined", () => {
    expect(resolveCacheTtlSeconds(undefined)).toBe(DEFAULT_CACHE_TTL_SECONDS);
  });

  it("returns the default for non-numeric env values", () => {
    expect(resolveCacheTtlSeconds("abc")).toBe(DEFAULT_CACHE_TTL_SECONDS);
  });

  it("floors at KV minimum (60s)", () => {
    expect(resolveCacheTtlSeconds("30")).toBe(60);
  });

  it("caps at 30 days", () => {
    expect(resolveCacheTtlSeconds(String(40 * 24 * 60 * 60))).toBe(30 * 24 * 60 * 60);
  });

  it("accepts a valid mid-range value", () => {
    expect(resolveCacheTtlSeconds("3600")).toBe(3600);
  });

  it("returns 0 when env value is '0' — disables caching entirely", () => {
    expect(resolveCacheTtlSeconds("0")).toBe(0);
  });
});
