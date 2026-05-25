import { describe, it, expect } from "vitest";
import { extractByokKey, resolveProviderKey } from "../src/byok.js";

function makeHeaders(map: Record<string, string>): Headers {
  const h = new Headers();
  for (const [k, v] of Object.entries(map)) h.set(k, v);
  return h;
}

describe("extractByokKey", () => {
  it("returns the header value when present and valid", () => {
    const h = makeHeaders({ "x-fortune-byok-groq": "gsk_abcdefghijk" });
    expect(extractByokKey(h, "groq")).toBe("gsk_abcdefghijk");
  });

  it("returns null when the header is missing", () => {
    expect(extractByokKey(makeHeaders({}), "groq")).toBeNull();
  });

  it("trims surrounding whitespace", () => {
    const h = makeHeaders({ "x-fortune-byok-groq": "  gsk_abcdefghijk  " });
    expect(extractByokKey(h, "groq")).toBe("gsk_abcdefghijk");
  });

  it("returns null for too-short values (looks bogus, not worth using)", () => {
    const h = makeHeaders({ "x-fortune-byok-groq": "abc" });
    expect(extractByokKey(h, "groq")).toBeNull();
  });

  it("returns null for excessively-long values", () => {
    const h = makeHeaders({ "x-fortune-byok-groq": "x".repeat(600) });
    expect(extractByokKey(h, "groq")).toBeNull();
  });

  it("returns null when the value contains control characters (prevents header smuggling)", () => {
    // We exercise the regex directly via a synthetic Headers-like
    // mock so we can include a tab without Headers normalization
    // stripping it. The control-char guard is the second layer of
    // defense after the runtime's header validation.
    const fakeHeaders = {
      get: (name: string) => (name === "x-fortune-byok-groq" ? "gsk_abc\tdef_long_key" : null),
    } as unknown as Headers;
    expect(extractByokKey(fakeHeaders, "groq")).toBeNull();
  });

  it("supports all recognized providers", () => {
    const providers = ["anthropic", "groq", "cerebras", "gemini", "openrouter", "github-models", "mistral"] as const;
    for (const p of providers) {
      const h = makeHeaders({ [`x-fortune-byok-${p}`]: "valid_key_12345" });
      expect(extractByokKey(h, p)).toBe("valid_key_12345");
    }
  });
});

describe("resolveProviderKey", () => {
  it("returns BYOK key when both BYOK and shared key are present (BYOK wins)", () => {
    const h = makeHeaders({ "x-fortune-byok-anthropic": "sk-ant-USER_KEY_VALUE" });
    expect(resolveProviderKey("anthropic", "sk-ant-SHARED_GATEWAY", h)).toEqual({
      key: "sk-ant-USER_KEY_VALUE",
      source: "byok",
    });
  });

  it("falls back to shared key when BYOK header is absent", () => {
    expect(resolveProviderKey("anthropic", "sk-ant-SHARED", makeHeaders({}))).toEqual({
      key: "sk-ant-SHARED",
      source: "shared",
    });
  });

  it("falls back to shared key when BYOK header is invalid", () => {
    const h = makeHeaders({ "x-fortune-byok-anthropic": "x" }); // too short
    expect(resolveProviderKey("anthropic", "sk-ant-SHARED", h)).toEqual({
      key: "sk-ant-SHARED",
      source: "shared",
    });
  });

  it("returns null when neither BYOK nor shared key is present", () => {
    expect(resolveProviderKey("anthropic", undefined, makeHeaders({}))).toBeNull();
  });
});
