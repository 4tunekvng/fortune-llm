import { describe, it, expect } from "vitest";
import { scrubReservedMetadata } from "../src/anthropic.js";

describe("scrubReservedMetadata", () => {
  it("strips metadata.fortune_route, leaves other fields intact", () => {
    const body = JSON.stringify({
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
      metadata: { fortune_route: "anthropic", user_id: "u_123" },
    });
    const out = JSON.parse(scrubReservedMetadata(body));
    expect(out.metadata).toEqual({ user_id: "u_123" });
    expect(out.model).toBe("claude-sonnet-4-6");
    expect(out.messages).toHaveLength(1);
  });

  it("drops the metadata object entirely when it becomes empty after scrub", () => {
    const body = JSON.stringify({
      messages: [],
      metadata: { fortune_route: "anthropic" },
    });
    const out = JSON.parse(scrubReservedMetadata(body));
    expect("metadata" in out).toBe(false);
  });

  it("returns the original string when no reserved keys are present", () => {
    const body = JSON.stringify({ messages: [], metadata: { user_id: "u_123" } });
    expect(scrubReservedMetadata(body)).toBe(body);
  });

  it("returns the original string when metadata is absent entirely", () => {
    const body = JSON.stringify({ messages: [{ role: "user", content: "hi" }] });
    expect(scrubReservedMetadata(body)).toBe(body);
  });

  it("returns the original string for malformed JSON (let upstream reject)", () => {
    const broken = "{not valid json";
    expect(scrubReservedMetadata(broken)).toBe(broken);
  });

  it("handles metadata that isn't an object (defensive — passes through)", () => {
    const body = JSON.stringify({ messages: [], metadata: "not-an-object" });
    expect(scrubReservedMetadata(body)).toBe(body);
  });

  it("handles null metadata (defensive — passes through)", () => {
    const body = JSON.stringify({ messages: [], metadata: null });
    expect(scrubReservedMetadata(body)).toBe(body);
  });

  it("strips metadata.fortune_require_tools, leaves user_id intact", () => {
    const body = JSON.stringify({
      messages: [],
      metadata: { fortune_require_tools: true, user_id: "u_123" },
    });
    const out = JSON.parse(scrubReservedMetadata(body));
    expect(out.metadata).toEqual({ user_id: "u_123" });
  });

  it("strips both reserved keys when both are present", () => {
    const body = JSON.stringify({
      messages: [],
      metadata: { fortune_route: "anthropic", fortune_require_tools: true, user_id: "u_123" },
    });
    const out = JSON.parse(scrubReservedMetadata(body));
    expect(out.metadata).toEqual({ user_id: "u_123" });
  });

  it("strips fortune_cache so Anthropic does not reject it as an extra input", () => {
    const body = JSON.stringify({
      messages: [],
      metadata: { fortune_cache: true, user_id: "u_123" },
    });
    const out = JSON.parse(scrubReservedMetadata(body));
    expect(out.metadata).toEqual({ user_id: "u_123" });
    expect(out.metadata).not.toHaveProperty("fortune_cache");
  });

  it("strips fortune_no_cache so Anthropic does not reject it as an extra input", () => {
    const body = JSON.stringify({
      messages: [],
      metadata: { fortune_no_cache: true, user_id: "u_123" },
    });
    const out = JSON.parse(scrubReservedMetadata(body));
    expect(out.metadata).toEqual({ user_id: "u_123" });
    expect(out.metadata).not.toHaveProperty("fortune_no_cache");
  });

  it("drops metadata entirely when only fortune_cache and fortune_no_cache are present", () => {
    const body = JSON.stringify({
      messages: [],
      metadata: { fortune_cache: true, fortune_no_cache: false },
    });
    const out = JSON.parse(scrubReservedMetadata(body));
    expect("metadata" in out).toBe(false);
  });
});
