/**
 * Exact-match response cache. Saves a non-streaming Anthropic-shaped
 * response keyed by a SHA-256 over the cacheable request fields, and
 * returns it on subsequent matching requests until TTL.
 *
 * Why bother:
 *   - Public-facing chat apps see surprising repetition — greetings,
 *     FAQs, "what can you do", boilerplate first turns. A 20-40% cache
 *     hit rate is plausible and that is 20-40% off every provider's
 *     daily quota for free.
 *   - When the free chain is saturating, every cached hit is one less
 *     request that needs to reach a provider at all, pushing the day
 *     we'd escalate to Anthropic further out.
 *
 * Eligibility (default policy):
 *   - request.stream is false (streaming requires SSE synthesis we're
 *     not building yet; phase 3 can add stream-from-cache)
 *   - temperature is 0 or absent (callers want determinism)
 *     OR metadata.fortune_cache === true (explicit opt-in)
 *   - no tools[] (tool_use responses are too sensitive to in-context
 *     state to safely return verbatim)
 *   - metadata.fortune_no_cache !== true (explicit opt-out)
 *
 * Hash inputs:
 *   model, system, messages, tools, tool_choice, temperature, top_p,
 *   top_k, stop_sequences, max_tokens. Skipped: stream, metadata
 *   (cosmetic), output_config (already routed straight to anthropic).
 */

import type { AnthropicMessagesRequest } from "./types.js";

export const DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60; // 24h
const KEY_PREFIX = "cache:";

export interface CachedResponse {
  body: string;
  /** Tier that originally served this response (workers-ai, gemini, …). */
  tier: string;
  /** Model id that originally served this response. */
  model: string;
  /** Original Date.now() at write time, for diagnostic header. */
  cachedAt: number;
}

/**
 * Returns true when the request is eligible for caching under the
 * default policy. Callers can short-circuit with explicit metadata.
 */
export function isCacheable(req: AnthropicMessagesRequest): boolean {
  const meta = req.metadata as
    | { fortune_no_cache?: boolean; fortune_cache?: boolean; fortune_require_tools?: boolean }
    | undefined;

  // Explicit opt-out wins over everything.
  if (meta?.fortune_no_cache === true) return false;
  // Streaming requests aren't cached in this version. Synthesizing SSE
  // from a buffered cached response is doable but deferred to phase 3.
  if (req.stream === true) return false;
  // Tool-using calls aren't cached. Tool responses are tightly coupled
  // to the consumer's in-context state and replaying a stale tool_use
  // can break the downstream agent loop.
  if (Array.isArray(req.tools) && req.tools.length > 0) return false;
  // require_tools is a routing knob; if set we want fresh.
  if (meta?.fortune_require_tools === true) return false;

  // Explicit opt-in always caches.
  if (meta?.fortune_cache === true) return true;

  // Default heuristic: cache when the caller asked for determinism.
  // temperature 0 (or 0.0) → cache. Any temperature > 0 → fresh.
  // Default temperature on Anthropic is 1.0 so apps that didn't set
  // temperature get fresh responses.
  if (typeof req.temperature === "number" && req.temperature === 0) return true;

  return false;
}

/**
 * Compute the deterministic cache key for a request. Only fields that
 * affect the response are hashed.
 *
 * Returns a short SHA-256-derived string prefixed with `cache:` so
 * it can share a KV namespace with the circuit breaker without
 * collision.
 */
export async function computeCacheKey(req: AnthropicMessagesRequest): Promise<string> {
  const canonical = JSON.stringify({
    model: req.model,
    system: req.system ?? null,
    messages: req.messages,
    tools: req.tools ?? null,
    tool_choice: req.tool_choice ?? null,
    temperature: typeof req.temperature === "number" ? req.temperature : null,
    top_p: typeof req.top_p === "number" ? req.top_p : null,
    top_k: typeof req.top_k === "number" ? req.top_k : null,
    stop_sequences: Array.isArray(req.stop_sequences) ? req.stop_sequences : null,
    max_tokens: req.max_tokens,
  });
  const enc = new TextEncoder().encode(canonical);
  const hashBuf = await crypto.subtle.digest("SHA-256", enc);
  const hashArr = Array.from(new Uint8Array(hashBuf));
  // base16 first 32 chars (16 bytes) — collision-safe at our scale, and
  // shorter keys make KV ops cheaper.
  const hex = hashArr.map((b) => b.toString(16).padStart(2, "0")).join("");
  return `${KEY_PREFIX}${hex.slice(0, 32)}`;
}

/**
 * Read a cached response. Returns null on miss, malformed entry, or KV
 * unavailable — never throws (cache misses must degrade silently).
 */
export async function readCache(
  key: string,
  kv: KVNamespace | undefined,
): Promise<CachedResponse | null> {
  if (!kv) return null;
  let raw: string | null;
  try {
    raw = await kv.get(key);
  } catch {
    return null;
  }
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as CachedResponse;
    if (typeof parsed.body !== "string") return null;
    return parsed;
  } catch {
    return null;
  }
}

/**
 * Write a response into the cache. Failures are swallowed — the request
 * succeeded; failing the response on a cache-write error would be silly.
 */
export async function writeCache(
  key: string,
  entry: CachedResponse,
  ttlSeconds: number,
  kv: KVNamespace | undefined,
): Promise<void> {
  if (!kv) return;
  const ttl = Math.max(60, Math.floor(ttlSeconds));
  try {
    await kv.put(key, JSON.stringify(entry), { expirationTtl: ttl });
  } catch {
    // ignore; this was best-effort
  }
}

/**
 * Resolve the cache TTL from env, with bounds. Default 24h, capped at
 * 30 days, floored at 60s (KV minimum).
 */
export function resolveCacheTtlSeconds(envValue: string | undefined): number {
  if (!envValue) return DEFAULT_CACHE_TTL_SECONDS;
  const n = Number(envValue);
  if (!Number.isFinite(n) || n <= 0) return DEFAULT_CACHE_TTL_SECONDS;
  const MAX = 30 * 24 * 60 * 60;
  return Math.min(Math.max(60, Math.floor(n)), MAX);
}
