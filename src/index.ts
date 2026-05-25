/**
 * fortune-llm — Anthropic-Messages-API-compatible gateway for Cloudflare
 * Workers. Routes to free backends by default (Workers AI, then Gemini)
 * and fails loudly when the free chain is exhausted. Paid Anthropic is
 * available only via explicit `metadata.fortune_route="anthropic"`.
 *
 * Routes:
 *   GET  /healthz         — liveness probe (no auth)
 *   POST /v1/messages     — runs the routed fallback chain
 *   *    /v1/*            — every other Anthropic endpoint forwarded only
 *                           when explicitly opted into Anthropic (legacy
 *                           endpoints like count_tokens / batches).
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { authenticate } from "./auth.js";
import { decideRoute, type BackendKind, type RouteChain } from "./route.js";
import { forwardToAnthropic } from "./anthropic.js";
import { callWorkersAi } from "./workers-ai.js";
import {
  callGeminiWithRotation,
  DEFAULT_GEMINI_MODEL,
  resolveGeminiKeys,
} from "./gemini.js";
import { callGroq, DEFAULT_GROQ_MODEL } from "./groq.js";
import {
  callOpenRouter,
  DEFAULT_OPENROUTER_MODELS,
  resolveOpenRouterModels,
} from "./openrouter.js";
import {
  getCircuitState,
  isQuotaError,
  resolveTripDurationMs,
  tripCircuit,
  type CircuitState,
} from "./circuit-breaker.js";
import {
  computeCacheKey,
  isCacheable,
  readCache,
  resolveCacheTtlSeconds,
  writeCache,
} from "./cache.js";
import {
  checkRateLimit,
  getClientIp,
  resolveRateLimitPerMin,
} from "./rate-limit.js";

interface Env {
  AI: { run(model: string, input: unknown): Promise<unknown> };
  ANTHROPIC_API_KEY?: string;
  /** Legacy single-key form for Gemini. */
  GEMINI_API_KEY?: string;
  /** Comma-separated multi-key form for Gemini. Each key gives independent quota. */
  GEMINI_API_KEYS?: string;
  /** Groq API key (https://console.groq.com/keys). Free tier with native tool use. */
  GROQ_API_KEY?: string;
  /** OpenRouter API key (https://openrouter.ai/keys). Free tier via :free models. */
  OPENROUTER_API_KEY?: string;
  GATEWAY_TOKEN?: string;
  DEFAULT_OSS_MODEL?: string;
  DEFAULT_GEMINI_MODEL?: string;
  DEFAULT_GROQ_MODEL?: string;
  /** Comma-separated OpenRouter model fallback list. Defaults to a hand-picked diverse list. */
  DEFAULT_OPENROUTER_MODELS?: string;
  // Per-backend circuit breaker, exact-match response cache, and per-IP
  // rate-limiting counters all share this KV namespace under different
  // key prefixes (`circuit:`, `cache:`, `rate:`). When unbound (e.g.
  // local dev), each subsystem no-ops gracefully.
  CIRCUIT?: KVNamespace;
  // Override the default circuit-breaker trip duration (1h, in ms).
  CIRCUIT_TRIP_DURATION_MS?: string;
  /** Cache TTL in seconds. Default 24h (86400). Set to 0 to disable. */
  CACHE_TTL_SECONDS?: string;
  /** Per-IP request cap per minute. Default 200. Set to 0 to disable. */
  RATE_LIMIT_PER_MIN?: string;
}

let circuitMissingWarned = false;

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/healthz") {
      const geminiKeys = resolveGeminiKeys(env.GEMINI_API_KEYS, env.GEMINI_API_KEY);
      const openRouterModels = resolveOpenRouterModels(env.DEFAULT_OPENROUTER_MODELS);
      return new Response(
        JSON.stringify({
          ok: true,
          oss_model: env.DEFAULT_OSS_MODEL ?? "@cf/google/gemma-4-26b-a4b-it",
          gemini_model: env.DEFAULT_GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL,
          groq_model: env.DEFAULT_GROQ_MODEL ?? DEFAULT_GROQ_MODEL,
          openrouter_models: openRouterModels,
          backends: {
            "workers-ai": true, // bound by the AI binding, always available
            gemini: geminiKeys.length > 0,
            // Omit gemini_key_count — the exact number of configured API keys
            // is operational detail that needn't be visible to unauthenticated
            // callers scanning the public /healthz probe.
            groq: Boolean(env.GROQ_API_KEY),
            openrouter: Boolean(env.OPENROUTER_API_KEY),
            // Do not expose whether a paid Anthropic key is configured to
            // unauthenticated callers — its presence is a billing secret.
            // Operators can check wrangler secrets or authenticate to infer it.
          },
        }),
        { headers: { "content-type": "application/json", "access-control-allow-origin": "*" } },
      );
    }

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    if (!url.pathname.startsWith("/v1/")) {
      return jsonError(404, "not_found", `No route for ${request.method} ${url.pathname}`);
    }

    const auth = authenticate(request, env.GATEWAY_TOKEN);
    if (!auth.ok) {
      return jsonError(auth.status, "authentication_error", auth.message);
    }

    // Per-IP rate limit. Sits between auth and upstream dispatch so
    // bad citizens don't trip our circuit breakers or burn free quota.
    // Fails open if KV is unavailable so we don't accidentally block
    // legitimate traffic on infra trouble.
    const rateLimitPerMin = resolveRateLimitPerMin(env.RATE_LIMIT_PER_MIN);
    if (rateLimitPerMin > 0) {
      const ip = getClientIp(request);
      const decision = await checkRateLimit(ip, rateLimitPerMin, env.CIRCUIT);
      if (!decision.allowed) {
        return new Response(
          JSON.stringify({
            type: "error",
            error: {
              type: "rate_limit_error",
              message: `Rate limit exceeded: ${decision.count}/${decision.limit} req/min from ${ip}. Retry in ${decision.retryAfterSeconds}s.`,
            },
          }),
          {
            status: 429,
            headers: {
              "content-type": "application/json",
              "access-control-allow-origin": "*",
              "retry-after": String(decision.retryAfterSeconds),
              "x-fortune-llm-rate-limit": `${decision.count}/${decision.limit}`,
            },
          },
        );
      }
    }

    let rawBody = "";
    if (request.method !== "GET" && request.method !== "HEAD") {
      const clRaw = request.headers.get("content-length");
      // Use Number() instead of parseInt() so scientific-notation values like
      // "1e7" are parsed to their full numeric value (parseInt("1e7") === 1,
      // defeating the fast-reject for any Content-Length expressed that way).
      const contentLength = clRaw !== null ? Number(clRaw) : NaN;
      // Fast-reject on a declared oversized body before reading it.
      if (Number.isFinite(contentLength) && contentLength > 1_048_576) {
        return jsonError(413, "request_too_large", "Request body exceeds 1 MB limit");
      }
      rawBody = await request.text();
      // Enforce the limit on the actual body size when Content-Length was absent
      // or non-numeric — a missing header must not be a bypass vector.
      // Use byteLength (not .length) so multi-byte UTF-8 characters don't
      // allow payloads larger than 1 MiB to slip through the character-count check.
      if (new TextEncoder().encode(rawBody).byteLength > 1_048_576) {
        return jsonError(413, "request_too_large", "Request body exceeds 1 MB limit");
      }
    }

    // Non-messages endpoints (count_tokens, batches, etc.) are Anthropic-
    // specific and don't translate cleanly. Only forward when Anthropic
    // is configured; otherwise reject so callers see the policy clearly.
    if (!(request.method === "POST" && url.pathname === "/v1/messages")) {
      if (!env.ANTHROPIC_API_KEY) {
        return jsonError(
          501,
          "not_implemented",
          `Endpoint ${url.pathname} is Anthropic-only and ANTHROPIC_API_KEY is not configured. Free backends only support /v1/messages.`,
        );
      }
      try {
        return withCors(await forwardToAnthropic(request, rawBody, env.ANTHROPIC_API_KEY));
      } catch (err) {
        return jsonError(502, "upstream_error", `Anthropic upstream failed: ${errorMessage(err)}`);
      }
    }

    let parsed: AnthropicMessagesRequest;
    try {
      parsed = JSON.parse(rawBody) as AnthropicMessagesRequest;
    } catch {
      return jsonError(400, "invalid_request_error", "Body must be valid JSON.");
    }
    if (!parsed || !Array.isArray(parsed.messages)) {
      return jsonError(400, "invalid_request_error", "Field 'messages' must be an array.");
    }
    if (parsed.messages.length === 0) {
      return jsonError(400, "invalid_request_error", "Field 'messages' must not be empty.");
    }
    if (
      typeof parsed.max_tokens !== "number" ||
      !Number.isFinite(parsed.max_tokens) ||
      !Number.isInteger(parsed.max_tokens) ||
      parsed.max_tokens <= 0
    ) {
      return jsonError(400, "invalid_request_error", "Field 'max_tokens' must be a positive integer.");
    }
    if (parsed.max_tokens > 65536) {
      return jsonError(400, "invalid_request_error", "Field 'max_tokens' must not exceed 65536.");
    }

    // Cache lookup: a hit short-circuits the entire chain. Saves the
    // full provider round-trip — biggest single quota multiplier on
    // top of the multi-provider chain itself.
    const cacheTtl = resolveCacheTtlSeconds(env.CACHE_TTL_SECONDS);
    const cacheEligible = cacheTtl > 0 && isCacheable(parsed);
    let cacheKey: string | null = null;
    if (cacheEligible) {
      cacheKey = await computeCacheKey(parsed);
      const cached = await readCache(cacheKey, env.CIRCUIT);
      if (cached) {
        return new Response(cached.body, {
          status: 200,
          headers: {
            "content-type": "application/json",
            "access-control-allow-origin": "*",
            "x-fortune-llm-route": cached.tier,
            "x-fortune-llm-model": cached.model,
            "x-fortune-llm-cache": "hit",
            "x-fortune-llm-cache-age-s": String(Math.floor((Date.now() - cached.cachedAt) / 1000)),
          },
        });
      }
    }

    const chain: RouteChain = decideRoute(parsed, {
      anthropicFallback: Boolean(env.ANTHROPIC_API_KEY),
    });
    const errors: Array<{ tier: BackendKind; error: string }> = [];
    const skipped: Array<{ tier: BackendKind; until: number }> = [];

    if (!env.CIRCUIT && !circuitMissingWarned) {
      circuitMissingWarned = true;
      console.warn(
        "fortune-llm: CIRCUIT KV namespace not bound — circuit breaker disabled. " +
          "Run `wrangler kv namespace create CIRCUIT` and add the id to wrangler.toml.",
      );
    }
    const tripDurationMs = resolveTripDurationMs(env.CIRCUIT_TRIP_DURATION_MS);

    for (let i = 0; i < chain.tiers.length; i++) {
      const tier = chain.tiers[i];
      if (!tier) continue;

      // Skip tiers whose circuit is open. Anthropic is never circuit-
      // gated — it's opt-in and paid; if the user explicitly asked for
      // it, give it through regardless.
      if (tier !== "anthropic") {
        const state: CircuitState = await getCircuitState(tier, env.CIRCUIT);
        if (state.open && typeof state.until === "number") {
          skipped.push({ tier, until: state.until });
          continue;
        }
      }

      try {
        const resp = await invokeTier(tier, env, parsed, request, rawBody);
        const headers = new Headers(resp.headers);
        headers.set("x-fortune-llm-route", tier);
        headers.set("x-fortune-llm-chain", chain.tiers.join(","));
        headers.set("x-fortune-llm-reason", chain.reason);
        if (errors.length > 0) {
          headers.set(
            "x-fortune-llm-prior-errors",
            errors.map((e) => `${e.tier}:${e.error.slice(0, 80)}`).join(" | "),
          );
        }
        if (skipped.length > 0) {
          headers.set("x-fortune-llm-skipped", formatSkippedHeader(skipped));
        }
        headers.set("access-control-allow-origin", "*");
        headers.set("access-control-expose-headers", EXPOSE_HEADERS);

        // Cache write: read the body so we can both store it and return
        // it. Only cache 200 JSON responses (status is from invokeTier;
        // we don't cache SSE streams or upstream errors).
        if (
          cacheEligible &&
          cacheKey &&
          resp.status === 200 &&
          (resp.headers.get("content-type") ?? "").includes("application/json")
        ) {
          const bodyText = await resp.text();
          const modelLabel = headers.get("x-fortune-llm-model") ?? tier;
          await writeCache(
            cacheKey,
            { body: bodyText, tier, model: modelLabel, cachedAt: Date.now() },
            cacheTtl,
            env.CIRCUIT,
          );
          headers.set("x-fortune-llm-cache", "miss-stored");
          return new Response(bodyText, { status: 200, statusText: resp.statusText, headers });
        }

        if (cacheEligible) {
          headers.set("x-fortune-llm-cache", "miss-skip");
        }
        return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
      } catch (err) {
        const msg = errorMessage(err);
        errors.push({ tier, error: msg });
        // Trip the circuit on quota / rate-limit signals so subsequent
        // requests skip this tier outright instead of repeatedly
        // burning a fetch on a guaranteed failure.
        // Note: do NOT push to `skipped` here — this tier was actually tried
        // (it belongs in `errors`). `skipped` is reserved for tiers whose
        // circuit was already open at the top of this request loop, so the
        // final error message doesn't double-report the same tier.
        if (tier !== "anthropic" && isQuotaError(err)) {
          await tripCircuit(tier, env.CIRCUIT, tripDurationMs, msg);
        }
        // continue to next tier
      }
    }

    // Chain exhausted — fail loudly. No silent escalation to paid.
    // If at least one tier was skipped because its circuit was open,
    // shape the response as a quota-exhausted error so consumers can
    // distinguish "all backends are throttled" from "all backends are
    // broken" and back off accordingly.
    const headers: HeadersInit = {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-expose-headers": EXPOSE_HEADERS,
    };
    if (skipped.length > 0) {
      (headers as Record<string, string>)["x-fortune-llm-skipped"] = formatSkippedHeader(skipped);
    }

    if (skipped.length > 0 && errors.length === 0) {
      // Every tier in the chain had its circuit open — pure quota state.
      return new Response(
        JSON.stringify({
          type: "error",
          error: {
            type: "quota_exhausted",
            message: `all backends unavailable: ${skipped
              .map((s) => `${s.tier} (open until ${new Date(s.until).toISOString()})`)
              .join(", ")}`,
          },
        }),
        { status: 503, headers },
      );
    }

    if (skipped.length > 0) {
      // Mixed: some tiers skipped (quota), some failed (other reasons).
      // Still a 503 quota_exhausted because the dominant signal is that
      // the backends are throttled — but include the failed-tier errors
      // so debugging is possible.
      return new Response(
        JSON.stringify({
          type: "error",
          error: {
            type: "quota_exhausted",
            message: `all backends unavailable: ${[
              ...skipped.map((s) => `${s.tier} (open until ${new Date(s.until).toISOString()})`),
              ...errors.map((e) => `${e.tier} (error: ${e.error})`),
            ].join(", ")}`,
          },
        }),
        { status: 503, headers },
      );
    }

    return jsonError(
      503,
      "all_backends_failed",
      `All free backends in chain [${chain.tiers.join(",")}] failed. ${errors
        .map((e) => `${e.tier}: ${e.error}`)
        .join(" | ")}`,
    );
  },
};

function formatSkippedHeader(skipped: Array<{ tier: BackendKind; until: number }>): string {
  return skipped.map((s) => `${s.tier}:${new Date(s.until).toISOString()}`).join(",");
}

async function invokeTier(
  tier: BackendKind,
  env: Env,
  parsed: AnthropicMessagesRequest,
  request: Request,
  rawBody: string,
): Promise<Response> {
  if (tier === "workers-ai") {
    const model = env.DEFAULT_OSS_MODEL ?? "@cf/google/gemma-4-26b-a4b-it";
    const resp = await callWorkersAi(env.AI, model, parsed);
    const headers = new Headers(resp.headers);
    headers.set("x-fortune-llm-model", model);
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
  }
  if (tier === "gemini") {
    const keys = resolveGeminiKeys(env.GEMINI_API_KEYS, env.GEMINI_API_KEY);
    if (keys.length === 0) {
      throw new Error("GEMINI_API_KEY not configured");
    }
    const model = env.DEFAULT_GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL;
    const resp = await callGeminiWithRotation(keys, model, parsed, isQuotaError);
    const headers = new Headers(resp.headers);
    headers.set("x-fortune-llm-model", model);
    if (keys.length > 1) {
      headers.set("x-fortune-llm-gemini-keys", String(keys.length));
    }
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
  }
  if (tier === "groq") {
    if (!env.GROQ_API_KEY) {
      throw new Error("GROQ_API_KEY not configured");
    }
    const model = env.DEFAULT_GROQ_MODEL ?? DEFAULT_GROQ_MODEL;
    const resp = await callGroq({ apiKey: env.GROQ_API_KEY, model }, parsed);
    const headers = new Headers(resp.headers);
    headers.set("x-fortune-llm-model", model);
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
  }
  if (tier === "openrouter") {
    if (!env.OPENROUTER_API_KEY) {
      throw new Error("OPENROUTER_API_KEY not configured");
    }
    const models = resolveOpenRouterModels(env.DEFAULT_OPENROUTER_MODELS);
    const resp = await callOpenRouter({ apiKey: env.OPENROUTER_API_KEY, models }, parsed);
    const headers = new Headers(resp.headers);
    // The response includes the model OpenRouter actually picked; surface
    // it as x-fortune-llm-model so consumers can see which fallback won.
    headers.set("x-fortune-llm-model", models[0] as string);
    headers.set("x-fortune-llm-openrouter-models", String(models.length));
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
  }
  // tier === "anthropic" — paid escape valve. Auto-appended when ANTHROPIC_API_KEY
  // is configured (free chain failed → paid), or reached via explicit metadata
  // override.
  if (!env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY not configured");
  }
  return forwardToAnthropic(request, rawBody, env.ANTHROPIC_API_KEY);
}

// Diagnostic response headers exposed to cross-origin clients.
const EXPOSE_HEADERS =
  "x-fortune-llm-route, x-fortune-llm-chain, x-fortune-llm-reason, " +
  "x-fortune-llm-prior-errors, x-fortune-llm-skipped, x-fortune-llm-model, " +
  "x-fortune-llm-gemini-keys";

function jsonError(status: number, type: string, message: string): Response {
  return new Response(JSON.stringify({ type: "error", error: { type, message } }), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      "access-control-expose-headers": EXPOSE_HEADERS,
    },
  });
}

function withCors(resp: Response): Response {
  const headers = new Headers(resp.headers);
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-expose-headers", EXPOSE_HEADERS);
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function corsHeaders(): HeadersInit {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "x-api-key, anthropic-version, anthropic-beta, content-type, authorization",
    "access-control-max-age": "86400",
    // Expose the diagnostic headers so cross-origin clients can read them.
    "access-control-expose-headers": EXPOSE_HEADERS,
  };
}
