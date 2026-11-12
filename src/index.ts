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
  callGeminiModels,
  resolveGeminiModels,
  DEFAULT_GEMINI_MODEL,
  DEFAULT_GEMINI_FALLBACK_MODELS,
  resolveGeminiKeys,
} from "./gemini.js";
import { callGroq, DEFAULT_GROQ_MODEL } from "./groq.js";
import { callCerebras, DEFAULT_CEREBRAS_MODEL } from "./cerebras.js";
import {
  callGitHubModels,
  DEFAULT_GITHUB_MODELS_MODEL,
} from "./github-models.js";
import { callMistral, DEFAULT_MISTRAL_MODEL } from "./mistral.js";
import {
  callOpenRouter,
  resolveOpenRouterModels,
} from "./openrouter.js";
import {
  getCircuitState,
  isQuotaError,
  resolveTripDurationMs,
  tripCircuit,
  tripDurationFor,
  type CircuitState,
} from "./circuit-breaker.js";
import {
  computeCacheKey,
  isCacheable,
  readCache,
  resolveCacheTtlSeconds,
  writeCache,
} from "./cache.js";
import { synthesizeAnthropicSSE } from "./sse.js";
import { resolveProviderKey, type ByokProvider } from "./byok.js";
import type { StatsEvent } from "./stats-do.js";
import type { CachedResponse } from "./cache-do.js";
// Re-export the DO classes so the Workers runtime can find them.
export { StatsDO } from "./stats-do.js";
export { CacheDO } from "./cache-do.js";
import {
  checkRateLimit,
  getRateLimitScope,
  resolveConsumerRateLimit,
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
  /** Cerebras API key (https://cloud.cerebras.ai/platform). Free tier, very fast. */
  CEREBRAS_API_KEY?: string;
  /** OpenRouter API key (https://openrouter.ai/keys). Free tier via :free models. */
  OPENROUTER_API_KEY?: string;
  /** GitHub Personal Access Token. Free tier via https://models.github.ai. */
  GITHUB_MODELS_API_KEY?: string;
  /** Mistral La Plateforme key. Free experimental tier — https://console.mistral.ai. */
  MISTRAL_API_KEY?: string;
  GATEWAY_TOKEN?: string;
  DEFAULT_OSS_MODEL?: string;
  DEFAULT_GEMINI_MODEL?: string;
  /** Comma-separated Gemini models tried after DEFAULT_GEMINI_MODEL's quota is spent. */
  GEMINI_FALLBACK_MODELS?: string;
  DEFAULT_GROQ_MODEL?: string;
  DEFAULT_CEREBRAS_MODEL?: string;
  DEFAULT_GITHUB_MODELS_MODEL?: string;
  DEFAULT_MISTRAL_MODEL?: string;
  /** Comma-separated OpenRouter model fallback list. */
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
  /**
   * Durable Object that owns the stats counters. Atomic increments via
   * SQL UPSERT inside the DO actor — no lost updates, no read lag.
   * Optional only because tests-without-bindings shouldn't crash; in
   * prod this binding is always present.
   */
  STATS_DO?: DurableObjectNamespace;
  /**
   * Durable Object that owns the response cache. Strongly consistent
   * reads-after-writes — rapid-fire identical requests hit the cache
   * immediately, unlike the prior KV-backed cache where the second
   * read could miss for up to 60s of edge-cache propagation.
   */
  CACHE_DO?: DurableObjectNamespace;
}

type StatsDOStub = {
  recordEvents(e: StatsEvent[], consumer: string | null): Promise<void>;
  getStats(): Promise<{
    date: string;
    totals: {
      requests: number;
      cache_hits: number;
      cache_misses: number;
      rate_limited: number;
      errors: number;
    };
    per_tier: Record<string, { ok: number; fail: number }>;
    per_consumer: Record<
      string,
      {
        totals: {
          requests: number;
          cache_hits: number;
          cache_misses: number;
          rate_limited: number;
          errors: number;
        };
        per_tier: Record<string, { ok: number; fail: number }>;
      }
    >;
  }>;
};

type CacheDOStub = {
  read(key: string): Promise<CachedResponse | null>;
  write(key: string, entry: CachedResponse, ttlSeconds: number): Promise<void>;
};

let circuitMissingWarned = false;

// ── Route handlers ───────────────────────────────────────────────────────────

async function handleStats(request: Request, env: Env): Promise<Response> {
  const statsAuth = authenticate(request, env.GATEWAY_TOKEN);
  if (!statsAuth.ok) {
    return jsonError(
      statsAuth.status,
      "authentication_error",
      statsAuth.message,
    );
  }
  const stub = env.STATS_DO
    ? (env.STATS_DO.get(
        env.STATS_DO.idFromName("stats-singleton"),
      ) as unknown as StatsDOStub)
    : null;
  if (!stub) {
    return new Response(
      JSON.stringify({ error: "STATS_DO binding not configured" }),
      {
        status: 503,
        headers: {
          "content-type": "application/json",
          "access-control-allow-origin": "*",
        },
      },
    );
  }
  // /stats reads from the DO so data is strongly consistent with all
  // prior recordEvents calls — no 60s KV edge-cache lag.
  const stats = await stub.getStats();
  const total = stats.totals.requests || 1;
  const cacheTotal =
    stats.totals.cache_hits + stats.totals.cache_misses || 1;
  return new Response(
    JSON.stringify({
      ...stats,
      derived: {
        cache_hit_rate: stats.totals.cache_hits / cacheTotal,
        error_rate: stats.totals.errors / total,
        rate_limited_rate: stats.totals.rate_limited / total,
      },
    }),
    {
      headers: {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
      },
    },
  );
}

function handleHealthz(env: Env): Response {
  const geminiKeys = resolveGeminiKeys(
    env.GEMINI_API_KEYS,
    env.GEMINI_API_KEY,
  );
  const openRouterModels = resolveOpenRouterModels(
    env.DEFAULT_OPENROUTER_MODELS,
  );
  return new Response(
    JSON.stringify({
      ok: true,
      oss_model: env.DEFAULT_OSS_MODEL ?? "@cf/google/gemma-4-26b-a4b-it",
      gemini_model: env.DEFAULT_GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL,
      gemini_fallback_models:
        env.GEMINI_FALLBACK_MODELS ?? DEFAULT_GEMINI_FALLBACK_MODELS,
      groq_model: env.DEFAULT_GROQ_MODEL ?? DEFAULT_GROQ_MODEL,
      cerebras_model: env.DEFAULT_CEREBRAS_MODEL ?? DEFAULT_CEREBRAS_MODEL,
      github_models_model:
        env.DEFAULT_GITHUB_MODELS_MODEL ?? DEFAULT_GITHUB_MODELS_MODEL,
      mistral_model: env.DEFAULT_MISTRAL_MODEL ?? DEFAULT_MISTRAL_MODEL,
      openrouter_models: openRouterModels,
      backends: {
        "workers-ai": true, // bound by the AI binding, always available
        gemini: geminiKeys.length > 0,
        // Omit gemini_key_count — the exact number of configured API keys
        // is operational detail that needn't be visible to unauthenticated
        // callers scanning the public /healthz probe.
        groq: Boolean(env.GROQ_API_KEY),
        cerebras: Boolean(env.CEREBRAS_API_KEY),
        "github-models": Boolean(env.GITHUB_MODELS_API_KEY),
        mistral: Boolean(env.MISTRAL_API_KEY),
        openrouter: Boolean(env.OPENROUTER_API_KEY),
        // Do not expose whether a paid Anthropic key is configured to
        // unauthenticated callers — its presence is a billing secret.
        // Operators can check wrangler secrets or authenticate to infer it.
      },
    }),
    {
      headers: {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
      },
    },
  );
}

// ── Request pipeline helpers ─────────────────────────────────────────────────

// Rate limit scoped by `x-fortune-consumer` header (when set) or by IP.
// Per-consumer scoping prevents one runaway consumer from draining everyone
// else's quota. Per-consumer overrides via RATE_LIMIT_PER_MIN_<CONSUMER> env.
// Fails open if KV is unavailable. Returns a 429 Response on limit breach,
// null when the request is allowed through.
async function applyRateLimit(
  request: Request,
  env: Env,
  statsEvents: StatsEvent[],
  flushStats: () => void,
): Promise<Response | null> {
  const globalRateLimit = resolveRateLimitPerMin(env.RATE_LIMIT_PER_MIN);
  if (globalRateLimit === 0) return null;
  const { scope, kind } = getRateLimitScope(request);
  const effectiveLimit =
    kind === "consumer"
      ? resolveConsumerRateLimit(
          scope,
          globalRateLimit,
          env as unknown as Record<string, string | undefined>,
        )
      : globalRateLimit;
  const decision = await checkRateLimit(scope, effectiveLimit, env.CIRCUIT);
  if (decision.allowed) return null;
  statsEvents.push({ kind: "request" }, { kind: "rate_limited" });
  flushStats();
  return new Response(
    JSON.stringify({
      type: "error",
      error: {
        type: "rate_limit_error",
        message:
          `Rate limit exceeded: ${decision.count}/${decision.limit}` +
          ` req/min for ${kind}=${scope}.` +
          ` Retry in ${decision.retryAfterSeconds}s.`,
      },
    }),
    {
      status: 429,
      headers: {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        // Expose rate-limit diagnostic headers to cross-origin clients.
        "access-control-expose-headers": EXPOSE_HEADERS,
        "retry-after": String(decision.retryAfterSeconds),
        "x-fortune-llm-rate-limit": `${decision.count}/${decision.limit}`,
        "x-fortune-llm-rate-scope": `${kind}:${scope}`,
      },
    },
  );
}

// Reads and size-checks the request body. Returns `{ rawBody }` on success
// or a 413 Response on oversize. GET/HEAD callers get rawBody "".
async function readBody(
  request: Request,
): Promise<{ rawBody: string } | Response> {
  if (request.method === "GET" || request.method === "HEAD") {
    return { rawBody: "" };
  }
  const clRaw = request.headers.get("content-length");
  // Use Number() instead of parseInt() so scientific-notation values like
  // "1e7" are parsed to their full numeric value (parseInt("1e7") === 1,
  // defeating the fast-reject for any Content-Length expressed that way).
  const contentLength = clRaw !== null ? Number(clRaw) : NaN;
  if (Number.isFinite(contentLength) && contentLength > 1_048_576) {
    return jsonError(
      413,
      "request_too_large",
      "Request body exceeds 1 MB limit",
    );
  }
  const rawBody = await request.text();
  // Enforce the limit on the actual body size when Content-Length was absent
  // or non-numeric — a missing header must not be a bypass vector.
  // Use byteLength (not .length) so multi-byte UTF-8 characters don't
  // allow payloads larger than 1 MiB to slip through the character-count check.
  if (new TextEncoder().encode(rawBody).byteLength > 1_048_576) {
    return jsonError(
      413,
      "request_too_large",
      "Request body exceeds 1 MB limit",
    );
  }
  return { rawBody };
}

// Validates the /v1/messages request shape. Returns null when valid,
// or a 400 Response describing the first field violation.
function validateMessages(
  parsed: AnthropicMessagesRequest,
): Response | null {
  if (!parsed || !Array.isArray(parsed.messages)) {
    return jsonError(
      400,
      "invalid_request_error",
      "Field 'messages' must be an array.",
    );
  }
  if (parsed.messages.length === 0) {
    return jsonError(
      400,
      "invalid_request_error",
      "Field 'messages' must not be empty.",
    );
  }
  if (
    typeof parsed.max_tokens !== "number" ||
    !Number.isFinite(parsed.max_tokens) ||
    !Number.isInteger(parsed.max_tokens) ||
    parsed.max_tokens <= 0
  ) {
    return jsonError(
      400,
      "invalid_request_error",
      "Field 'max_tokens' must be a positive integer.",
    );
  }
  if (parsed.max_tokens > 65536) {
    return jsonError(
      400,
      "invalid_request_error",
      "Field 'max_tokens' must not exceed 65536.",
    );
  }
  return null;
}

// ── Cache helpers ─────────────────────────────────────────────────────────────

// Reads the cache for `cacheKey` and, on a hit, returns a ready Response.
// Returns null on a cache miss or a malformed SSE entry (fall through to
// fresh dispatch). Does NOT push cache_miss on a miss — the dispatch loop
// does that when the upstream call completes, avoiding a double-count.
async function lookupAndServeCache(
  cacheKey: string,
  cacheStub: CacheDOStub | null,
  envCircuit: KVNamespace | undefined,
  wantsStream: boolean,
  statsEvents: StatsEvent[],
  flushStats: () => void,
): Promise<Response | null> {
  const cached = cacheStub
    ? await cacheStub.read(cacheKey)
    : await readCache(cacheKey, envCircuit);
  if (!cached) return null;

  const cacheAge = Math.floor((Date.now() - cached.cachedAt) / 1000);
  const diagnosticHeaders: Record<string, string> = {
    "access-control-allow-origin": "*",
    "x-fortune-llm-route": cached.tier,
    "x-fortune-llm-model": cached.model,
    "x-fortune-llm-cache": wantsStream ? "hit-stream" : "hit",
    "x-fortune-llm-cache-age-s": String(cacheAge),
  };

  if (wantsStream) {
    // Reconstruct message + emit as Anthropic SSE.
    let msg;
    try {
      msg = JSON.parse(cached.body);
    } catch {
      msg = null;
    }
    if (msg && typeof msg === "object" && Array.isArray(msg.content)) {
      const sseResp = synthesizeAnthropicSSE(msg);
      const headers = new Headers(sseResp.headers);
      for (const [k, v] of Object.entries(diagnosticHeaders)) {
        headers.set(k, v);
      }
      statsEvents.push({ kind: "cache_hit" });
      flushStats();
      return new Response(sseResp.body, { status: 200, headers });
    }
    // Malformed cached entry — fall through to a fresh dispatch.
    return null;
  }

  statsEvents.push({ kind: "cache_hit" });
  flushStats();
  return new Response(cached.body, {
    status: 200,
    headers: {
      "content-type": "application/json",
      ...diagnosticHeaders,
    },
  });
}

// ── Chain dispatch ────────────────────────────────────────────────────────────

type ChainResult =
  | { ok: true; response: Response }
  | {
      ok: false;
      errors: Array<{ tier: BackendKind; error: string }>;
      skipped: Array<{ tier: BackendKind; until: number }>;
    };

// Iterates `chain.tiers`, skipping circuit-open backends and catching per-tier
// errors. On the first success, optionally caches the response body and
// synthesizes SSE if `wantsStream`. Returns `{ ok: true, response }` on the
// first successful tier, or `{ ok: false, errors, skipped }` once all tiers
// are exhausted.
async function runChain(
  chain: RouteChain,
  env: Env,
  effectiveParsed: AnthropicMessagesRequest,
  request: Request,
  rawBody: string,
  cacheEligible: boolean,
  cacheKey: string | null,
  wantsStream: boolean,
  cacheStub: CacheDOStub | null,
  cacheTtl: number,
  tripDurationMs: number,
  statsEvents: StatsEvent[],
  flushStats: () => void,
): Promise<ChainResult> {
  const errors: Array<{ tier: BackendKind; error: string }> = [];
  const skipped: Array<{ tier: BackendKind; until: number }> = [];

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
      const resp = await invokeTier(
        tier,
        env,
        effectiveParsed,
        request,
        rawBody,
      );
      const headers = new Headers(resp.headers);
      headers.set("x-fortune-llm-route", tier);
      headers.set("x-fortune-llm-chain", chain.tiers.join(","));
      headers.set("x-fortune-llm-reason", chain.reason);
      if (errors.length > 0) {
        headers.set("x-fortune-llm-prior-errors", formatPriorErrorsHeader(errors));
      }
      if (skipped.length > 0) {
        headers.set("x-fortune-llm-skipped", formatSkippedHeader(skipped));
      }
      headers.set("access-control-allow-origin", "*");
      headers.set("access-control-expose-headers", EXPOSE_HEADERS);

      // Cache write: read the body so we can both store it and return
      // it. Only cache 200 JSON responses — SSE responses (which only
      // appear when caching was ineligible, i.e. !cacheEligible) skip
      // this branch and stream through normally.
      if (
        cacheEligible &&
        cacheKey &&
        resp.status === 200 &&
        (resp.headers.get("content-type") ?? "").includes("application/json")
      ) {
        const bodyText = await resp.text();
        const modelLabel = headers.get("x-fortune-llm-model") ?? tier;
        // For stream requests: parse before caching so we never persist a
        // non-JSON body. A parse failure returns 502 without touching cache,
        // letting the next identical request try a fresh upstream call.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        let streamMsg: any = undefined;
        if (wantsStream) {
          try {
            streamMsg = JSON.parse(bodyText);
          } catch {
            statsEvents.push({ kind: "error" });
            flushStats();
            return {
              ok: true,
              response: jsonError(
                502,
                "upstream_error",
                `${tier} returned non-JSON body; cannot synthesize SSE stream`,
              ),
            };
          }
        }
        const cacheEntry = {
          body: bodyText,
          tier,
          model: modelLabel,
          cachedAt: Date.now(),
        };
        if (cacheStub) {
          await cacheStub.write(cacheKey, cacheEntry, cacheTtl);
        } else {
          await writeCache(cacheKey, cacheEntry, cacheTtl, env.CIRCUIT);
        }
        if (wantsStream) {
          const sseResp = synthesizeAnthropicSSE(streamMsg);
          const sseHeaders = new Headers(sseResp.headers);
          headers.forEach((v, k) => sseHeaders.set(k, v));
          sseHeaders.set("content-type", "text/event-stream; charset=utf-8");
          sseHeaders.set("x-fortune-llm-cache", "miss-stored-stream");
          statsEvents.push({ kind: "cache_miss" }, { kind: "tier_ok", tier });
          flushStats();
          return {
            ok: true,
            response: new Response(sseResp.body, {
              status: 200,
              headers: sseHeaders,
            }),
          };
        }
        headers.set("x-fortune-llm-cache", "miss-stored");
        statsEvents.push({ kind: "cache_miss" }, { kind: "tier_ok", tier });
        flushStats();
        return {
          ok: true,
          response: new Response(bodyText, {
            status: 200,
            statusText: resp.statusText,
            headers,
          }),
        };
      }

      if (cacheEligible) {
        headers.set("x-fortune-llm-cache", "miss-skip");
        statsEvents.push({ kind: "cache_miss" });
      }
      statsEvents.push({ kind: "tier_ok", tier });
      flushStats();
      return {
        ok: true,
        response: new Response(resp.body, {
          status: resp.status,
          statusText: resp.statusText,
          headers,
        }),
      };
    } catch (err) {
      const msg = errorMessage(err);
      console.error(`fortune-llm: tier ${tier} failed:`, msg);
      errors.push({ tier, error: msg });
      statsEvents.push({ kind: "tier_fail", tier });
      // Trip the circuit on quota / rate-limit signals so subsequent
      // requests skip this tier outright instead of repeatedly
      // burning a fetch on a guaranteed failure.
      // Note: do NOT push to `skipped` here — this tier was actually tried
      // (it belongs in `errors`). `skipped` is reserved for tiers whose
      // circuit was already open at the top of this request loop.
      if (tier !== "anthropic" && isQuotaError(err)) {
        // A BYOK request exhausting the caller's own donated key must not
        // disable this tier for all other users on the shared gateway pool.
        if (!request.headers.get(`x-fortune-byok-${tier}`)) {
          await tripCircuit(
            tier,
            env.CIRCUIT,
            tripDurationFor(msg, tripDurationMs),
            msg,
          );
        }
      }
    }
  }

  return { ok: false, errors, skipped };
}

// Builds the final 503 response after all chain tiers are exhausted.
// Distinguishes pure quota (all skipped, no errors), mixed (some skipped
// + some errored), and all-errored cases.
function chainExhaustedResponse(
  errors: Array<{ tier: BackendKind; error: string }>,
  skipped: Array<{ tier: BackendKind; until: number }>,
  chain: RouteChain,
  statsEvents: StatsEvent[],
  flushStats: () => void,
): Response {
  const headers: HeadersInit = {
    "content-type": "application/json",
    "access-control-allow-origin": "*",
    "access-control-expose-headers": EXPOSE_HEADERS,
  };
  if (skipped.length > 0) {
    (headers as Record<string, string>)["x-fortune-llm-skipped"] =
      formatSkippedHeader(skipped);
  }
  statsEvents.push({ kind: "error" });
  flushStats();

  if (skipped.length > 0 && errors.length === 0) {
    // Every tier in the chain had its circuit open — pure quota state.
    return new Response(
      JSON.stringify({
        type: "error",
        error: {
          type: "quota_exhausted",
          message: `all backends unavailable: ${skipped
            .map(
              (s) =>
                `${s.tier} (open until ${new Date(s.until).toISOString()})`,
            )
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
            ...skipped.map(
              (s) =>
                `${s.tier} (open until ${new Date(s.until).toISOString()})`,
            ),
            ...errors.map((e) => `${e.tier} (failed)`),
          ].join(", ")}`,
        },
      }),
      { status: 503, headers },
    );
  }

  return jsonError(
    503,
    "all_backends_failed",
    `All free backends in chain [${chain.tiers.join(",")}] failed: ${errors
      .map((e) => e.tier)
      .join(", ")}.`,
  );
}

// ── Main handler ─────────────────────────────────────────────────────────────

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<Response> {
    const url = new URL(request.url);
    // Stats events accumulate over the lifetime of this request and
    // are flushed to the StatsDO via ctx.waitUntil so the write doesn't
    // add latency to the response path. The DO serializes increments
    // through a single actor — no read-modify-write races, no KV cache lag.
    const statsEvents: StatsEvent[] = [];
    // Consumer identity: captured once at request entry so every stats
    // event lands in the same per-consumer bucket even if the header is
    // removed or rewritten mid-flow.
    const consumerHeader = request.headers.get("x-fortune-consumer");
    const flushStats = () => {
      if (statsEvents.length === 0) return;
      const stub = env.STATS_DO
        ? (env.STATS_DO.get(
            env.STATS_DO.idFromName("stats-singleton"),
          ) as unknown as StatsDOStub)
        : null;
      if (stub) {
        const batch = statsEvents.slice();
        ctx.waitUntil(stub.recordEvents(batch, consumerHeader));
      }
      statsEvents.length = 0;
    };

    // GET /stats — auth-protected observability (strongly consistent via DO).
    if (request.method === "GET" && url.pathname === "/stats") {
      return handleStats(request, env);
    }

    if (request.method === "GET" && url.pathname === "/healthz") {
      return handleHealthz(env);
    }

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() });
    }

    if (!url.pathname.startsWith("/v1/")) {
      return jsonError(
        404,
        "not_found",
        `No route for ${request.method} ${url.pathname}`,
      );
    }

    const auth = authenticate(request, env.GATEWAY_TOKEN);
    if (!auth.ok) {
      return jsonError(auth.status, "authentication_error", auth.message);
    }

    const rateLimitResp = await applyRateLimit(
      request,
      env,
      statsEvents,
      flushStats,
    );
    if (rateLimitResp) return rateLimitResp;

    const bodyResult = await readBody(request);
    if (bodyResult instanceof Response) return bodyResult;
    const { rawBody } = bodyResult;

    // Non-messages endpoints (count_tokens, batches, etc.) are Anthropic-
    // specific and don't translate cleanly. Only forward when Anthropic
    // is configured; otherwise reject so callers see the policy clearly.
    if (!(request.method === "POST" && url.pathname === "/v1/messages")) {
      if (!env.ANTHROPIC_API_KEY) {
        return jsonError(
          501,
          "not_implemented",
          `Endpoint ${url.pathname} is Anthropic-only and ANTHROPIC_API_KEY` +
            ` is not configured. Free backends only support /v1/messages.`,
        );
      }
      try {
        return withCors(
          await forwardToAnthropic(request, rawBody, env.ANTHROPIC_API_KEY),
        );
      } catch (err) {
        console.error("fortune-llm: Anthropic upstream failed:", errorMessage(err));
        return jsonError(
          502,
          "upstream_error",
          "Anthropic upstream failed. Please try again.",
        );
      }
    }

    let parsed: AnthropicMessagesRequest;
    try {
      parsed = JSON.parse(rawBody) as AnthropicMessagesRequest;
    } catch {
      return jsonError(
        400,
        "invalid_request_error",
        "Body must be valid JSON.",
      );
    }
    const validationErr = validateMessages(parsed);
    if (validationErr) return validationErr;

    // Past validation — this is a real /v1/messages request.
    statsEvents.push({ kind: "request" });

    // Cache lookup: a hit short-circuits the entire chain. Saves the
    // full provider round-trip — biggest single quota multiplier on
    // top of the multi-provider chain itself.
    //
    // Streaming variant: cached entries are always stored as JSON
    // (Anthropic message shape). When the consumer asked for `stream: true`,
    // we synthesize SSE from the cached JSON via sse.ts. This way one cache
    // entry serves both stream and non-stream callers for the same prompt.
    const wantsStream = parsed.stream === true;
    const cacheTtl = resolveCacheTtlSeconds(env.CACHE_TTL_SECONDS);
    const cacheEligible = cacheTtl > 0 && isCacheable(parsed);
    // Cache lives in a DO actor — reads/writes are strongly consistent
    // so rapid-fire identical requests hit cache immediately. Fall back to
    // KV cache when CACHE_DO isn't present (local dev without the migration).
    const cacheStub: CacheDOStub | null = env.CACHE_DO
      ? (env.CACHE_DO.get(
          env.CACHE_DO.idFromName("cache-singleton"),
        ) as unknown as CacheDOStub)
      : null;
    let cacheKey: string | null = null;
    if (cacheEligible) {
      cacheKey = await computeCacheKey(parsed);
      const hit = await lookupAndServeCache(
        cacheKey,
        cacheStub,
        env.CIRCUIT,
        wantsStream,
        statsEvents,
        flushStats,
      );
      if (hit) return hit;
    }

    // Streaming + cacheable + cache miss: force the upstream call to
    // non-streaming so we can buffer the full JSON, cache it, AND
    // synthesize SSE back to the consumer. This is the entire reason
    // stream-from-cache works — by canonicalizing on non-stream upstream,
    // one cache entry serves both stream and non-stream callers indefinitely.
    const effectiveParsed: AnthropicMessagesRequest =
      cacheEligible && wantsStream ? { ...parsed, stream: false } : parsed;

    const chain: RouteChain = decideRoute(parsed, {
      anthropicFallback: Boolean(env.ANTHROPIC_API_KEY),
    });

    if (!env.CIRCUIT && !circuitMissingWarned) {
      circuitMissingWarned = true;
      console.warn(
        "fortune-llm: CIRCUIT KV namespace not bound — circuit breaker disabled. " +
          "Run `wrangler kv namespace create CIRCUIT` and add the id to wrangler.toml.",
      );
    }
    const tripDurationMs = resolveTripDurationMs(env.CIRCUIT_TRIP_DURATION_MS);

    const result = await runChain(
      chain,
      env,
      effectiveParsed,
      request,
      rawBody,
      cacheEligible,
      cacheKey,
      wantsStream,
      cacheStub,
      cacheTtl,
      tripDurationMs,
      statsEvents,
      flushStats,
    );
    if (result.ok) return result.response;
    return chainExhaustedResponse(
      result.errors,
      result.skipped,
      chain,
      statsEvents,
      flushStats,
    );
  },
};

// ── Tier dispatcher ───────────────────────────────────────────────────────────

/**
 * Tiers that take one API key and one model and speak the OpenAI-compatible
 * wire format. Each entry names the env key (also used in the "not
 * configured" error) and how to pick the model.
 */
const SINGLE_MODEL_TIERS = {
  groq: {
    keyName: "GROQ_API_KEY",
    model: (env: Env) => env.DEFAULT_GROQ_MODEL ?? DEFAULT_GROQ_MODEL,
    call: callGroq,
  },
  cerebras: {
    keyName: "CEREBRAS_API_KEY",
    model: (env: Env) => env.DEFAULT_CEREBRAS_MODEL ?? DEFAULT_CEREBRAS_MODEL,
    call: callCerebras,
  },
  "github-models": {
    keyName: "GITHUB_MODELS_API_KEY",
    model: (env: Env) =>
      env.DEFAULT_GITHUB_MODELS_MODEL ?? DEFAULT_GITHUB_MODELS_MODEL,
    call: callGitHubModels,
  },
  mistral: {
    keyName: "MISTRAL_API_KEY",
    model: (env: Env) => env.DEFAULT_MISTRAL_MODEL ?? DEFAULT_MISTRAL_MODEL,
    call: callMistral,
  },
} as const satisfies Partial<
  Record<
    BackendKind & ByokProvider,
    {
      keyName: keyof Env;
      model: (env: Env) => string;
      call: (
        opts: { apiKey: string; model: string },
        req: AnthropicMessagesRequest,
      ) => Promise<Response>;
    }
  >
>;

function isSingleModelTier(
  tier: BackendKind,
): tier is keyof typeof SINGLE_MODEL_TIERS {
  return Object.hasOwn(SINGLE_MODEL_TIERS, tier);
}

/**
 * Copy a backend response with the gateway's per-tier diagnostic headers
 * added. A key the caller supplied (BYOK) is flagged so they can tell
 * their own quota was used.
 */
function withTierHeaders(
  resp: Response,
  extra: Record<string, string>,
  keySource?: "byok" | "shared",
): Response {
  const headers = new Headers(resp.headers);
  for (const [name, value] of Object.entries(extra)) headers.set(name, value);
  if (keySource === "byok") headers.set("x-fortune-llm-byok", "true");
  return new Response(resp.body, {
    status: resp.status,
    statusText: resp.statusText,
    headers,
  });
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
    return withTierHeaders(resp, { "x-fortune-llm-model": model });
  }
  // BYOK: every non-workers-ai tier supports per-request key override via
  // x-fortune-byok-<provider>: <key>. Header wins; shared env key is the
  // fallback. `null` → tier not configured (handler throws and dispatcher
  // advances to the next tier).
  const resolveKey = (provider: ByokProvider, shared: string | undefined) =>
    resolveProviderKey(provider, shared, request.headers);

  if (tier === "gemini") {
    const byok = resolveKey("gemini", undefined);
    const keys = byok
      ? [byok.key]
      : resolveGeminiKeys(env.GEMINI_API_KEYS, env.GEMINI_API_KEY);
    if (keys.length === 0) {
      throw new Error("GEMINI_API_KEY not configured");
    }
    const models = resolveGeminiModels(
      env.DEFAULT_GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL,
      env.GEMINI_FALLBACK_MODELS,
    );
    const { resp, model } = await callGeminiModels(
      keys,
      models,
      parsed,
      isQuotaError,
    );
    return withTierHeaders(
      resp,
      {
        "x-fortune-llm-model": model,
        ...(keys.length > 1
          ? { "x-fortune-llm-gemini-keys": String(keys.length) }
          : {}),
      },
      byok?.source,
    );
  }
  if (isSingleModelTier(tier)) {
    const spec = SINGLE_MODEL_TIERS[tier];
    const resolved = resolveKey(tier, env[spec.keyName]);
    if (!resolved) throw new Error(`${spec.keyName} not configured`);
    const model = spec.model(env);
    const resp = await spec.call({ apiKey: resolved.key, model }, parsed);
    return withTierHeaders(
      resp,
      { "x-fortune-llm-model": model },
      resolved.source,
    );
  }
  if (tier === "openrouter") {
    const resolved = resolveKey("openrouter", env.OPENROUTER_API_KEY);
    if (!resolved) throw new Error("OPENROUTER_API_KEY not configured");
    const models = resolveOpenRouterModels(env.DEFAULT_OPENROUTER_MODELS);
    const resp = await callOpenRouter(
      { apiKey: resolved.key, models },
      parsed,
    );
    return withTierHeaders(
      resp,
      {
        "x-fortune-llm-model": models[0] ?? "unknown",
        "x-fortune-llm-openrouter-models": String(models.length),
      },
      resolved.source,
    );
  }
  // tier === "anthropic" — paid escape valve. Auto-appended when ANTHROPIC_API_KEY
  // is configured (free chain failed → paid), or reached via explicit metadata
  // override. BYOK applies here too — a consumer can pass their own Anthropic
  // key to offload billing.
  const anth = resolveKey("anthropic", env.ANTHROPIC_API_KEY);
  if (!anth) throw new Error("ANTHROPIC_API_KEY not configured");
  const resp = await forwardToAnthropic(request, rawBody, anth.key);
  return anth.source === "byok" ? withTierHeaders(resp, {}, "byok") : resp;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

// Diagnostic response headers exposed to cross-origin clients.
const EXPOSE_HEADERS =
  "x-fortune-llm-route, x-fortune-llm-chain, x-fortune-llm-reason, " +
  "x-fortune-llm-prior-errors, x-fortune-llm-skipped, x-fortune-llm-model, " +
  "x-fortune-llm-gemini-keys, x-fortune-llm-openrouter-models, " +
  "x-fortune-llm-cache, x-fortune-llm-cache-age-s, x-fortune-llm-rate-limit, " +
  "x-fortune-llm-rate-scope, x-fortune-llm-byok";

/**
 * Names each failed tier with only the leading `<Label> <status>` part of its
 * error (the text before the first colon). The rest is the raw upstream body,
 * which can carry account details, so it stays in `console.error`.
 * Control characters make `Headers.set` throw, which would discard the
 * successful tier's response, so collapse them to spaces.
 */
export function formatPriorErrorsHeader(
  errors: Array<{ tier: BackendKind; error: string }>,
): string {
  return errors
    .map((e) => `${e.tier}:${(e.error.split(":")[0] ?? "").slice(0, 80)}`)
    .join(" | ")
    .replace(/[\x00-\x1f\x7f]+/g, " ");
}

function formatSkippedHeader(
  skipped: Array<{ tier: BackendKind; until: number }>,
): string {
  return skipped
    .map((s) => `${s.tier}:${new Date(s.until).toISOString()}`)
    .join(",");
}

function jsonError(status: number, type: string, message: string): Response {
  return new Response(
    JSON.stringify({ type: "error", error: { type, message } }),
    {
      status,
      headers: {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-expose-headers": EXPOSE_HEADERS,
      },
    },
  );
}

function withCors(resp: Response): Response {
  const headers = new Headers(resp.headers);
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-expose-headers", EXPOSE_HEADERS);
  return new Response(resp.body, {
    status: resp.status,
    statusText: resp.statusText,
    headers,
  });
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function corsHeaders(): HeadersInit {
  return {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    // Include all custom request headers that cross-origin clients send so
    // the preflight passes. Without x-fortune-consumer the browser blocks the
    // per-consumer rate-limit header; without x-fortune-byok-* BYOK calls
    // from browser apps always fail the preflight and never reach the Worker.
    "access-control-allow-headers":
      "x-api-key, anthropic-version, anthropic-beta, content-type, " +
      "authorization, x-fortune-consumer, x-fortune-byok-anthropic, " +
      "x-fortune-byok-groq, x-fortune-byok-cerebras, x-fortune-byok-gemini, " +
      "x-fortune-byok-openrouter, x-fortune-byok-github-models, " +
      "x-fortune-byok-mistral",
    "access-control-max-age": "86400",
    // Expose the diagnostic headers so cross-origin clients can read them.
    "access-control-expose-headers": EXPOSE_HEADERS,
  };
}
