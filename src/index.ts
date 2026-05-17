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
import { callGemini, DEFAULT_GEMINI_MODEL } from "./gemini.js";

interface Env {
  AI: { run(model: string, input: unknown): Promise<unknown> };
  ANTHROPIC_API_KEY?: string;
  GEMINI_API_KEY?: string;
  GATEWAY_TOKEN?: string;
  DEFAULT_OSS_MODEL?: string;
  DEFAULT_GEMINI_MODEL?: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/healthz") {
      return new Response(
        JSON.stringify({
          ok: true,
          oss_model: env.DEFAULT_OSS_MODEL ?? "@cf/meta/llama-4-scout-17b-16e-instruct",
          gemini_model: env.DEFAULT_GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL,
          gemini_configured: Boolean(env.GEMINI_API_KEY),
          anthropic_configured: Boolean(env.ANTHROPIC_API_KEY),
        }),
        { headers: { "content-type": "application/json" } },
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

    let rawBody = "";
    if (request.method !== "GET" && request.method !== "HEAD") {
      rawBody = await request.text();
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
    if (
      typeof parsed.max_tokens !== "number" ||
      !Number.isFinite(parsed.max_tokens) ||
      !Number.isInteger(parsed.max_tokens) ||
      parsed.max_tokens <= 0
    ) {
      return jsonError(400, "invalid_request_error", "Field 'max_tokens' must be a positive integer.");
    }

    const chain: RouteChain = decideRoute(parsed);
    const errors: Array<{ tier: BackendKind; error: string }> = [];

    for (let i = 0; i < chain.tiers.length; i++) {
      const tier = chain.tiers[i];
      if (!tier) continue;
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
        headers.set("access-control-allow-origin", "*");
        return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
      } catch (err) {
        errors.push({ tier, error: errorMessage(err) });
        // continue to next tier
      }
    }

    // Chain exhausted — fail loudly. No silent escalation to paid.
    return jsonError(
      503,
      "all_backends_failed",
      `All free backends in chain [${chain.tiers.join(",")}] failed. ${errors
        .map((e) => `${e.tier}: ${e.error}`)
        .join(" | ")}`,
    );
  },
};

async function invokeTier(
  tier: BackendKind,
  env: Env,
  parsed: AnthropicMessagesRequest,
  request: Request,
  rawBody: string,
): Promise<Response> {
  if (tier === "workers-ai") {
    const model = env.DEFAULT_OSS_MODEL ?? "@cf/meta/llama-4-scout-17b-16e-instruct";
    const resp = await callWorkersAi(env.AI, model, parsed);
    const headers = new Headers(resp.headers);
    headers.set("x-fortune-llm-model", model);
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
  }
  if (tier === "gemini") {
    if (!env.GEMINI_API_KEY) {
      throw new Error("GEMINI_API_KEY not configured");
    }
    const model = env.DEFAULT_GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL;
    const resp = await callGemini({ apiKey: env.GEMINI_API_KEY, model }, parsed);
    const headers = new Headers(resp.headers);
    headers.set("x-fortune-llm-model", model);
    return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
  }
  // tier === "anthropic" — paid escape valve. Only reached via explicit metadata override.
  if (!env.ANTHROPIC_API_KEY) {
    throw new Error("ANTHROPIC_API_KEY not configured");
  }
  return forwardToAnthropic(request, rawBody, env.ANTHROPIC_API_KEY);
}

function jsonError(status: number, type: string, message: string): Response {
  return new Response(JSON.stringify({ type: "error", error: { type, message } }), {
    status,
    headers: { "content-type": "application/json", "access-control-allow-origin": "*" },
  });
}

function withCors(resp: Response): Response {
  const headers = new Headers(resp.headers);
  headers.set("access-control-allow-origin", "*");
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
  };
}
