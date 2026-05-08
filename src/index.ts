/**
 * fortune-llm — Anthropic-Messages-API-compatible gateway for Cloudflare
 * Workers. See README.md for the full architecture.
 *
 * Routes:
 *   GET  /healthz         — liveness probe (no auth)
 *   POST /v1/messages     — Anthropic Messages API; routes to Workers AI
 *                           by default, escalates to Anthropic when the
 *                           request needs capabilities OSS can't deliver.
 *   *    /v1/*            — every other Anthropic endpoint is forwarded
 *                           as-is (count_tokens, batches, etc.).
 */

import type { AnthropicMessagesRequest } from "./types.js";
import { authenticate } from "./auth.js";
import { decideRoute, type RouteDecision } from "./route.js";
import { forwardToAnthropic } from "./anthropic.js";
import { callWorkersAi } from "./workers-ai.js";

interface Env {
  AI: { run(model: string, input: unknown): Promise<unknown> };
  ANTHROPIC_API_KEY?: string;
  GATEWAY_TOKEN?: string;
  DEFAULT_OSS_MODEL?: string;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "GET" && url.pathname === "/healthz") {
      return new Response(
        JSON.stringify({
          ok: true,
          model: env.DEFAULT_OSS_MODEL ?? "@cf/meta/llama-4-scout-17b-16e-instruct",
          anthropic_fallback: Boolean(env.ANTHROPIC_API_KEY),
        }),
        { headers: { "content-type": "application/json" } },
      );
    }

    // CORS preflight — Anthropic SDK doesn't actually trigger one, but
    // browser-side direct callers might.
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(),
      });
    }

    if (!url.pathname.startsWith("/v1/")) {
      return new Response(
        JSON.stringify({
          type: "error",
          error: { type: "not_found", message: `No route for ${request.method} ${url.pathname}` },
        }),
        { status: 404, headers: { "content-type": "application/json" } },
      );
    }

    const auth = authenticate(request, env.GATEWAY_TOKEN);
    if (!auth.ok) {
      return new Response(
        JSON.stringify({
          type: "error",
          error: { type: "authentication_error", message: auth.message },
        }),
        { status: auth.status, headers: { "content-type": "application/json" } },
      );
    }

    // Read body once so we can both inspect (route) and forward (Anthropic).
    let rawBody = "";
    if (request.method !== "GET" && request.method !== "HEAD") {
      rawBody = await request.text();
    }

    // Anything other than POST /v1/messages — just forward to Anthropic.
    // Those endpoints (count_tokens, batches, etc.) aren't worth
    // re-implementing on Workers AI.
    if (!(request.method === "POST" && url.pathname === "/v1/messages")) {
      if (!env.ANTHROPIC_API_KEY) {
        return jsonError(
          501,
          "anthropic_unavailable",
          `Endpoint ${url.pathname} is forwarded to Anthropic but ANTHROPIC_API_KEY is not configured.`,
        );
      }
      return forwardToAnthropic(request, rawBody, env.ANTHROPIC_API_KEY);
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
    if (typeof parsed.max_tokens !== "number") {
      return jsonError(400, "invalid_request_error", "Field 'max_tokens' is required.");
    }

    const decision: RouteDecision = decideRoute(parsed);

    if (decision.kind === "anthropic") {
      if (!env.ANTHROPIC_API_KEY) {
        return jsonError(
          503,
          "anthropic_unavailable",
          `Request needed Anthropic (${decision.reason}) but ANTHROPIC_API_KEY is not configured on the gateway.`,
        );
      }
      const resp = await forwardToAnthropic(request, rawBody, env.ANTHROPIC_API_KEY);
      // Tag the response so consumer apps can observe routing in dev.
      const headers = new Headers(resp.headers);
      headers.set("x-fortune-llm-route", "anthropic");
      headers.set("x-fortune-llm-reason", decision.reason);
      return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
    }

    // Workers AI path
    try {
      const model = env.DEFAULT_OSS_MODEL ?? "@cf/meta/llama-4-scout-17b-16e-instruct";
      const resp = await callWorkersAi(env.AI, model, parsed);
      const headers = new Headers(resp.headers);
      headers.set("x-fortune-llm-route", "workers-ai");
      headers.set("x-fortune-llm-model", model);
      headers.set("x-fortune-llm-reason", decision.reason);
      return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
    } catch (err) {
      // If Workers AI itself errored (rate limit, model temporarily down)
      // and Anthropic is configured, transparently retry on Anthropic.
      if (env.ANTHROPIC_API_KEY) {
        const fallback = await forwardToAnthropic(request, rawBody, env.ANTHROPIC_API_KEY);
        const headers = new Headers(fallback.headers);
        headers.set("x-fortune-llm-route", "anthropic");
        headers.set("x-fortune-llm-reason", `workers-ai-failed: ${errorMessage(err)}`);
        return new Response(fallback.body, {
          status: fallback.status,
          statusText: fallback.statusText,
          headers,
        });
      }
      return jsonError(502, "upstream_error", `Workers AI failed: ${errorMessage(err)}`);
    }
  },
};

function jsonError(status: number, type: string, message: string): Response {
  return new Response(JSON.stringify({ type: "error", error: { type, message } }), {
    status,
    headers: { "content-type": "application/json" },
  });
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
