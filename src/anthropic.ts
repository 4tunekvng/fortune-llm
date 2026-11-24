/**
 * Anthropic forward path. Reverse-proxies to api.anthropic.com using the
 * gateway's own ANTHROPIC_API_KEY secret.
 *
 * We *almost* forward the body as-is, but we have to scrub gateway-internal
 * metadata hints (`metadata.fortune_route`) and the sampling parameters
 * Claude no longer accepts before sending. Anthropic's API
 * rejects unknown metadata fields with `invalid_request_error: metadata.X:
 * Extra inputs are not permitted` — verified 2026-05-22 with sonnet-4-6.
 */

const UPSTREAM = "https://api.anthropic.com";

// The most output tokens any caller may ask for. A request over it is
// lowered to it, not refused: current Claude Code asks for more than this, and
// the cap exists only to stop a runaway bill (2147483647 was once accepted).
export const MAX_OUTPUT_TOKENS = 65536;

// Metadata keys the gateway reserves for its own routing. Stripped from
// the request body before it goes upstream so Anthropic doesn't reject
// them as "Extra inputs are not permitted".
const RESERVED_METADATA_KEYS = [
  "fortune_route",
  "fortune_require_tools",
  "fortune_cache",
  "fortune_no_cache",
];

// Sampling knobs Claude no longer takes. Claude 4.7+ models return a 400 for
// any non-default value, and API version 2026-10-21 (Python SDK 2.x) rejects
// them outright, even at the defaults
// (https://platform.claude.com/docs/en/api/migration-2026-10-21). Callers
// keep sending them for the free providers that honor them, and for the
// temperature-0 cache heuristic, so they are dropped only on this hop.
const SAMPLING_KEYS = ["temperature", "top_p", "top_k"];

export async function forwardToAnthropic(
  request: Request,
  rawBody: string,
  upstreamKey: string,
): Promise<Response> {
  const inUrl = new URL(request.url);
  // Preserve the path + query so /v1/messages, /v1/messages/count_tokens,
  // /v1/messages/batches etc. all work uniformly.
  const upstreamUrl = `${UPSTREAM}${inUrl.pathname}${inUrl.search}`;

  // Carry through anthropic-version + anthropic-beta if present.
  const headers = new Headers();
  headers.set("content-type", "application/json");
  headers.set("x-api-key", upstreamKey);
  const version = request.headers.get("anthropic-version");
  if (version) headers.set("anthropic-version", version);
  const beta = request.headers.get("anthropic-beta");
  if (beta) headers.set("anthropic-beta", beta);

  // Strip gateway-internal metadata and sampling fields. Body-only requests; ignore the
  // scrub step for GET/HEAD which carry no body.
  let outgoingBody: string | undefined = rawBody;
  if (request.method !== "GET" && request.method !== "HEAD" && rawBody) {
    outgoingBody = scrubForAnthropic(rawBody);
  }

  const upstream = await fetch(upstreamUrl, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : outgoingBody,
  });

  // Stream the response straight back. fetch() in Workers preserves the
  // chunked / SSE shape — we don't have to do anything special.
  const respHeaders = new Headers(upstream.headers);
  // Strip CORS-y bits we don't want to leak.
  respHeaders.delete("alt-svc");
  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: respHeaders,
  });
}

export function scrubForAnthropic(rawBody: string): string {
  try {
    const parsed = JSON.parse(rawBody);
    if (!parsed || typeof parsed !== "object") return rawBody;
    let touched = false;
    // This hop forwards the raw body, so the cap index.ts applied to its
    // parsed copy has to be applied here too.
    if (
      typeof parsed.max_tokens === "number" &&
      parsed.max_tokens > MAX_OUTPUT_TOKENS
    ) {
      parsed.max_tokens = MAX_OUTPUT_TOKENS;
      touched = true;
    }
    for (const k of SAMPLING_KEYS) {
      if (k in parsed) {
        delete parsed[k];
        touched = true;
      }
    }
    if (parsed.metadata && typeof parsed.metadata === "object") {
      let metaTouched = false;
      for (const k of RESERVED_METADATA_KEYS) {
        if (k in parsed.metadata) {
          delete parsed.metadata[k];
          metaTouched = true;
        }
      }
      // If metadata is now empty, drop it entirely — Anthropic accepts no
      // metadata field, but an empty object is fine too. Either is safe.
      if (metaTouched && Object.keys(parsed.metadata).length === 0) {
        delete parsed.metadata;
      }
      touched ||= metaTouched;
    }
    return touched ? JSON.stringify(parsed) : rawBody;
  } catch {
    // Not JSON or malformed — forward as-is and let upstream reject it.
    return rawBody;
  }
}
