/**
 * Anthropic forward path. Reverse-proxies to api.anthropic.com using the
 * gateway's own ANTHROPIC_API_KEY secret.
 *
 * We *almost* forward the body as-is, but we have to scrub gateway-internal
 * metadata hints (`metadata.fortune_route`) before sending. Anthropic's API
 * rejects unknown metadata fields with `invalid_request_error: metadata.X:
 * Extra inputs are not permitted` — verified 2026-05-22 with sonnet-4-6.
 */

const UPSTREAM = "https://api.anthropic.com";

// Metadata keys the gateway reserves for its own routing. Stripped from
// the request body before it goes upstream so Anthropic doesn't reject it.
const RESERVED_METADATA_KEYS = ["fortune_route"];

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

  // Strip gateway-internal metadata fields. Body-only requests; ignore the
  // scrub step for GET/HEAD which carry no body.
  let outgoingBody: string | undefined = rawBody;
  if (request.method !== "GET" && request.method !== "HEAD" && rawBody) {
    outgoingBody = scrubReservedMetadata(rawBody);
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

export function scrubReservedMetadata(rawBody: string): string {
  try {
    const parsed = JSON.parse(rawBody);
    if (parsed && typeof parsed === "object" && parsed.metadata && typeof parsed.metadata === "object") {
      let touched = false;
      for (const k of RESERVED_METADATA_KEYS) {
        if (k in parsed.metadata) {
          delete parsed.metadata[k];
          touched = true;
        }
      }
      // If metadata is now empty, drop it entirely — Anthropic accepts no
      // metadata field, but an empty object is fine too. Either is safe.
      if (touched && Object.keys(parsed.metadata).length === 0) {
        delete parsed.metadata;
      }
      if (touched) return JSON.stringify(parsed);
    }
    return rawBody;
  } catch {
    // Not JSON or malformed — forward as-is and let upstream reject it.
    return rawBody;
  }
}
