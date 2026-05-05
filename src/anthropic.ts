/**
 * Anthropic forward path. Reverse-proxies to api.anthropic.com using the
 * gateway's own ANTHROPIC_API_KEY secret. The body is forwarded as-is —
 * we don't try to mutate the user's request.
 */

const UPSTREAM = "https://api.anthropic.com";

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

  const upstream = await fetch(upstreamUrl, {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : rawBody,
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
