/**
 * Auth on the gateway is intentionally simple: a single shared bearer
 * token (the GATEWAY_TOKEN secret). Consumer apps put it in
 * `ANTHROPIC_API_KEY` so the @anthropic-ai/sdk forwards it as the
 * `x-api-key` header — exactly the same shape Anthropic itself accepts.
 *
 * The token is *not* meant as fine-grained auth. It exists to keep
 * randos off the free Workers AI tier; rotate it any time abuse shows up.
 */

export interface AuthOk {
  ok: true;
}

export interface AuthFail {
  ok: false;
  status: number;
  message: string;
}

export function authenticate(request: Request, expected: string | undefined): AuthOk | AuthFail {
  if (!expected) {
    // Misconfigured deploy — refuse rather than silently letting everyone in.
    return {
      ok: false,
      status: 500,
      message: "Gateway is not configured: GATEWAY_TOKEN secret is missing.",
    };
  }

  const xApiKey = request.headers.get("x-api-key");
  const auth = request.headers.get("authorization");
  const bearer = auth?.toLowerCase().startsWith("bearer ")
    ? auth.slice(7).trim()
    : null;

  const presented = xApiKey ?? bearer;
  if (!presented) {
    return {
      ok: false,
      status: 401,
      message: "Missing x-api-key (or Authorization: Bearer) header.",
    };
  }

  // Constant-time compare to avoid timing oracles on the secret.
  if (!constantTimeEqual(presented, expected)) {
    return {
      ok: false,
      status: 403,
      message: "Invalid gateway token.",
    };
  }

  return { ok: true };
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) {
    mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return mismatch === 0;
}
