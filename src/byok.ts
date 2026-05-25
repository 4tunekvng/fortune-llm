/**
 * BYOK (Bring Your Own Key) header support.
 *
 * Consumers can pass `x-fortune-byok-<provider>: <key>` headers to use
 * their OWN provider key for this specific request instead of the
 * gateway's shared key. The most useful case: friends with their own
 * unused Groq / Gemini / OpenRouter accounts can donate quota to a
 * specific app without having their key shared with everyone else's
 * traffic.
 *
 * Recognized providers (one header each):
 *   x-fortune-byok-anthropic
 *   x-fortune-byok-groq
 *   x-fortune-byok-cerebras
 *   x-fortune-byok-gemini
 *   x-fortune-byok-openrouter
 *   x-fortune-byok-github-models
 *   x-fortune-byok-mistral
 *
 * Security:
 *   - BYOK keys are never logged or echoed in headers/body.
 *   - Validation: only the header-provider's own key is used. We don't
 *     trust headers for choosing the *upstream* provider; routing is
 *     unchanged. We only swap the key.
 *   - Each header value is trimmed and length-bounded; obviously-malformed
 *     keys are ignored (and the shared key is used as a fallback).
 *
 * Resolution:
 *   - `resolveKey(provider, env, headers)` returns the per-request key
 *     if present and valid, else falls back to the gateway's shared
 *     key from env, else null. Tier handlers throw on null.
 */

/** Recognized BYOK providers. Keep in sync with header names below. */
export type ByokProvider =
  | "anthropic"
  | "groq"
  | "cerebras"
  | "gemini"
  | "openrouter"
  | "github-models"
  | "mistral";

/** Reasonable bounds on key length (looser than any real provider). */
const MIN_KEY_LEN = 8;
const MAX_KEY_LEN = 512;

/**
 * Read the BYOK header for a given provider. Returns null when absent,
 * malformed, or out of length bounds. Never throws.
 */
export function extractByokKey(headers: Headers, provider: ByokProvider): string | null {
  const raw = headers.get(`x-fortune-byok-${provider}`);
  if (!raw) return null;
  const trimmed = raw.trim();
  if (trimmed.length < MIN_KEY_LEN || trimmed.length > MAX_KEY_LEN) return null;
  // Reject control characters / newlines that could break header
  // forwarding or get logged-then-leaked.
  if (/[\x00-\x1f\x7f]/.test(trimmed)) return null;
  return trimmed;
}

/**
 * Resolve the effective key for a provider on this request: BYOK
 * header wins, falls back to the gateway's shared key. Returns null
 * if neither source has a key — caller should throw a "not configured"
 * error so the dispatcher can advance to the next tier.
 */
export function resolveProviderKey(
  provider: ByokProvider,
  sharedKey: string | undefined,
  headers: Headers,
): { key: string; source: "byok" | "shared" } | null {
  const byok = extractByokKey(headers, provider);
  if (byok) return { key: byok, source: "byok" };
  if (sharedKey) return { key: sharedKey, source: "shared" };
  return null;
}
