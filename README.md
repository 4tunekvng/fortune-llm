# fortune-llm

A Cloudflare Worker that exposes the **Anthropic Messages API** and
routes through a free-first fallback chain — paid Anthropic is the
last-resort tier appended automatically when the worker has an
`ANTHROPIC_API_KEY` configured.

```
                ┌──────────────────────────────────────────────────────────────────────────┐
POST /v1/messages   decideRoute(body, { anthropicFallback })                                │
                ├──────────────────────────────────────────────────────────────────────────┤
                │ default / tools[]     →  [groq, workers-ai, gemini, openrouter,*anthropic]│
                │ image content         →  [gemini, openrouter, *anthropic]                 │
                │ very long context     →  [gemini, openrouter, workers-ai, *anthropic]     │
                │ output_config (JSON)  →  [anthropic]   (free tiers can't speak it)        │
                │                                                                            │
                │ metadata.fortune_route="anthropic"  →  [anthropic]   (force paid)         │
                │ metadata.fortune_route="free"       →  free chain only (no paid fallback) │
                │ metadata.fortune_route="groq"       →  [groq]                              │
                │ metadata.fortune_route="workers-ai" →  [workers-ai]                        │
                │ metadata.fortune_route="gemini"     →  [gemini]                            │
                │ metadata.fortune_route="openrouter" →  [openrouter]                        │
                └──────────────────────────────────────────────────────────────────────────┘
                  * anthropic appended ONLY when ANTHROPIC_API_KEY is configured.
                  The free chain stacks 4 independent quota pools, so with all
                  four configured + Gemini key rotation, anthropic is reached
                  only after every pool is exhausted — rare in practice.
                                   │
                                   ▼
                        Try each tier in order. First one
                        that succeeds returns. If every tier
                        in the chain fails → 503 (fail loud).
```

## Why

Every consumer app I run (`fortune_agents/*`, `moonshots/*`,
`Lena/lena_agent`, …) imports `@anthropic-ai/sdk` and reads
`ANTHROPIC_API_KEY`. This gateway makes that environment forward to free
backends without any consumer code change:

```
ANTHROPIC_BASE_URL=https://fortune-llm.<your-cf-account>.workers.dev
ANTHROPIC_API_KEY=<gateway-token>            # not an Anthropic key
```

**Policy:** free first, paid as last-resort.

- The default chain always tries free tiers (Workers AI, Gemini) first.
- When `ANTHROPIC_API_KEY` is configured on the worker, Anthropic is
  appended as the **last-resort tier** to every default chain. We only
  ever escalate to paid after every free tier has failed or been
  circuit-broken.
- If the worker has no `ANTHROPIC_API_KEY`, the gateway fails loudly
  with 503 when the free chain is exhausted — *no silent escalation*.
- Callers that want to opt out of paid escalation per-request can pass
  `metadata.fortune_route="free"`; that locks the chain to free-only
  even when Anthropic is available.
- Callers that want to force paid for a specific request can pass
  `metadata.fortune_route="anthropic"` (single-tier chain).

## Backends

| Tier | Default model | Cost | Capabilities | Notes |
|------|---------------|------|--------------|-------|
| `groq` | `llama-3.3-70b-versatile` | Free, generous RPM/RPD | Text, native tool use, fastest inference | Goes first in the default chain. Configurable via `DEFAULT_GROQ_MODEL`. |
| `workers-ai` | `@cf/google/gemma-4-26b-a4b-it` | Free up to 10k neurons/day (per Cloudflare account) | Text, tool use | Separate quota from every other tier. Configurable via `DEFAULT_OSS_MODEL`. |
| `gemini` | `gemini-2.5-flash` | Free up to RPM/RPD caps (per API key) | Text, tool use, vision, 1M+ context | Supports multi-key rotation: set `GEMINI_API_KEYS` (comma-separated) to multiply quota. Configurable via `DEFAULT_GEMINI_MODEL`. |
| `openrouter` | `meta-llama/llama-3.3-70b-instruct:free` | Free via `:free` model variants | Text, tool use, vision (some models), long context | Independent quota pool that doesn't share with Cloudflare/Google/Groq. Configurable via `DEFAULT_OPENROUTER_MODEL`. |
| `anthropic` | Whatever the caller asked for | Paid | Frontier | Last-resort: appended automatically when `ANTHROPIC_API_KEY` is configured; can also be forced via `metadata.fortune_route="anthropic"`. |

The whole point of stacking four free tiers is that each one has an
independent quota pool — exhausting all of them on the same day is rare,
which means `anthropic` (the only one that bills) is rarely reached even
under significant load. Add a few more Gemini keys (friend-donated) and
the chain comfortably handles thousands of requests/day on free.

Diagnostic headers on every response:

```
x-fortune-llm-route:                 groq | workers-ai | gemini | openrouter | anthropic
x-fortune-llm-model:                 the actual model executed
x-fortune-llm-chain:                 groq,workers-ai,gemini,openrouter
x-fortune-llm-reason:                why this chain was picked
x-fortune-llm-prior-errors:          tier:msg | tier:msg   (only when a tier failed before success)
x-fortune-llm-gemini-keys:           N                       (only when N > 1 keys are configured)
x-fortune-llm-openrouter-models:     N                       (size of the OpenRouter fallback list)
x-fortune-llm-skipped:               tier:<ISO>,…            (tiers whose circuit was open)
x-fortune-llm-cache:                 hit | miss-stored | miss-skip
x-fortune-llm-cache-age-s:           seconds since the cached entry was written (hits only)
x-fortune-llm-rate-limit:            count/limit (only on 429 responses)
```

## Response cache

Exact-match KV cache keyed on the request hash. A second identical
request inside the TTL returns the cached response without touching any
provider — biggest single quota multiplier on top of the multi-provider
chain.

| Setting | Default | Notes |
|---|---|---|
| TTL | 24h | Override via `CACHE_TTL_SECONDS` in `[vars]`. Floored at 60s (KV min), capped at 30 days. Set 0 to disable. |
| Auto-cache when… | `temperature === 0` and no tools and not streaming | Apps that ask for determinism get the determinism benefit automatically. |
| Per-request opt-in | `metadata.fortune_cache = true` | Caches even with temperature > 0. |
| Per-request opt-out | `metadata.fortune_no_cache = true` | Always wins. |
| Never cached | Streaming requests, tool-using requests, `fortune_require_tools` | Too in-context-sensitive to safely return verbatim. |

## Rate limit

Per-IP request counter, KV-backed, 60-second buckets. Stops a runaway
loop or leaked-token abuse from draining the shared free quota for
every other consumer.

| Setting | Default | Notes |
|---|---|---|
| Limit | 200 req/min per IP | Override via `RATE_LIMIT_PER_MIN` in `[vars]`. Set 0 to disable. |
| Identification | `cf-connecting-ip` → `x-forwarded-for` first hop → `"unknown"` | Cloudflare always sets `cf-connecting-ip` for non-CF clients. |
| Response on cap | `429 {"type":"rate_limit_error","message":…}` with `Retry-After` header | Standard back-off semantics. |
| Failure mode | Fails open on KV unavailability | Better to miss a limit than block all traffic on infra trouble. |

## Endpoints

| Method | Path | Behavior |
|--------|------|----------|
| GET | `/healthz` | Liveness; reports default OSS + Gemini models and which backends are configured. |
| POST | `/v1/messages` | Routes per the diagram above. |
| `*` | `/v1/*` | Other Anthropic endpoints (count_tokens, batches) are forwarded **only** if `ANTHROPIC_API_KEY` is configured. Otherwise 501. |
| OPTIONS | any | CORS preflight. |

## Deploy

Cloudflare account with Workers (free) and Workers AI (free tier covers
~10k neurons/day). A Gemini API key is free from Google AI Studio.

```
npm install
npm test
npm run login                              # opens browser; one-time wrangler login
npm run secret:set:token                   # any random string; consumers send this as their API key

# Free providers — set as many as you can; each is an independent quota pool.
npm run secret:set:groq                    # https://console.groq.com/keys
npm run secret:set:openrouter              # https://openrouter.ai/keys
npm run secret:set:gemini                  # one Gemini key (https://aistudio.google.com/apikey), OR:
npm run secret:set:gemini:multi            # comma-separated multiple Gemini keys → N× quota

npm run secret:set:anthropic               # OPTIONAL — paid last-resort. Omit for strict free-only.

npx wrangler kv namespace create CIRCUIT             # one-time: circuit-breaker state
npx wrangler kv namespace create CIRCUIT --preview   # one-time: preview env
# copy the printed ids into wrangler.toml (replace REPLACE_WITH_KV_NAMESPACE_ID etc.)
npm run deploy
```

Wrangler prints `https://fortune-llm.<account>.workers.dev`. Paste that
URL into each consumer app's `ANTHROPIC_BASE_URL`, and the gateway token
into `ANTHROPIC_API_KEY`.

`npm run tail` streams live logs.

To swap the default OSS or Gemini model, edit `wrangler.toml`
(`DEFAULT_OSS_MODEL`, `DEFAULT_GEMINI_MODEL`) and redeploy.

## Adding new backends

The gateway is built around a single `BackendKind` union and a
`decideRoute` function returning an ordered chain. To add a backend
(e.g. Groq, a local Ollama via Cloudflare Tunnel, etc.):

1. Add the kind to `src/route.ts:BackendKind` and `decideRoute`.
2. Create `src/<backend>.ts` exporting a `call<Backend>` that accepts
   an `AnthropicMessagesRequest` and returns an Anthropic-shaped
   `Response` (or SSE stream).
3. Add the new tier to the `invokeTier` switch in `src/index.ts`.

## Circuit breaker

When a free-tier backend (`workers-ai` or `gemini`) blows its quota or
hits a rate limit, the gateway used to silently retry on every request —
wasting Worker CPU and propagating latency to every consumer. Now each
quota / rate-limit error trips a **per-backend circuit** stored in a KV
namespace (`CIRCUIT`). While the circuit is open the dispatcher skips
that tier entirely.

| Behavior | Detail |
|---|---|
| Trip signals | `RESOURCE_EXHAUSTED` (Gemini), `429`, `rate limit`, `quota`, `neurons exhausted`, `daily limit exceeded` |
| Not tripped | Timeouts, transient 5xx, network errors, `max_tokens exceeded` (those still retry) |
| Default open duration | 1 hour. Override via `[vars] CIRCUIT_TRIP_DURATION_MS = "1800000"` (30 min). Capped at 24h. |
| Anthropic tier | Never circuit-gated — it's explicitly opt-in and paid, the user asked for it. |
| KV unbound | Breaker no-ops with a one-time warning; behavior reverts to the old retry-every-time chain. |
| Response when all tiers skipped | `503` with `{"type":"error","error":{"type":"quota_exhausted","message":"all backends unavailable: workers-ai (open until <ISO>), gemini (open until <ISO>)"}}` |
| Diagnostic header | `x-fortune-llm-skipped: workers-ai:<ISO>,gemini:<ISO>` on every response where a tier was skipped |

One-time setup (also shown in [Deploy](#deploy) above):

```
npx wrangler kv namespace create CIRCUIT
npx wrangler kv namespace create CIRCUIT --preview
# paste the printed ids into wrangler.toml
```

## Limitations

- Workers AI does not support vision; vision requests go to Gemini.
- Workers AI tool-call quality on Llama 4 Scout is good but not
  Sonnet-equivalent for very long agentic chains.
- `count_tokens` and `batches` endpoints are Anthropic-specific —
  they only work if Anthropic is configured. Most consumer apps don't
  use them.
- Streaming for Workers AI tool-using calls is *synthesized* from a
  buffered non-streamed call. The wire shape is correct Anthropic SSE
  but the deltas arrive in one burst rather than truly token-by-token.

## License

MIT.
