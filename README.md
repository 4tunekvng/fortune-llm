# fortune-llm

A Cloudflare Worker that exposes the **Anthropic Messages API** and
routes requests through a free-by-default fallback chain:

```
                ┌─────────────────────────────────────────────────────────┐
POST /v1/messages   decideRoute(body)                                      │
                ├─────────────────────────────────────────────────────────┤
                │ default                  →  [workers-ai, gemini]         │
                │ image content present    →  [gemini]                     │
                │ very long context        →  [gemini, workers-ai]         │
                │ metadata.fortune_route="anthropic"  →  [anthropic]       │
                │ metadata.fortune_route="workers-ai" →  [workers-ai]      │
                │ metadata.fortune_route="gemini"     →  [gemini]          │
                └─────────────────────────────────────────────────────────┘
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

**Policy:** zero paid Anthropic by default. Paid Anthropic is opt-in
only via the consumer request setting `metadata.fortune_route="anthropic"`.
When the free chain is exhausted, the gateway returns a 503 — *no
silent escalation to paid*.

## Backends

| Tier | Model | Cost | Capabilities |
|------|-------|------|--------------|
| `workers-ai` | `@cf/meta/llama-4-scout-17b-16e-instruct` (configurable) | Free up to 10k neurons/day | Text, tool use, long-ish context |
| `gemini` | `gemini-2.5-flash` (configurable) | Free up to API rate caps | Text, tool use, vision, long context |
| `anthropic` | Whatever the caller asked for | Paid | Frontier — only when explicitly opted into |

Diagnostic headers on every response:

```
x-fortune-llm-route:          workers-ai | gemini | anthropic
x-fortune-llm-model:           the actual model executed
x-fortune-llm-chain:           workers-ai,gemini
x-fortune-llm-reason:          why this chain was picked
x-fortune-llm-prior-errors:    tier:msg | tier:msg   (only when a tier failed before success)
```

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
npm run login                       # opens browser; one-time wrangler login
npm run secret:set:token            # any random string; consumers send this as their API key
npm run secret:set:gemini           # paste your free Gemini API key from AI Studio
npm run secret:set:anthropic        # OPTIONAL — only if you want the paid escape valve enabled
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
