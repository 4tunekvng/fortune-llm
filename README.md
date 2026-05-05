# fortune-llm

A tiny Cloudflare Worker that exposes the **Anthropic Messages API** but
routes most requests to **Cloudflare Workers AI** (Llama 3.3 70B by default,
free up to 10k neurons/day) and only escalates to real Anthropic when a
request needs capabilities OSS models can't reliably deliver.

## Why

Every consumer app I run (`fortune_agents/*`, `moonshots/*`, `Lena/lena_agent`,
…) imports `@anthropic-ai/sdk` and reads `ANTHROPIC_API_KEY`. Asking each
end-user to bring their own key is a non-starter for public demos. With
this gateway, the app's `.env` becomes:

```
ANTHROPIC_BASE_URL=https://fortune-llm.<your-cf-account>.workers.dev
ANTHROPIC_API_KEY=<gateway-token>            # not an Anthropic key
```

…and the SDK transparently calls the gateway. No code change in the
consumer.

## Routing

```
                         ┌────────────────────────────────────┐
   POST /v1/messages ───>│ decideRoute(body)                  │
                         └────────────────────────────────────┘
                                    │
            no tools, no images, ≤ ~16k tokens
                                    │
                                    ▼
                         Cloudflare Workers AI  →  Llama 3.3 70B  (FREE)
                         (response translated back to Anthropic SSE)

                         tools[] | image | long ctx | metadata override
                                    │
                                    ▼
                         api.anthropic.com         (PAID, your key)
```

The router is in [`src/route.ts`](src/route.ts). It's pure, deterministic,
unit-tested. It also honors `metadata.fortune_route = "anthropic"` /
`"workers-ai"` for explicit per-request overrides.

If Workers AI is rate-limited or down, the gateway transparently retries
on Anthropic (you still pay, but apps stay up).

## Endpoints

| Method | Path                  | Behavior                                              |
|--------|-----------------------|-------------------------------------------------------|
| GET    | `/healthz`            | liveness probe; reports default model + fallback flag |
| POST   | `/v1/messages`        | routes per the diagram above                          |
| `*`    | `/v1/*`               | every other Anthropic endpoint forwarded as-is        |
| OPTIONS| any                   | CORS preflight                                        |

Responses include three diagnostic headers for dev:

```
x-fortune-llm-route:    workers-ai | anthropic
x-fortune-llm-model:    @cf/meta/llama-3.3-70b-instruct-fp8-fast
x-fortune-llm-reason:   why this request landed where it did
```

## Deploy

You need a Cloudflare account with Workers (free) and Workers AI (free
tier covers ~10k neurons/day). Then:

```bash
npm install
npm run typecheck
npm test                                            # 24 unit tests, ~250ms

# 1. Set the gateway token consumer apps will send. Generate any random string.
wrangler secret put GATEWAY_TOKEN

# 2. (optional but recommended) Set your real Anthropic key — used as the
#    paid fallback for tool-use, vision, long context, and Workers AI outages.
wrangler secret put ANTHROPIC_API_KEY

# 3. Deploy.
wrangler deploy
```

Wrangler prints a `https://fortune-llm.<account>.workers.dev` URL. That's
the value to put in each consumer app's `ANTHROPIC_BASE_URL`.

To swap the OSS model later, edit `wrangler.toml` (`DEFAULT_OSS_MODEL`)
or set it via `wrangler secret put`.

## Tested with

- [`apertus`](../fortune_agents/apertus) — structured-output reflections
- [`weekend-cofounder`](../fortune_agents/weekend-cofounder) — Sonnet briefings + adaptive thinking
- [`knowledge-compounder`](../fortune_agents/knowledge-compounder) — capture pipeline + essay drafting
- [`network-agent`](../fortune_agents/network-agent) — drafted nudges
- [`maestro`](../fortune_agents/maestro) — post-set coaching summary
- [`moonshots/morphos`](../moonshots/morphos) — natural-language → swarm formation
- [`Lena/lena_agent`](../Lena/lena_agent) — agent runner over the Lena REST API

For the agent-style apps (especially `lena_agent` with 37 tools), every
request will land on the Anthropic path because the router sees `tools[]`.
That's expected — those flows genuinely need Sonnet's tool-call quality.
The free tier still covers all the simple chat in the rest of the
portfolio.

## Limitations

- **Vision** is not translated to Workers AI; it goes to Anthropic.
- **Tool use** is not translated to Workers AI; it goes to Anthropic.
- **Anthropic batches / count_tokens** endpoints are forwarded as-is.
- **Token counting** on the Workers AI path is estimated (chars / 4).
- **Streaming** is supported — Workers AI's NDJSON-ish stream is
  re-encoded as Anthropic's `message_start` / `content_block_delta` /
  `message_stop` SSE events so `@anthropic-ai/sdk`'s parser is happy.

## License

MIT.
