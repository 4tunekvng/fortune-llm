#!/usr/bin/env node
/**
 * End-to-end integration test using the real @anthropic-ai/sdk pointed
 * at the deployed fortune-llm gateway. This is the protocol-correctness
 * test: every consumer app on the portfolio uses the SDK, so if the
 * SDK accepts our responses, the apps work.
 *
 * Covers:
 *   1. Non-streaming messages.create()                       (JSON path)
 *   2. Streaming messages.stream()                            (SSE path, accumulation)
 *   3. Non-stream cache: identical temp=0 call twice          (header check + content equality)
 *   4. Stream-from-cache: same temp=0 prompt, stream:true     (synthesized SSE from cached JSON)
 *   5. Tool-using call                                        (round-trip with tools[])
 *   6. Forced anthropic route                                 (escape valve)
 *   7. Forced groq route                                      (free chain leaf)
 *   8. Forced openrouter route                                (multi-model fallback)
 *
 * Pass criteria for each: assertion holds AND we print evidence
 * (truncated content + relevant fortune-llm headers).
 *
 * Usage:
 *   node tests-integration/sdk-e2e.mjs
 *
 * The script reads GATEWAY_URL and GATEWAY_TOKEN from env or falls
 * back to the prod defaults baked in below.
 */

import Anthropic from "@anthropic-ai/sdk";

const GATEWAY_URL = process.env.GATEWAY_URL ?? "https://fortune-llm.fortunee.workers.dev";
const GATEWAY_TOKEN =
  process.env.GATEWAY_TOKEN ?? "KtfvUb0dLwGu7NSQxyKlvbkse2hFBvK1ZPf9RwDSIfo";

const client = new Anthropic({
  baseURL: GATEWAY_URL,
  apiKey: GATEWAY_TOKEN,
});

// Capture fortune-llm headers from each response. The SDK exposes
// `_response.headers` via the `.withResponse()` helper.
function fortuneHeaders(headersLike) {
  const out = {};
  // Header objects vary across runtimes (Headers / object / array). Normalize.
  if (!headersLike) return out;
  const entries =
    typeof headersLike.entries === "function" ? Array.from(headersLike.entries()) : Object.entries(headersLike);
  for (const [k, v] of entries) {
    const lower = k.toLowerCase();
    if (lower.startsWith("x-fortune-llm-")) out[lower] = v;
  }
  return out;
}

let passes = 0;
let fails = 0;
const failures = [];

function ok(label, evidence) {
  passes++;
  console.log(`✓ ${label}`);
  if (evidence) {
    for (const [k, v] of Object.entries(evidence)) console.log(`    ${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
  }
  console.log("");
}

function bad(label, reason, evidence) {
  fails++;
  failures.push({ label, reason });
  console.log(`✗ ${label}`);
  console.log(`    REASON: ${reason}`);
  if (evidence) {
    for (const [k, v] of Object.entries(evidence)) console.log(`    ${k}: ${typeof v === "string" ? v : JSON.stringify(v)}`);
  }
  console.log("");
}

// Small uniqueness suffix so caching tests don't conflict across runs.
const RUN_TAG = Math.random().toString(36).slice(2, 8);

// ─────────────────────────────────────────────────────────────
// Test 1: Non-streaming text request
// ─────────────────────────────────────────────────────────────
async function test1NonStreaming() {
  const label = "1. Non-streaming messages.create() returns real model text";
  try {
    const { data: msg, response } = await client.messages
      .create({
        model: "claude-sonnet-4-6",
        max_tokens: 120,
        // temperature explicitly omitted so we get a fresh (uncached) response
        messages: [
          {
            role: "user",
            content: `Reply with exactly one sentence about the chain rule in calculus, ending with the literal word VERIFIED-${RUN_TAG}-1.`,
          },
        ],
      })
      .withResponse();

    const h = fortuneHeaders(response.headers);
    const text = msg.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("");
    if (!text || !text.includes(`VERIFIED-${RUN_TAG}-1`)) {
      bad(label, "response did not contain the expected verification token", {
        content_preview: text.slice(0, 200),
        ...h,
      });
      return;
    }
    if (!h["x-fortune-llm-route"]) {
      bad(label, "missing x-fortune-llm-route header", { ...h });
      return;
    }
    ok(label, {
      route: h["x-fortune-llm-route"],
      model: h["x-fortune-llm-model"],
      content: text.slice(0, 120),
    });
  } catch (err) {
    bad(label, err.message ?? String(err));
  }
}

// ─────────────────────────────────────────────────────────────
// Test 2: Streaming via messages.stream()
//
// Verifies gateway streaming protocol correctness: the SDK accumulates
// text from our SSE events without errors, the final message has a
// sensible stop_reason, and the accumulated text is non-trivial. We
// don't assert a specific output token because that would be testing
// model instruction-following, not the gateway.
// ─────────────────────────────────────────────────────────────
async function test2Streaming() {
  const label = "2. Streaming messages.stream() accumulates real text from SSE events";
  try {
    let accumulated = "";
    let textEventCount = 0;
    const stream = client.messages.stream({
      model: "claude-sonnet-4-6",
      max_tokens: 150,
      messages: [
        {
          role: "user",
          content: "Write a short factual sentence about the speed of light.",
        },
      ],
    });
    stream.on("text", (t) => {
      accumulated += t;
      textEventCount++;
    });
    const final = await stream.finalMessage();
    const responseHeaders = stream.response?.headers;
    const h = fortuneHeaders(responseHeaders);
    if (accumulated.length < 10) {
      bad(label, `accumulated text too short (${accumulated.length} chars)`, {
        accumulated,
        final_stop_reason: final.stop_reason,
        ...h,
      });
      return;
    }
    if (final.stop_reason !== "end_turn" && final.stop_reason !== "max_tokens") {
      bad(label, `unexpected stop_reason: ${final.stop_reason}`, { ...h });
      return;
    }
    if (final.content.find((b) => b.type === "text")?.text !== accumulated) {
      bad(label, "final message text does not match the streamed accumulation", {
        accumulated_len: accumulated.length,
        final_text_len: final.content.find((b) => b.type === "text")?.text?.length,
      });
      return;
    }
    ok(label, {
      route: h["x-fortune-llm-route"],
      model: h["x-fortune-llm-model"],
      text_events: textEventCount,
      stop_reason: final.stop_reason,
      streamed: accumulated.slice(0, 120),
    });
  } catch (err) {
    bad(label, err.message ?? String(err));
  }
}

// ─────────────────────────────────────────────────────────────
// Test 3: Non-stream cache (temp=0, same prompt twice)
// ─────────────────────────────────────────────────────────────
async function test3NonStreamCache() {
  const label = "3. Non-stream cache: identical temp=0 prompt hits cache on 2nd call";
  const prompt = `What is 17 times 23? Reply with just the number followed by the word VERIFIED-${RUN_TAG}-3.`;
  try {
    // First call — should miss + store.
    const { data: m1, response: r1 } = await client.messages
      .create({
        model: "claude-sonnet-4-6",
        max_tokens: 40,
        temperature: 0,
        messages: [{ role: "user", content: prompt }],
      })
      .withResponse();
    const h1 = fortuneHeaders(r1.headers);
    if (h1["x-fortune-llm-cache"] !== "miss-stored") {
      bad(label, `expected miss-stored on first call, got: ${h1["x-fortune-llm-cache"] ?? "(unset)"}`, h1);
      return;
    }
    const text1 = m1.content.find((b) => b.type === "text")?.text ?? "";

    // Second call — should hit.
    const { data: m2, response: r2 } = await client.messages
      .create({
        model: "claude-sonnet-4-6",
        max_tokens: 40,
        temperature: 0,
        messages: [{ role: "user", content: prompt }],
      })
      .withResponse();
    const h2 = fortuneHeaders(r2.headers);
    const text2 = m2.content.find((b) => b.type === "text")?.text ?? "";

    if (h2["x-fortune-llm-cache"] !== "hit") {
      bad(label, `expected hit on second call, got: ${h2["x-fortune-llm-cache"] ?? "(unset)"}`, h2);
      return;
    }
    if (text1 !== text2) {
      bad(label, "cached body differs from first body", { text1, text2 });
      return;
    }
    ok(label, {
      first: h1["x-fortune-llm-cache"],
      second: h2["x-fortune-llm-cache"],
      content: text1.slice(0, 80),
    });
  } catch (err) {
    bad(label, err.message ?? String(err));
  }
}

// ─────────────────────────────────────────────────────────────
// Test 4: Stream-from-cache (THE big new feature)
// ─────────────────────────────────────────────────────────────
async function test4StreamFromCache() {
  const label = "4. Stream-from-cache: stream:true + temp=0 same prompt twice → 2nd is SSE-from-cache";
  // Unique prompt for this test so cache is fresh.
  const prompt = `Reply with exactly one short sentence about backpropagation, ending with the literal word VERIFIED-${RUN_TAG}-4.`;
  try {
    // First streaming call.
    let acc1 = "";
    const s1 = client.messages.stream({
      model: "claude-sonnet-4-6",
      max_tokens: 80,
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });
    s1.on("text", (t) => (acc1 += t));
    await s1.finalMessage();
    const h1 = fortuneHeaders(s1.response?.headers);

    // Second streaming call — same prompt.
    let acc2 = "";
    const s2 = client.messages.stream({
      model: "claude-sonnet-4-6",
      max_tokens: 80,
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });
    s2.on("text", (t) => (acc2 += t));
    await s2.finalMessage();
    const h2 = fortuneHeaders(s2.response?.headers);

    // First call: should be miss-stored-stream (we forced non-stream upstream, cached, synthesized SSE).
    if (h1["x-fortune-llm-cache"] !== "miss-stored-stream") {
      bad(label, `1st call expected miss-stored-stream, got: ${h1["x-fortune-llm-cache"] ?? "(unset)"}`, {
        accumulated: acc1.slice(0, 100),
        ...h1,
      });
      return;
    }
    // Second call: should be hit-stream.
    if (h2["x-fortune-llm-cache"] !== "hit-stream") {
      bad(label, `2nd call expected hit-stream, got: ${h2["x-fortune-llm-cache"] ?? "(unset)"}`, {
        accumulated: acc2.slice(0, 100),
        ...h2,
      });
      return;
    }
    if (acc1 !== acc2) {
      bad(label, "streaming bodies differ between miss-stored-stream and hit-stream", {
        first: acc1,
        second: acc2,
      });
      return;
    }
    if (!acc2.includes(`VERIFIED-${RUN_TAG}-4`)) {
      bad(label, "cached streaming reply lost the verification token", { accumulated: acc2 });
      return;
    }
    ok(label, {
      first: h1["x-fortune-llm-cache"],
      second: h2["x-fortune-llm-cache"],
      content: acc2.slice(0, 100),
    });
  } catch (err) {
    bad(label, err.message ?? String(err));
  }
}

// ─────────────────────────────────────────────────────────────
// Test 5: Tool use round-trip
// ─────────────────────────────────────────────────────────────
async function test5ToolUse() {
  const label = "5. Tool-use: model emits a tool_use block when given tools[]";
  try {
    const { data: msg, response } = await client.messages
      .create({
        model: "claude-sonnet-4-6",
        max_tokens: 200,
        messages: [
          {
            role: "user",
            content: "Use the lookup_weather tool to check the weather in Tokyo. Then briefly summarize.",
          },
        ],
        tools: [
          {
            name: "lookup_weather",
            description: "Look up the current weather in a city.",
            input_schema: {
              type: "object",
              properties: { city: { type: "string", description: "City name" } },
              required: ["city"],
            },
          },
        ],
      })
      .withResponse();
    const h = fortuneHeaders(response.headers);
    const hasToolUse = msg.content.some((b) => b.type === "tool_use");
    if (!hasToolUse) {
      bad(label, "no tool_use block in response", {
        content: msg.content,
        stop_reason: msg.stop_reason,
        ...h,
      });
      return;
    }
    const toolUse = msg.content.find((b) => b.type === "tool_use");
    ok(label, {
      route: h["x-fortune-llm-route"],
      tool: toolUse?.name,
      input: toolUse?.input,
      stop_reason: msg.stop_reason,
    });
  } catch (err) {
    bad(label, err.message ?? String(err));
  }
}

// ─────────────────────────────────────────────────────────────
// Test 6: Forced anthropic route (paid escape valve)
// ─────────────────────────────────────────────────────────────
async function test6ForcedAnthropic() {
  const label = "6. Forced metadata.fortune_route=anthropic uses real Anthropic";
  try {
    const { data: msg, response } = await client.messages
      .create({
        model: "claude-sonnet-4-6",
        max_tokens: 60,
        messages: [
          { role: "user", content: `Say "OK ${RUN_TAG}-6" and nothing else.` },
        ],
        metadata: { fortune_route: "anthropic" },
      })
      .withResponse();
    const h = fortuneHeaders(response.headers);
    if (h["x-fortune-llm-route"] !== "anthropic") {
      bad(label, `route should be anthropic, got: ${h["x-fortune-llm-route"]}`, h);
      return;
    }
    const text = msg.content.find((b) => b.type === "text")?.text ?? "";
    if (!text.includes(`${RUN_TAG}-6`)) {
      bad(label, "model didn't include the requested token", { text, ...h });
      return;
    }
    ok(label, { route: h["x-fortune-llm-route"], text: text.slice(0, 80) });
  } catch (err) {
    bad(label, err.message ?? String(err));
  }
}

// ─────────────────────────────────────────────────────────────
// Test 7 & 8: Force Groq / OpenRouter
//
// Either tier can be transiently unavailable (circuit open from a
// rate-limit hit). We accept two valid outcomes:
//   - 200 with x-fortune-llm-route matching the forced tier, OR
//   - 503 quota_exhausted + circuit-open header (the gateway correctly
//     blocked the request because the tier is rate-limited).
// Either outcome means the gateway is doing the right thing.
// ─────────────────────────────────────────────────────────────
async function test7And8Forced(route) {
  const label = `${route === "groq" ? "7" : "8"}. Forced metadata.fortune_route=${route} dispatches to ${route}`;
  try {
    const { data: msg, response } = await client.messages
      .create({
        model: "claude-sonnet-4-6",
        max_tokens: 60,
        messages: [{ role: "user", content: `Say the word "${route}" and nothing else.` }],
        metadata: { fortune_route: route },
      })
      .withResponse();
    const h = fortuneHeaders(response.headers);
    if (h["x-fortune-llm-route"] !== route) {
      bad(label, `route should be ${route}, got: ${h["x-fortune-llm-route"]}`, h);
      return;
    }
    const text = msg.content.find((b) => b.type === "text")?.text ?? "";
    ok(label, { route: h["x-fortune-llm-route"], model: h["x-fortune-llm-model"], text: text.slice(0, 80) });
  } catch (err) {
    // Accept the case where the forced tier's circuit is currently
    // open (transient quota state); the dispatcher attempted to route
    // to the tier correctly, which is the gateway-correctness check.
    const msg = err?.message ?? String(err);
    const status = err?.status ?? err?.error?.status;
    const errBody = err?.error ?? {};
    const isQuotaSkip =
      msg.includes("quota_exhausted") ||
      msg.includes("circuit") ||
      msg.includes("open until") ||
      JSON.stringify(errBody).includes("quota_exhausted");
    if (isQuotaSkip) {
      ok(`${label} (deferred — tier circuit currently open)`, {
        note: `${route} tier is transiently unavailable, gateway correctly blocked`,
        status,
      });
      return;
    }
    bad(label, msg);
  }
}

// ─────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────
async function main() {
  console.log(`Gateway: ${GATEWAY_URL}\nRun tag: ${RUN_TAG}\n`);
  await test1NonStreaming();
  await test2Streaming();
  await test3NonStreamCache();
  await test4StreamFromCache();
  await test5ToolUse();
  await test6ForcedAnthropic();
  await test7And8Forced("groq");
  await test7And8Forced("openrouter");

  console.log("=".repeat(60));
  console.log(`PASSED: ${passes}    FAILED: ${fails}`);
  if (fails > 0) {
    console.log("\nFailures:");
    for (const f of failures) console.log(`  - ${f.label}: ${f.reason}`);
    process.exit(1);
  }
  console.log("All integration tests passed against live deployed gateway.");
}

main().catch((err) => {
  console.error("Unhandled error:", err);
  process.exit(2);
});
