#!/usr/bin/env node
/**
 * Latency and success replay against the live gateway, one request shape per
 * app: a Leggo chat turn, a Maestro curriculum, a Thesis (Knox) letter read
 * with a structured schema, and a Tellar room question. The prompts are
 * synthetic text of each app's real size; the tool lists, schema shapes,
 * max_tokens and metadata copy what each app sends.
 *
 * Usage:
 *   GATEWAY_TOKEN=... node tests-integration/replay-latency.mjs \
 *     [--mode chain|pinned] [--runs 4] [--concurrency 8] [--timeout-ms 60000]
 *     [--apps leggo,maestro,thesis,tellar] [--tiers gemini,workers-ai,...]
 *     [--out results.jsonl]
 *
 * chain:  every app the way it calls the gateway today, held to free tiers.
 *         Tellar pins one tier per call and walks its own list, so it is
 *         replayed that way; the others send fortune_route="free".
 * pinned: each app pinned to each tier in --tiers, to size per-tier limits.
 *
 * A run succeeds when the reply is HTTP 200 and its content is usable: the
 * structured reply parses and matches the schema, Leggo's turn calls a tool,
 * and Tellar's reply says something. Every response skips the cache.
 */

import { appendFileSync } from "node:fs";
import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";
import { z } from "zod";

const GATEWAY_URL =
  process.env.GATEWAY_URL ?? "https://fortune-llm.fortunee.workers.dev";
const GATEWAY_TOKEN = process.env.GATEWAY_TOKEN;
if (!GATEWAY_TOKEN) {
  console.error("Set GATEWAY_TOKEN to a gateway token first.");
  process.exit(2);
}

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
}
const MODE = arg("mode", "chain");
const RUNS = Number(arg("runs", "4"));
const CONCURRENCY = Number(arg("concurrency", "8"));
const TIMEOUT_MS = Number(arg("timeout-ms", "60000"));
const APPS = arg("apps", "leggo,maestro,thesis,tellar").split(",");
const TIERS = arg("tiers", "groq,mistral,workers-ai,gemini,openrouter").split(",");
const OUT = arg("out", null);

// ── Synthetic text of a given size ───────────────────────────────────────────

const WORDS = (
  "plan group friend evening weekend message draft reply calendar dinner " +
  "walk park coffee check note remind share list idea time place people " +
  "project goal practice drill speak pace pause story letter claim denial " +
  "coverage appeal review policy member provider service date amount reason"
).split(" ");

function filler(chars, seed) {
  let s = seed;
  const out = [];
  let len = 0;
  while (len < chars) {
    const n = 8 + (s % 9);
    const sentence = [];
    for (let i = 0; i < n; i++) {
      s = (s * 1103515245 + 12345) % 2147483648;
      sentence.push(WORDS[s % WORDS.length]);
    }
    const line = sentence.join(" ") + ". ";
    out.push(line);
    len += line.length;
  }
  return out.join("").slice(0, chars);
}

function system(intro, chars, seed) {
  return [
    {
      type: "text",
      text: `${intro}\n\n${filler(chars - intro.length, seed)}`,
      cache_control: { type: "ephemeral" },
    },
  ];
}

const tool = (name, description, properties, required = []) => ({
  name,
  description,
  input_schema: { type: "object", properties, required },
});
const str = (description) => ({ type: "string", description });

// ── Leggo: a chat turn, 10 tools, the reply must call one ────────────────────

const LEGGO_TOOLS = [
  tool("read_user_context", "Read the user's profile and preferences.", {}),
  tool("read_orbit", "List people in the user's orbit.", {
    filter: { type: "string", enum: ["all", "drifting", "close"] },
  }),
  tool("read_groups", "List the user's group chats.", {}),
  tool("read_group", "Read one group.", { group_id: str("Group id") }, [
    "group_id",
  ]),
  tool(
    "read_thread",
    "Read recent messages in a thread.",
    {
      thread_id: str("Thread id"),
      scope: { type: "string", enum: ["recent", "all"] },
    },
    ["thread_id"],
  ),
  tool("read_calendar", "Read the user's calendar.", {
    window: { type: "string", enum: ["today", "week", "month"] },
  }),
  tool("read_events_near", "Find events near the user.", {
    query: str("What kind of event"),
  }),
  tool("read_person", "Read one person.", { person_id: str("Person id") }, [
    "person_id",
  ]),
  tool(
    "draft_plan_for_group",
    "Draft a plan for a group.",
    {
      group_id: str("Group id"),
      title: str("Plan title"),
      when: str("When, in plain words"),
      message: str("The message to send the group"),
    },
    ["group_id", "title", "message"],
  ),
  tool(
    "draft_message_for_person",
    "Draft a message to one person.",
    { person_id: str("Person id"), message: str("The message") },
    ["person_id", "message"],
  ),
];

function leggoRequest() {
  return {
    body: {
      model: "claude-sonnet-4-6",
      max_tokens: 2048,
      system: system(
        "You are Leggo, an AI agent that lives inside the user's group chats " +
          "and threads as a group member. Keep social momentum alive: notice " +
          "drift, draft messages, surface plans. Use the read tools first.",
        5100,
        1,
      ),
      tools: LEGGO_TOOLS,
      messages: [
        {
          role: "user",
          content:
            "Me: Who haven't I seen in a while, and what could we do this " +
            "weekend? Something cheap, outdoors if the weather holds.",
        },
      ],
      metadata: { user_id: "leggo-app", fortune_require_tools: true },
    },
    check(msg) {
      const names = new Set(LEGGO_TOOLS.map((t) => t.name));
      const call = msg.content.find((b) => b.type === "tool_use");
      if (!call) return "no tool call";
      if (!names.has(call.name)) return `unknown tool ${call.name}`;
      return null;
    },
  };
}

// ── Maestro: a 14-day curriculum through messages.parse ─────────────────────

const targets = z.object({
  fillersPerMinMax: z.number(),
  wpmMin: z.number(),
  wpmMax: z.number(),
  eyeContactMin: z.number(),
  gestureVarietyMin: z.number(),
});
const SkillSchema = z.object({
  id: z.string().max(1000),
  name: z.string().max(1000),
  tagline: z.string().max(1000),
  description: z.string().max(1000),
  kind: z.enum(["speaking", "practice"]),
  drills: z.array(
    z.object({
      id: z.string().max(1000),
      title: z.string().max(1000),
      description: z.string().max(4000),
      day: z.number().int(),
      durationSec: z.number().int(),
      prompt: z.string().max(4000),
      targets,
    }),
  ),
});

function maestroRequest() {
  const fmt = zodOutputFormat(SkillSchema);
  return {
    body: {
      model: "claude-sonnet-5-5",
      max_tokens: 8192,
      system: system(
        "You are Maestro, an agentic skill tutor. Given a person's goal in " +
          "their own words, you produce a 14-day curriculum they can practice " +
          "indoors, 6 to 12 minutes a day.",
        2700,
        2,
      ),
      messages: [
        {
          role: "user",
          content:
            "Goal: I want to stop saying 'um' so much when I present at " +
            "work and sound calmer in meetings. Write 14 drills, one a day.",
        },
      ],
      output_config: { format: { type: "json_schema", schema: fmt.schema } },
    },
    schema: SkillSchema,
  };
}

// ── Thesis (Knox): read one denial letter into a 35-field record ────────────

const ExtractionSchema = z.object({
  docType: z.enum(["eob", "denial", "policy", "premium", "claim", "other"]),
  line: z.enum(["health", "auto", "home", "dental", "vision", "life"]),
  planType: z.enum([
    "erisa_self_funded",
    "aca_marketplace",
    "medicare_advantage",
    "medicaid",
    "commercial",
    "other",
  ]),
  reasonCategory: z.enum([
    "medical_necessity",
    "experimental",
    "out_of_network",
    "prior_authorization",
    "coding_error",
    "not_covered",
    "timely_filing",
    "other",
  ]),
  isActionable: z.boolean(),
  clinicalEvidence: z.boolean(),
  proceduralViolation: z.boolean(),
  premiumCents: z.number().int(),
  appealWindowDays: z.number().int(),
  billedCents: z.number().int(),
  allowedCents: z.number().int(),
  planPaidCents: z.number().int(),
  patientRespCents: z.number().int(),
  helpfulFacts: z.array(z.string()),
  cptCodes: z.array(z.string()),
  title: z.string().min(3).max(120),
  summary: z.string().min(8).max(480),
  insurer: z.string(),
  memberName: z.string(),
  memberId: z.string(),
  groupNumber: z.string(),
  claimNumber: z.string(),
  provider: z.string(),
  serviceDate: z.string(),
  letterDate: z.string(),
  deadlineDate: z.string(),
  denialReason: z.string(),
  policyClause: z.string(),
  diagnosis: z.string(),
  treatment: z.string(),
  appealAddress: z.string(),
  contactPhone: z.string(),
  externalReview: z.string(),
  nextStep: z.string(),
  notes: z.string(),
});

const LETTER = [
  "Northwind Health Plan, Appeals Department, PO Box 1200, Springfield.",
  "Date: October 30, 2026. Member: Jordan Avery. Member ID: NWH-4471-220.",
  "Group number: 88120. Claim number: CLM-2026-118842.",
  "Provider: Riverside Orthopedic Associates. Date of service: October 2, 2026.",
  "Re: Notice of adverse benefit determination.",
  "We reviewed the claim for an MRI of the right knee (CPT 73721) billed at",
  "$2,450.00. The allowed amount is $0.00 and the plan paid $0.00. You may",
  "owe the provider $2,450.00. The request was denied because the service was",
  "not medically necessary under section 4.2 of your certificate of coverage:",
  "conservative treatment (six weeks of physical therapy) was not documented",
  "before imaging. Diagnosis on the claim: M23.91, internal derangement of the",
  "right knee. You have 180 days from the date of this letter to appeal. Send",
  "your appeal with any medical records to the address above, or call",
  "1-800-555-0142. If the denial is upheld you may ask for an external review",
  "by an independent review organization through your state insurance",
  "department. This plan is an employer-sponsored self-funded plan governed by",
  "ERISA. ",
].join(" ");

function thesisRequest() {
  const fmt = zodOutputFormat(ExtractionSchema);
  const letter = (LETTER + filler(6000 - LETTER.length, 3)).slice(0, 6000);
  return {
    body: {
      model: "claude-sonnet-4-6",
      max_tokens: 3000,
      system: system(
        "You are Knox's intake analyst. You read one insurance document (an " +
          "email, an Explanation of Benefits, or a denial letter) and extract " +
          "a single structured record. Use empty strings, 0 and 'other' for " +
          "unknowns.",
        2300,
        4,
      ),
      messages: [
        {
          role: "user",
          content:
            "Source: uploaded letter. Received: 2026-11-01.\n\n" +
            `<document>\n${letter}\n</document>`,
        },
      ],
      output_config: { format: { type: "json_schema", schema: fmt.schema } },
    },
    schema: ExtractionSchema,
    headers: { "x-fortune-consumer": "knox" },
  };
}

// ── Tellar: a visitor's room question, 9 tools, its own tier list ───────────

const TELLAR_TOOLS = [
  tool("show_work", "Show a project card.", { id: str("Project id") }, ["id"]),
  tool("show_writing", "Show a piece of writing.", { id: str("Piece id") }, [
    "id",
  ]),
  tool("show_links", "Show links.", { topic: str("Topic") }),
  tool("show_card", "Show a card.", {
    card: { type: "string", enum: ["contact", "resume", "now"] },
  }),
  tool("open_story", "Open a story.", { id: str("Story id") }, ["id"]),
  tool("open_room", "Open a room.", {
    room: { type: "string", enum: ["building", "sports", "reading", "music"] },
  }),
  tool("offer_note", "Offer to take a note for Fortune.", {}),
  tool("draft_poll", "Draft a poll.", { question: str("Question") }, [
    "question",
  ]),
  tool("suggest", "Suggest follow-up questions.", {
    items: { type: "array", items: { type: "string" } },
  }),
];
const TELLAR_ROUTES = ["gemini", "mistral", "groq", "openrouter", "workers-ai"];

function tellarRequest() {
  return {
    body: {
      model: "claude-sonnet-5",
      max_tokens: 4096,
      system: [
        ...system(
          "You are the AI version of the site's owner, talking with a visitor " +
            "on their own website, in the first person. Keep answers short and " +
            "warm; use a tool when a card would show it better.",
          24000,
          5,
        ),
        { type: "text", text: "Room: building. Today: 2026-11-21." },
      ],
      tools: TELLAR_TOOLS,
      messages: [
        {
          role: "user",
          content: "What are you building right now, and why that?",
        },
      ],
    },
    check(msg) {
      const said = msg.content.some(
        (b) => b.type === "tool_use" || (b.type === "text" && b.text.trim()),
      );
      return said ? null : "empty reply";
    },
    clientRoutes: TELLAR_ROUTES,
  };
}

const BUILDERS = {
  leggo: leggoRequest,
  maestro: maestroRequest,
  thesis: thesisRequest,
  tellar: tellarRequest,
};

// ── One call ─────────────────────────────────────────────────────────────────

function judge(spec, msg) {
  if (!Array.isArray(msg?.content)) return "no content";
  if (spec.schema) {
    const text = msg.content.find((b) => b.type === "text")?.text ?? "";
    let json;
    try {
      json = JSON.parse(text);
    } catch {
      return `not JSON (stop ${msg.stop_reason})`;
    }
    const res = spec.schema.safeParse(json);
    return res.success ? null : `schema: ${res.error.issues[0]?.message}`;
  }
  return spec.check(msg);
}

async function callOnce(spec, route, deadline) {
  const body = {
    ...spec.body,
    metadata: {
      ...spec.body.metadata,
      fortune_route: route,
      fortune_no_cache: true,
    },
  };
  const started = Date.now();
  const left = Math.max(1, deadline - started);
  try {
    const resp = await fetch(`${GATEWAY_URL}/v1/messages`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": GATEWAY_TOKEN,
        "anthropic-version": "2023-06-01",
        ...spec.headers,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(left),
    });
    const text = await resp.text();
    const ms = Date.now() - started;
    const h = (k) => resp.headers.get(`x-fortune-llm-${k}`);
    const base = {
      status: resp.status,
      ms,
      served: h("route"),
      model: h("model"),
      prior: h("prior-errors"),
      skipped: h("skipped"),
    };
    if (resp.status !== 200) {
      let type = null;
      try {
        type = JSON.parse(text)?.error?.type ?? null;
      } catch {}
      return { ...base, ok: false, why: `HTTP ${resp.status} ${type ?? ""}` };
    }
    const why = judge(spec, JSON.parse(text));
    return { ...base, ok: why === null, why };
  } catch (err) {
    const ms = Date.now() - started;
    const timedOut = err?.name === "TimeoutError" || err?.name === "AbortError";
    return {
      status: 0,
      ms,
      served: null,
      ok: false,
      why: timedOut ? `client timeout ${TIMEOUT_MS} ms` : String(err),
    };
  }
}

// Tellar walks its own tier list, one pinned call each, until one answers.
async function callClientChain(spec, deadline) {
  const started = Date.now();
  const tries = [];
  for (const route of spec.clientRoutes) {
    if (Date.now() >= deadline) break;
    const r = await callOnce(spec, route, deadline);
    tries.push(`${route}:${r.ok ? "ok" : r.why}`);
    if (r.ok) return { ...r, ms: Date.now() - started, tries };
  }
  return {
    status: 0,
    ms: Date.now() - started,
    served: null,
    ok: false,
    why: "every tier failed",
    tries,
  };
}

// ── Plan, run, summarise ─────────────────────────────────────────────────────

function plan() {
  const jobs = [];
  for (let run = 0; run < RUNS; run++) {
    for (const app of APPS) {
      if (MODE === "chain") jobs.push({ app, route: "chain" });
      else for (const tier of TIERS) jobs.push({ app, route: tier });
    }
  }
  return jobs;
}

async function runJob(job) {
  const spec = BUILDERS[job.app]();
  const deadline = Date.now() + TIMEOUT_MS;
  const r =
    job.route === "chain"
      ? spec.clientRoutes
        ? await callClientChain(spec, deadline)
        : await callOnce(spec, "free", deadline)
      : await callOnce(spec, job.route, deadline);
  const row = { at: new Date().toISOString(), ...job, ...r };
  if (OUT) appendFileSync(OUT, JSON.stringify(row) + "\n");
  const tag = r.ok ? "ok  " : "FAIL";
  console.log(
    `${tag} ${job.app.padEnd(8)} ${job.route.padEnd(10)} ` +
      `${String(r.ms).padStart(6)} ms  served=${r.served ?? "-"}` +
      (r.ok ? "" : `  ${r.why}`) +
      (r.prior ? `  prior=[${r.prior}]` : "") +
      (r.tries ? `  tries=[${r.tries.join(" ")}]` : ""),
  );
  return row;
}

function pct(sorted, p) {
  if (sorted.length === 0) return null;
  const i = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, i)];
}

function summarise(rows, key) {
  const groups = new Map();
  for (const r of rows) {
    const k = key(r);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  const lines = [];
  for (const [k, rs] of [...groups.entries()].sort()) {
    // Latency counts every run, failures included at the time they took,
    // so a timeout cannot hide from p95.
    const ms = rs.map((r) => r.ms).sort((a, b) => a - b);
    const ok = rs.filter((r) => r.ok).length;
    lines.push(
      `| ${k} | ${rs.length} | ${(pct(ms, 50) / 1000).toFixed(1)} s | ` +
        `${(pct(ms, 95) / 1000).toFixed(1)} s | ${ok}/${rs.length} ` +
        `(${Math.round((100 * ok) / rs.length)}%) |`,
    );
  }
  return lines;
}

async function main() {
  const jobs = plan();
  console.log(
    `mode=${MODE} runs=${RUNS} jobs=${jobs.length} concurrency=${CONCURRENCY} ` +
      `timeout=${TIMEOUT_MS}ms gateway=${GATEWAY_URL}`,
  );
  const rows = [];
  let next = 0;
  async function worker() {
    while (next < jobs.length) {
      const job = jobs[next++];
      rows.push(await runJob(job));
    }
  }
  await Promise.all(Array.from({ length: CONCURRENCY }, worker));

  const header = ["| group | runs | p50 | p95 | success |", "|---|---|---|---|---|"];
  if (MODE === "chain") {
    console.log("\nPer app:");
    console.log([...header, ...summarise(rows, (r) => r.app)].join("\n"));
    console.log("\nPer serving tier (successful runs):");
    console.log(
      [
        ...header,
        ...summarise(
          rows.filter((r) => r.ok),
          (r) => r.served ?? "none",
        ),
      ].join("\n"),
    );
    console.log("\nOverall:");
    console.log([...header, ...summarise(rows, () => "all")].join("\n"));
  } else {
    console.log("\nPer tier and app (pinned):");
    console.log(
      [...header, ...summarise(rows, (r) => `${r.route} / ${r.app}`)].join("\n"),
    );
    console.log("\nPer tier (pinned):");
    console.log([...header, ...summarise(rows, (r) => r.route)].join("\n"));
  }
}

await main();
