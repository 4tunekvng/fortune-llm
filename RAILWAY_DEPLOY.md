# Deploying portfolio apps to Railway

This is the unified recipe for the apps that won't run on Vercel — either
because they need a persistent SQLite file (Drizzle / better-sqlite3 apps)
or they're Python-based.

The pattern below is **the same for every app** — different repo name, sometimes a
different env var name for the DB path. Once you've done it once it's ~5 min per app.

## What you do once

1. Sign up at **[railway.com](https://railway.com)** with GitHub.
2. Generate a Railway token from your account settings if you want to use the CLI.
   The dashboard alone is enough for the deploys below, no CLI required.

## What you do per app

### Stage 1 — Create the project

1. Click **New Project** → **Deploy from GitHub repo**.
2. Pick the repo from the list. Railway auto-detects Node.js / Next.js / Python from `package.json` or `pyproject.toml`. No Dockerfile required.
3. The first deploy will start automatically — and may fail because the env vars / volume aren't set yet. That's expected.

### Stage 2 — Add a persistent volume (only for SQLite apps)

In the project's **Variables → Volume** section (or the **+ Volume** button):

1. Click **Add Volume**.
2. **Mount path**: `/data`.
3. Name: anything (e.g. `app-data`).
4. Click **Create**.

Apps that need this:
| Repo | Env var to set | Value |
|---|---|---|
| `delegate` | `DATABASE_PATH` | `/data/delegate.db` |
| `knowledge-compounder` | `DATABASE_PATH` | `/data/knowledge.db` |
| `weekend-cofounder` | `WC_DB_PATH` | `/data/cofounder.db` |
| `network-agent` (server only) | `NETWORK_DB_PATH` | `/data/network.db` |
| `morphos` | n/a (no persistent state) | — skip volume |
| `personal-world-model` | `BACKEND_DB_PATH` | `/data/pwm.db` |

### Stage 3 — Set environment variables

In **Variables**, add these for every app (the gateway lets users use the app without their own Anthropic key):

```
ANTHROPIC_BASE_URL = https://fortune-llm.fortunee.workers.dev
ANTHROPIC_API_KEY  = <your gateway token>      ← mark sensitive
```

App-specific extras:

| Repo | Extra vars |
|---|---|
| `weekend-cofounder` | `GITHUB_TOKEN` (your read-only PAT — gives the briefing engine access to your repo); also set `WC_TEST_MODE=0` to disable the fakes |
| `network-agent` (server) | `IMESSAGE_DB_PATH` only if you want real iMessage import; omit for the seeded mock dataset |
| `personal-world-model` (backend) | none beyond ANTHROPIC + the volume |
| `Lena/lena_agent` | `LENA_API_BASE_URL=https://test.lena.africa/v1/`, `ANTHROPIC_MODEL=claude-sonnet-4-6` (you've already done this on Vercel) |

### Stage 4 — Generate a public domain

1. **Settings → Networking → Public Networking → Generate Domain**.
2. Railway prints a URL like `delegate-production-abc.up.railway.app`.
3. The deploy redeploys with the public domain. ~30s.

### Stage 5 — Initialize the SQLite file (first deploy only)

Most apps create the schema on first startup (the `ensureSchema(...)` call in `client.ts`).
**knowledge-compounder** is the exception — its schema lives in Drizzle migrations and you'd run them via `railway run npm run db:push` if needed.

For each, hit the public URL once → it should land on the home page or seed-redirect page.

### Stage 6 — Disable any preview-protection toggles

Railway doesn't gate by default, but if you see a login page when opening in incognito, check **Settings → Networking** for an auth toggle and switch it off.

## Per-app quick reference

| Repo | Platform | Volume | DB env var | Extra env |
|---|---|---|---|---|
| `apertus` | Vercel | — (Postgres add-on) | `POSTGRES_PRISMA_URL` | — |
| `lena_agent` | Vercel | — | — | `LENA_API_BASE_URL` |
| `maestro` | Vercel | — | — | — |
| `delegate` | Railway | `/data` | `DATABASE_PATH=/data/delegate.db` | — |
| `knowledge-compounder` | Railway | `/data` | `DATABASE_PATH=/data/knowledge.db` | — |
| `weekend-cofounder` | Railway | `/data` | `WC_DB_PATH=/data/cofounder.db` | `GITHUB_TOKEN`, `WC_TEST_MODE=0` |
| `network-agent` (server) | Railway | `/data` | `NETWORK_DB_PATH=/data/network.db` | (web hosted separately on Vercel) |
| `morphos` | Railway | — | — | — |
| `personal-world-model` (backend) | Railway | `/data` | `BACKEND_DB_PATH=/data/pwm.db` | — |
| `personal-world-model` (frontend) | Vercel | — | — | `VITE_API_URL=<railway backend url>` |

## Smoke test — same for all

After the deploy URL is live, open in **incognito**. Walk the golden path of each app (apertus library, weekend-cofounder briefing, etc.) end-to-end without ever entering an API key. If a network call fails:

- Check Railway **Deployments → Logs** for the runtime error.
- The most common cause is one of: env var typo, sensitive var saved only in Production but the deploy is Preview, or the volume not mounted (you'll see "no such file or directory" trying to write to `/data/...`).

## Why two platforms

Vercel's serverless functions are stateless — perfect for Postgres-backed or stateless apps like apertus / lena_agent / maestro. They can't hold a SQLite file because the filesystem is ephemeral and there are no persistent volumes.

Railway runs apps as long-lived containers with attachable volumes, so SQLite stays put. We could move the SQLite apps to Vercel by switching to Postgres / Turso, but that's a real refactor (every `db.select().all()` becomes async — touches dozens of files). Volumes are the cheaper move.
