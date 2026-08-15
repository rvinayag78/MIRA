# Free live demo (Render)

Railway wants a paid plan for several services. This path uses **one** free Render web service plus two free managed datastores.

| Piece | Where | Cost |
|-------|--------|------|
| UI + API + worker | Render free web service (sleeps after ~15 min idle) | $0 |
| Postgres + pgvector | [Neon](https://neon.tech) Free | $0 |
| Redis | [Upstash](https://upstash.com) Free | $0 |

Share the Render URL. First load after idle can take about a minute.

## 1. Neon (Postgres)

1. Create a project at [neon.tech](https://neon.tech) (no card).
2. Dashboard → **Connect** → copy the **direct** connection string, not the pooled/`-pooler` one. MIRA uses session `SET LOCAL` / RLS, which breaks on the pooler.
3. It should look like `postgresql://...@ep-....neon.tech/neondb?sslmode=require`.

Schema (including `CREATE EXTENSION vector`) runs when the app boots.

## 2. Upstash (Redis)

1. Create a Redis database at [upstash.com](https://upstash.com).
2. Copy `REDIS_URL` (`rediss://...`).

## 3. Render

1. [Render](https://render.com) → **New** → **Blueprint**.
2. Connect `rvinayag78/MIRA` and use `render.yaml`.
3. Set these env vars (Blueprint marks them as needed):

   - `DATABASE_URL` — Neon **direct** URL
   - `REDIS_URL` — Upstash URL
   - `ASSEMBLYAI_API_KEY`
   - `VOYAGE_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `ELEVENLABS_API_KEY`

4. Deploy. The public URL is `https://mira-....onrender.com`.

Or: **New Web Service** → this repo → Docker → Dockerfile at repo root → instance type **Free**.

## Limits

- Free Render sleeps after 15 minutes idle; the next visitor waits on a cold start.
- Audio files are ephemeral (no disk on free). Text chat still works after sleep because transcripts live in Neon; voice clone / TTS files do not persist.
- Anyone with the URL can burn provider credits. Cap keys and Render spend.

## Always-on alternative

A ~$5/mo VPS (Hetzner/DigitalOcean) can run `docker compose -f docker-compose.railway.yml up --build` with no platform paywall.
