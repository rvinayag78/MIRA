# Deploy MIRA on Railway

Railway does **not** build a monorepo from the GitHub root. **New project → Deploy from GitHub** on `MIRA` will fail (no `package.json` / `Dockerfile` at repo root). Create an empty project, then add four services.

One public URL: the **web** service. Next.js proxies `/backend` to the API. API and the ARQ worker run in the same container so recordings stay on one volume.

## Services

| Service | Source | Root directory | Public? |
|---------|--------|----------------|---------|
| `db` | Docker image `pgvector/pgvector:pg16` | — | no |
| `redis` | Docker image `redis:7-alpine` | — | no |
| `api` | this GitHub repo | `apps/api` | no |
| `web` | this GitHub repo | `apps/web` | **yes — share this** |

Do **not** set a Dockerfile target. `apps/web/Dockerfile` is production-only. Railway does not support build targets.

## 1. Empty project + databases

1. [Railway](https://railway.com) → **New project** → **Empty project**.
2. **db** — Add service → Docker image → `pgvector/pgvector:pg16`.  
   Variables: `POSTGRES_USER=mira`, `POSTGRES_PASSWORD=<random>`, `POSTGRES_DB=mira`.  
   Volume at `/var/lib/postgresql/data`.
3. **redis** — Add service → Docker image → `redis:7-alpine`.

Use the pgvector image (not Railway’s default Postgres). Schema needs `CREATE EXTENSION vector`.

## 2. API service

Add service → GitHub repo `rvinayag78/MIRA`.

- **Root directory:** `apps/api` (Settings → Build).
- Volume at `/data`.
- Variables:

```
DATABASE_URL=postgresql://mira:${{db.POSTGRES_PASSWORD}}@${{db.RAILWAY_PRIVATE_DOMAIN}}:5432/mira
REDIS_URL=redis://${{redis.RAILWAY_PRIVATE_DOMAIN}}:6379
AUDIO_DIR=/data/audio
TTS_DIR=/data/tts
API_CORS_ORIGINS=*
RERANK_ENABLED=false
ASSEMBLYAI_API_KEY=
VOYAGE_API_KEY=
ANTHROPIC_API_KEY=
ELEVENLABS_API_KEY=
```

`apps/api/railway.toml` starts `/app/start.sh` (migrate + worker + uvicorn). Set the four API keys in the Railway UI.

## 3. Web service

Add service → GitHub repo `rvinayag78/MIRA`.

- **Root directory:** `apps/web`.
- **Generate a public domain.**
- Variables:

```
API_INTERNAL_URL=http://${{api.RAILWAY_PRIVATE_DOMAIN}}:8000
NEXT_PUBLIC_API_URL=/backend
HOSTNAME=0.0.0.0
```

`NEXT_PUBLIC_API_URL` is inlined at **build** time. Keep it `/backend`.

If the API private URL uses a different port, check the API service’s `PORT` (Railway may set one). Then either pin `PORT=8000` on the API service or point `API_INTERNAL_URL` at that port.

## 4. Share the demo

Web service → Settings → Networking → **Generate domain**.

That URL is the demo (`/maker`, then share `/k/{agent_id}?token=...`).

## Local production images

```bash
cp .env.example .env   # fill keys
docker compose -f docker-compose.railway.yml up --build
```

Visit http://localhost:3000.

## Cost / abuse

Anyone with the URL can create agents and burn provider credits. Use Railway spend limits and provider billing caps.
