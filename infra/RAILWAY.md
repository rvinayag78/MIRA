# Deploy MIRA on Railway

One public URL. Visitors use Maker and Keeper in the browser; Next.js proxies `/backend` to FastAPI on the private network. API and the ARQ worker run in the same container so recordings stay on a shared volume.

## Services

| Service | Image / build | Public? |
|---------|----------------|---------|
| `db` | `pgvector/pgvector:pg16` | no |
| `redis` | `redis:7-alpine` | no |
| `api` | `apps/api` (uvicorn + worker) | no |
| `web` | `apps/web` production Next.js | **yes — share this** |

## 1. Put the repo on GitHub

Railway builds from GitHub. Push `main` to `https://github.com/rvinayag78/MIRA.git`.

## 2. Create the project from Compose

1. [Railway](https://railway.com) → **New project** → **Deploy from GitHub repo** → `MIRA`.
2. If Railway asks for a Compose file, choose `docker-compose.railway.yml`.
3. If it only detects `docker-compose.yml` (local dev), skip that import and add the four services below from the repo instead.

### Manual services (if Compose import is not used)

Create an empty Railway project, then:

1. **db** — New service → Docker image `pgvector/pgvector:pg16`.  
   Variables: `POSTGRES_USER=mira`, `POSTGRES_PASSWORD=<random>`, `POSTGRES_DB=mira`.  
   Mount a volume at `/var/lib/postgresql/data`.
2. **redis** — New service → Docker image `redis:7-alpine`.
3. **api** — New service → GitHub repo, root directory `apps/api`.  
   Custom start command: `/app/start.sh`.  
   Mount a volume at `/data`.
4. **web** — New service → GitHub repo, root directory `apps/web`.  
   Dockerfile target `prod`.  
   **Generate a public domain** on this service.

API variables:

```
DATABASE_URL=postgresql://mira:${{db.POSTGRES_PASSWORD}}@${{db.RAILWAY_PRIVATE_DOMAIN}}:5432/mira
REDIS_URL=redis://${{redis.RAILWAY_PRIVATE_DOMAIN}}:6379
AUDIO_DIR=/data/audio
TTS_DIR=/data/tts
API_CORS_ORIGINS=*
RERANK_ENABLED=false
PORT=8000
ASSEMBLYAI_API_KEY=
VOYAGE_API_KEY=
ANTHROPIC_API_KEY=
ELEVENLABS_API_KEY=
```

Web variables:

```
API_INTERNAL_URL=http://${{api.RAILWAY_PRIVATE_DOMAIN}}:8000
NEXT_PUBLIC_API_URL=/backend
PORT=3000
HOSTNAME=0.0.0.0
```

Set the four API keys in the Railway UI (never commit them). After the first web deploy, confirm `NEXT_PUBLIC_API_URL=/backend` was present at **build** time — it is inlined into the browser bundle.

## 3. Share the demo

Open the **web** service → **Settings** → **Networking** → **Generate domain**.

That URL is the live demo (`/maker` to record, then share `/k/{agent_id}?token=...`).

## 4. Local check of the production images

```bash
cp .env.example .env   # fill keys
docker compose -f docker-compose.railway.yml up --build
```

Then visit http://localhost:3000.

## Cost / abuse

Anyone with the URL can create agents and burn AssemblyAI, Voyage, Anthropic, and ElevenLabs credits. Use Railway spend limits and provider billing caps. This is a demo, not a multi-tenant product.

## Render

Same split works on Render (web + API Docker services, Postgres with `CREATE EXTENSION vector`, Redis, disk on `/data`). Railway is the path this repo is wired for.
