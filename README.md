# MIRA

Maker records memories → AssemblyAI transcription → Voyage embeddings in Postgres/pgvector → ElevenLabs voice clone. Keepers chat with a grounded Claude agent (hybrid RRF retrieval, Haiku route + Sonnet answer).

## Stack

| Layer | Choice |
|-------|--------|
| Transcription | AssemblyAI (batch, diarized) |
| Embeddings | voyage-context-4 @ 1024d |
| Storage | Postgres 16 + pgvector + RLS |
| Retrieval | Hybrid pgvector + FTS, RRF |
| Rerank | Voyage rerank-2.5 (eval-gated, off by default) |
| Generation | Haiku 4.5 (route) + Sonnet 5 (ground) |
| API | Async FastAPI + ARQ workers |
| UI | Next.js App Router |

## Quick start

```bash
cp .env.example .env
# fill API keys

docker compose up --build
```

- Maker UI: http://localhost:3000/maker
- Keeper: share link from maker UI (`/k/{agent_id}?token=...`)
- API docs: http://localhost:8000/docs

## Live demo (Railway)

Shareable public URL for the whole stack (web + API + worker + Postgres/pgvector + Redis):

See [`infra/RAILWAY.md`](infra/RAILWAY.md). Production compose: [`docker-compose.railway.yml`](docker-compose.railway.yml).

### Local without Docker

```bash
# Postgres 16 + pgvector + Redis required
psql "$DATABASE_URL" -f apps/api/app/db/schema.sql

cd apps/api && python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
# another terminal
arq app.workers.settings.WorkerSettings

cd apps/web && npm install && npm run dev
```

## Evals

```bash
cd apps/api && pytest evals/ -q
# or
python -m evals.run_gate
```

Rerank stays disabled (`RERANK_ENABLED=false`) until the CI gate passes Recall@k / faithfulness thresholds.
