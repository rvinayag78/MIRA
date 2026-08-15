# Infra

Postgres 16 + pgvector and Redis are defined in the root [`docker-compose.yml`](../docker-compose.yml).

Schema is applied on first DB boot via:

`apps/api/app/db/schema.sql`

Re-apply manually:

```bash
psql "$DATABASE_URL" -f apps/api/app/db/schema.sql
# or
cd apps/api && python -m app.db.migrate
```

## Live demo

Free path (Render + Neon + Upstash): [RENDER.md](./RENDER.md).

Railway (paid multi-service): [RAILWAY.md](./RAILWAY.md).
