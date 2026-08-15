#!/bin/sh
set -e
python -m app.db.migrate
arq app.workers.settings.WorkerSettings &
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
