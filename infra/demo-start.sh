#!/bin/sh
set -e

cd /api
python -m app.db.migrate
arq app.workers.settings.WorkerSettings &
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

i=0
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; do
  i=$((i + 1))
  if [ "$i" -ge 30 ]; then
    echo "API failed to start"
    exit 1
  fi
  sleep 1
done

cd /web
export HOSTNAME=0.0.0.0
export PORT="${PORT:-3000}"
exec node server.js
