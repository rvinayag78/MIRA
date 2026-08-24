#!/bin/sh
# Boot API + worker + Next.js for the single-container Render demo.
# Always bind Next on $PORT so Render health checks don't HTML-502 when migrate is slow.

cd /api

echo "Running database migrate..."
if python -m app.db.migrate; then
  echo "Migrate ok"
else
  echo "WARNING: database migrate failed — API may be degraded"
fi

arq app.workers.settings.WorkerSettings &
uvicorn app.main:app --host 127.0.0.1 --port 8000 &

i=0
until python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; do
  i=$((i + 1))
  if [ "$i" -ge 60 ]; then
    echo "WARNING: API failed to become healthy — starting web UI anyway"
    break
  fi
  sleep 1
done

cd /web
export HOSTNAME=0.0.0.0
export PORT="${PORT:-3000}"
echo "Starting Next.js on 0.0.0.0:${PORT}"
exec node server.js
