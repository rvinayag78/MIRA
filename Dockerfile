FROM node:22-bookworm-slim AS web-builder
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json* ./
RUN npm ci
COPY apps/web/ ./
ARG NEXT_PUBLIC_API_URL=/backend
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /api
COPY apps/api/pyproject.toml ./
COPY apps/api/app ./app
COPY apps/api/start.sh ./start.sh
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -e .

WORKDIR /web
COPY --from=web-builder /web/.next/standalone ./
COPY --from=web-builder /web/.next/static ./.next/static

COPY infra/demo-start.sh /demo-start.sh
RUN chmod +x /demo-start.sh

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV API_INTERNAL_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_API_URL=/backend
ENV AUDIO_DIR=/tmp/mira-audio
ENV TTS_DIR=/tmp/mira-tts
ENV API_CORS_ORIGINS=*

EXPOSE 3000
CMD ["/bin/sh", "/demo-start.sh"]
