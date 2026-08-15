FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY app ./app
COPY start.sh ./start.sh

RUN pip install --no-cache-dir -e . \
    && chmod +x start.sh

EXPOSE 8000

CMD ["/bin/sh", "/app/start.sh"]
