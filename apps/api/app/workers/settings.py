from arq.connections import RedisSettings

from app.config import get_settings
from app.workers.tasks import clone_agent_voice, ingest_memory


def _redis_settings() -> RedisSettings:
    url = get_settings().redis_url
    # redis://host:port/db
    return RedisSettings.from_dsn(url)


class WorkerSettings:
    functions = [ingest_memory, clone_agent_voice]
    redis_settings = _redis_settings()
    max_jobs = 5
