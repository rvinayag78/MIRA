"""Apply schema.sql to DATABASE_URL."""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from app.config import get_settings


async def main() -> None:
    settings = get_settings()
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    conn = await asyncpg.connect(settings.database_url)
    try:
        await conn.execute(schema)
        print("Schema applied.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
