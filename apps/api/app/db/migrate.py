"""Apply schema.sql to DATABASE_URL."""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from app.config import get_settings


def _statements(sql: str) -> list[str]:
    """Split schema.sql into single statements (no dollar-quoting in this file)."""
    statements: list[str] = []
    buf: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
    leftover = "\n".join(buf).strip()
    if leftover:
        statements.append(leftover)
    return statements


async def apply_schema() -> None:
    settings = get_settings()
    schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
    conn = await asyncpg.connect(settings.asyncpg_dsn)
    try:
        for stmt in _statements(schema):
            await conn.execute(stmt)
        print("Schema applied.")
    finally:
        await conn.close()


async def main() -> None:
    last_error: Exception | None = None
    for attempt in range(1, 31):
        try:
            await apply_schema()
            return
        except (OSError, asyncpg.PostgresError) as exc:
            last_error = exc
            print(f"Waiting for database ({attempt}/30): {exc}")
            await asyncio.sleep(2)
    raise RuntimeError("Database migrate failed") from last_error


if __name__ == "__main__":
    asyncio.run(main())
