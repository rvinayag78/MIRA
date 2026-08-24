"""Apply schema.sql and additive migrate_*.sql files to DATABASE_URL."""

from __future__ import annotations

import asyncio
from pathlib import Path

import asyncpg

from app.config import get_settings


def _statements(sql: str) -> list[str]:
    """Split SQL into statements. Handles simple DO $$ ... $$ blocks."""
    statements: list[str] = []
    buf: list[str] = []
    in_dollar = False
    for line in sql.splitlines():
        stripped = line.strip()
        if not in_dollar and (not stripped or stripped.startswith("--")):
            continue
        if "$$" in stripped:
            # Toggle for each $$ pair on the line (DO $$ ... $$; is two toggles).
            count = stripped.count("$$")
            for _ in range(count):
                in_dollar = not in_dollar
        buf.append(line)
        if not in_dollar and stripped.endswith(";"):
            stmt = "\n".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
    leftover = "\n".join(buf).strip()
    if leftover:
        statements.append(leftover)
    return statements


async def _run_sql_file(conn: asyncpg.Connection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    for stmt in _statements(sql):
        await conn.execute(stmt)
    print(f"Applied {path.name}")


def _is_retryable_db_error(exc: BaseException) -> bool:
    """Retry only connection / availability failures, not permanent SQL errors."""
    if isinstance(exc, OSError):
        return True
    if isinstance(
        exc,
        (
            asyncpg.CannotConnectNowError,
            asyncpg.ConnectionDoesNotExistError,
            asyncpg.ConnectionFailureError,
            asyncpg.InterfaceError,
        ),
    ):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "connection refused",
            "could not connect",
            "timeout",
            "server closed the connection",
            "the database system is starting up",
        )
    )


async def apply_schema() -> None:
    settings = get_settings()
    db_dir = Path(__file__).resolve().parent
    connect_kwargs: dict = {"dsn": settings.asyncpg_dsn}
    if settings.asyncpg_ssl is not None:
        connect_kwargs["ssl"] = settings.asyncpg_ssl
    conn = await asyncpg.connect(**connect_kwargs)
    try:
        # Additive ALTERs first when core tables already exist, so schema.sql
        # indexes/policies that assume new columns do not race ahead.
        has_chunks = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'chunks'
            )
            """
        )
        if has_chunks:
            for path in sorted(db_dir.glob("migrate_*.sql")):
                await _run_sql_file(conn, path)

        await _run_sql_file(conn, db_dir / "schema.sql")

        # Fresh installs: schema created tables with new columns; still run
        # migrates for indexes/policies that live only in migrate_*.sql.
        for path in sorted(db_dir.glob("migrate_*.sql")):
            await _run_sql_file(conn, path)
        print("Schema applied.")
    finally:
        await conn.close()


async def main() -> None:
    last_error: Exception | None = None
    for attempt in range(1, 31):
        try:
            await apply_schema()
            return
        except Exception as exc:  # noqa: BLE001 — classify retry vs fail-fast
            last_error = exc
            if not _is_retryable_db_error(exc):
                print(f"Migrate failed (not retrying): {exc}")
                raise RuntimeError("Database migrate failed") from exc
            print(f"Waiting for database ({attempt}/30): {exc}")
            await asyncio.sleep(2)
    raise RuntimeError("Database migrate failed") from last_error


if __name__ == "__main__":
    asyncio.run(main())
