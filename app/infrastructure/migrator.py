"""
Lightweight migration runner.

Discovers migration files in app/migrations/ named NNN_<name>.sql or NNN_<name>.py,
applies them in order, and records each in the schema_migrations table so they run
exactly once.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import aiosqlite

_MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"
_MIGRATION_RE = re.compile(r"^\d{3}_\w+\.(sql|py)$")


async def run(conn: aiosqlite.Connection) -> None:
    await _ensure_table(conn)
    applied = await _get_applied(conn)
    for stem, path in _list_pending(applied):
        await _apply(conn, path)
        await conn.execute(
            "INSERT INTO schema_migrations(name) VALUES (?)", (stem,)
        )
        await conn.commit()


async def _ensure_table(conn: aiosqlite.Connection) -> None:
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL UNIQUE,
            applied_at TEXT  NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    await conn.commit()


async def _get_applied(conn: aiosqlite.Connection) -> set[str]:
    cursor = await conn.execute("SELECT name FROM schema_migrations")
    return {row[0] for row in await cursor.fetchall()}


def _list_pending(applied: set[str]) -> list[tuple[str, Path]]:
    files = sorted(
        f
        for f in _MIGRATIONS_DIR.iterdir()
        if f.is_file() and _MIGRATION_RE.match(f.name) and f.name != "__init__.py"
    )
    return [(f.stem, f) for f in files if f.stem not in applied]


async def _apply(conn: aiosqlite.Connection, path: Path) -> None:
    if path.suffix == ".sql":
        sql = path.read_text(encoding="utf-8")
        for stmt in _split_sql(sql):
            await conn.execute(stmt)
    else:
        spec = importlib.util.spec_from_file_location(f"_migration_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        await mod.apply(conn)


def _split_sql(sql: str) -> list[str]:
    return [s.strip() for s in sql.split(";") if s.strip()]
