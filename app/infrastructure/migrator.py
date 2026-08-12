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
        try:
            # _apply records the migration into schema_migrations inside the
            # migration's own transaction: a crash between "applied" and
            # "recorded" would otherwise re-run a non-idempotent migration.
            await _apply(conn, path, record_stem=stem)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


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
        if f.is_file() and _MIGRATION_RE.match(f.name)
    )
    return [(f.stem, f) for f in files if f.stem not in applied]


async def _apply(
    conn: aiosqlite.Connection, path: Path, record_stem: str | None = None
) -> None:
    """Applies one migration file; with ``record_stem`` also records it.

    The schema_migrations INSERT joins the migration's own transaction so
    "applied" and "recorded" commit together. Tests re-apply migrations by
    hand without recording — hence the parameter is optional.
    """
    if path.suffix == ".sql":
        # executescript correctly handles multi-statement scripts (including
        # triggers with BEGIN..END blocks, semicolons inside string literals,
        # and inline comments) — all corner cases the previous naive
        # str.split(";") splitter could not.
        #
        # However, sqlite3.executescript implicitly issues a COMMIT before
        # running the script, which would auto-commit each DDL statement as
        # it runs. To preserve atomicity (so a failure halfway through leaves
        # no partially applied schema), we explicitly wrap the script content
        # in BEGIN ... COMMIT. On failure, run() catches the exception and
        # calls conn.rollback(), which drops the in-flight transaction and
        # reverts every DDL statement that ran since BEGIN.
        #
        # Convention: migration .sql files MUST NOT contain their own BEGIN
        # or COMMIT statements — the runner adds them.
        # rstrip() + ensure trailing ';' so the appended COMMIT does not
        # collide with the last statement when the file omits the
        # terminator (some of our migrations do).
        sql = path.read_text(encoding="utf-8").rstrip()
        if not sql.endswith(";"):
            sql += ";"
        record = ""
        if record_stem is not None:
            # The stem comes from _MIGRATION_RE-matched filenames (\w+ only),
            # never from user input, so the interpolation is safe — and it has
            # to be inline: executescript takes no bound parameters.
            record = (
                f"INSERT INTO schema_migrations(name) VALUES ('{record_stem}');\n"
            )
        await conn.executescript(f"BEGIN;\n{sql}\n{record}COMMIT;")
    else:
        spec = importlib.util.spec_from_file_location(f"_migration_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        await mod.apply(conn)
        if record_stem is not None:
            # .py migrations run statements on the caller's connection without
            # committing; this INSERT joins that open transaction and run()
            # commits both together.
            await conn.execute(
                "INSERT INTO schema_migrations(name) VALUES (?)", (record_stem,)
            )
