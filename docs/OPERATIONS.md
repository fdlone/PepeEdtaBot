# Operations Runbook

This document covers the minimum operational procedures for a long-running
`PepeEdtaBot` deployment.

## Scope

The bot uses:

- long polling to the Telegram Bot API;
- SQLite in WAL mode;
- stdout logging;
- Docker / Compose as the primary deployment shape in this repository.

There is no HTTP admin endpoint, no built-in metrics endpoint, and no
application-level backup scheduler.

## Logging

The application writes plain logs to stdout. Rotation must be configured by the
runtime around the container or process.

Recommended minimum:

- keep at least 3 rotated log files;
- cap one log file at a finite size;
- monitor container restart count and error bursts separately from log files.

### Docker daemon `json-file` rotation

If the host uses Docker's default `json-file` log driver, configure daemon
rotation on the host, for example:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

This is a host-level setting, not a repository file.

### Journald / platform-managed logs

If the container platform already rotates stdout logs, verify:

- retention is finite;
- logs are searchable by container name;
- restart events and stderr are retained together with stdout.

## SQLite WAL Maintenance

The bot enables `PRAGMA journal_mode=WAL`. During normal operation this creates
the main database plus optional `-wal` and `-shm` side files.

Files to treat as one logical database:

- `data/markov.db`
- `data/markov.db-wal`
- `data/markov.db-shm`

### When to run a checkpoint

Run a manual checkpoint when:

- the `-wal` file grows unexpectedly;
- you want a cleaner cold backup;
- the bot was restarted after a crash and WAL size stays high.

### Manual checkpoint command

Run on the deployment host from the repository root:

```bash
python -c "import sqlite3; conn = sqlite3.connect('data/markov.db'); print(conn.execute('PRAGMA wal_checkpoint(TRUNCATE);').fetchall()); conn.close()"
```

Expected result shape: one SQLite row from `wal_checkpoint`. After a successful
truncate checkpoint, the WAL file should shrink substantially or disappear when
idle.

For the most conservative maintenance window, stop the bot before the command.

## Backups

Two safe backup modes are supported here.

### Option A: cold file copy

Use when you can stop the bot briefly.

1. Stop the bot container or process.
2. Copy `data/markov.db`, `data/markov.db-wal`, and `data/markov.db-shm` if
   they exist.
3. Start the bot again.

If you stop the bot cleanly and run a checkpoint first, usually only
`data/markov.db` remains, but the procedure should still tolerate WAL/SHM
files.

### Option B: online SQLite backup

Use when you need a backup without stopping the bot.

```bash
python -c "import sqlite3; src = sqlite3.connect('data/markov.db'); dst = sqlite3.connect('data/markov.backup.sqlite'); src.backup(dst); dst.close(); src.close()"
```

After the online backup finishes, optionally run a WAL checkpoint to reduce WAL
growth on the live database.

### Backup verification

A backup is not complete until it is verified.

Minimum verification:

```bash
python -c "import sqlite3; conn = sqlite3.connect('data/markov.backup.sqlite'); print(conn.execute('PRAGMA integrity_check;').fetchone()); conn.close()"
```

Expected result: `('ok',)`.

## Restore

Restore only with the bot stopped.

1. Stop the bot container or process.
2. Move the current live files out of the way:
   `data/markov.db`, `data/markov.db-wal`, `data/markov.db-shm`.
3. Copy the chosen backup into place as `data/markov.db`.
4. Ensure file ownership matches the runtime user.
5. Start the bot.
6. Review startup logs and run a quick smoke check with `/ping` and `/stats`.

If the backup was created as a single SQLite file, do not restore stale `-wal`
or `-shm` files next to it.

## Minimal Post-Restart Checks

After restart or restore, verify:

- container/process is running;
- logs show normal startup without migration failure;
- `/ping` responds;
- `/stats` works in a group chat;
- `data/markov.db` is writable by the runtime user.

## Database Retention

The `messages` table is bounded automatically per chat. After each insert, the
bot atomically keeps only the newest `MESSAGES_RETENTION_PER_CHAT` rows for that
chat. The configured cap must be at least `TEXT_CACHE_MAX_MESSAGES`, otherwise
startup fails to protect verbatim-copy detection.

The aggregated Markov tables (`starts`, `starts3`, `transitions`,
`transitions3`, `transitions1`) are not pruned and continue to represent all
learned history. `pivo_daily_usage` is cleaned up automatically on bot startup
via `cleanup_pivo_daily_usage`. The dialogue-flavor tables `chat_emoji_stats`
and `chat_hot_ngrams` decay on startup and then lazily about once a day from
the learning path (counts not bumped within 7 days are halved, rows reaching
zero are deleted), so their sliding windows keep moving without restarts.

Deleting rows does not shrink the main SQLite file by itself. Run `VACUUM`
during a maintenance window to return unused pages to the filesystem. A manual
`PRAGMA wal_checkpoint(TRUNCATE)` may still be needed to shrink the WAL file.

### Checking per-table disk usage

```bash
python -c "
import sqlite3
conn = sqlite3.connect('data/markov.db')
rows = conn.execute('SELECT name, SUM(pgsize) FROM dbstat GROUP BY name ORDER BY 2 DESC').fetchall()
for name, size in rows:
    print(f'{name}: {size // 1024} KB')
conn.close()
"
```

Note: `starts`, `starts3`, `transitions`, `transitions3`, `transitions1` tables
store aggregated counts derived from message text — they are not tied to
individual `messages` rows. To reset all Markov data for a chat, use the `/clear`
command inside the group (admin-only). The bot will rebuild its model naturally
as new messages arrive.

### Recommended maintenance schedule

For a small active group (up to ~10 users, continuous use):

- Monthly: check database file size; checkpoint WAL if needed.
- Quarterly: review per-table size and schedule `VACUUM` if deleted pages should
  be returned to the filesystem.

## Healthcheck Reality

The Docker `HEALTHCHECK` verifies that Python starts and that the SQLite
database file exists and is readable. It does not prove:

- Telegram polling is healthy;
- SQLite writes succeed;
- the event loop is making progress.

Treat the healthcheck as a minimal smoke probe, not a strong readiness signal.
