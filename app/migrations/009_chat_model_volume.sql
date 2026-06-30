-- 009_chat_model_volume
--
-- Audit finding D2 ([07_database], [09_performance] P2): save_message_and_update_model
-- ran two SELECT SUM(cnt) ... WHERE chat_id=? aggregations on every learned
-- message to compute the model "volume" (the readiness gate value). That is an
-- O(rows-per-chat) scan on the hottest write path, growing with the model.
--
-- Replace it with a per-chat running total maintained incrementally. transitions
-- counts are only ever written by save_message_and_update_model (which now also
-- increments this table) and removed by clear_chat (which resets the row), so the
-- totals stay exactly in sync with SUM(cnt) over the transitions tables.
--
-- This migration creates the table and backfills it from the current sums for
-- every existing chat.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

CREATE TABLE IF NOT EXISTS chat_model_volume (
    chat_id INTEGER PRIMARY KEY,
    volume2 INTEGER NOT NULL DEFAULT 0,
    volume3 INTEGER NOT NULL DEFAULT 0
);

INSERT INTO chat_model_volume (chat_id, volume2)
SELECT chat_id, SUM(cnt) FROM transitions GROUP BY chat_id
ON CONFLICT(chat_id) DO UPDATE SET volume2 = excluded.volume2;

INSERT INTO chat_model_volume (chat_id, volume3)
SELECT chat_id, SUM(cnt) FROM transitions3 GROUP BY chat_id
ON CONFLICT(chat_id) DO UPDATE SET volume3 = excluded.volume3;
