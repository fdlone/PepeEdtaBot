-- 010_pivo_pool_usage
--
-- Dialogue-generation improvement S2 (dialogue-liveliness track, stage 2; see docs/CLOSED.md):
-- /pivo weighted anti-repeat. The message builder picks top/body/bottom parts
-- uniformly, so the same intros become recognizable within a few calls. To avoid
-- that, remember the last N indices used per pool per chat and exclude them from
-- the next pick.
--
-- The table is keyed by (chat_hash, pool_name) where pool_name is
-- "<variant>_<slot>" (variant: default|target, slot: top|body|bottom). It is tiny
-- (bounded by chats x pools, ~6 rows per chat) so it needs no time-based
-- retention, matching how chat_members is keyed by HMAC hash and not day-pruned.
-- recent_indices is a comma-separated list of ints, most-recent last.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

CREATE TABLE IF NOT EXISTS pivo_pool_usage (
    chat_hash      TEXT NOT NULL,
    pool_name      TEXT NOT NULL,
    recent_indices TEXT NOT NULL DEFAULT '',
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_hash, pool_name)
);
