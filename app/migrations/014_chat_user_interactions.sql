-- 014_chat_user_interactions
--
-- Dialogue-generation improvement L2 ([DIALOGUE_GENERATION_ACTION_PLAN] Stage 4):
-- per-user quirks. Counts how many times the bot answered a given user's
-- direct address, so "regulars" can occasionally get a short vocative message
-- before the reply. Rows decay on the flavor-decay cadence (see
-- Database.decay_chat_user_interactions) so regular status fades after ~a
-- month of silence instead of accruing forever.
--
-- Privacy: raw chat_id matches the model tables and rides clear_chat's
-- one-statement wipe; user_hash is HMAC-SHA256 under PIVO_HMAC_SECRET — the
-- same anonymization as /pivo subscriptions: no names, no usernames, no
-- reversible identity, no message payload. Only a counter is stored.
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them.

CREATE TABLE IF NOT EXISTS chat_user_interactions (
    chat_id    INTEGER NOT NULL,
    user_hash  TEXT NOT NULL,
    cnt        INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_id, user_hash)
);
