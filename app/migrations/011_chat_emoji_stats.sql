-- 011_chat_emoji_stats
--
-- Dialogue-generation improvement M3 ([DIALOGUE_GENERATION_ACTION_PLAN] Stage 3):
-- emoji channel. The Markov model drops emojis (TOKEN_RE keeps only \w+ and a few
-- punctuation marks), so the bot never speaks the chat's emoji vocabulary. This
-- table records a per-chat emoji frequency so replies can occasionally end with an
-- emoji this chat actually uses, without polluting the word model.
--
-- Keyed by raw chat_id to match the Markov model tables (messages, starts,
-- transitions...), which are all keyed by chat_id and cleared together in
-- clear_chat. Emojis are not PII; the table is per-chat aggregate only (no author).
-- updated_at drives the decay maintenance (stale rows are halved so dead memes
-- fade). The migration runner wraps this script in BEGIN/COMMIT; do not add them.

CREATE TABLE IF NOT EXISTS chat_emoji_stats (
    chat_id    INTEGER NOT NULL,
    emoji      TEXT NOT NULL,
    cnt        INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_id, emoji)
);
