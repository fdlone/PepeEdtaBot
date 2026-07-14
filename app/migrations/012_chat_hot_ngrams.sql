-- 012_chat_hot_ngrams
--
-- Dialogue-generation improvement L1 (dialogue-liveliness track, stage 4; see docs/CLOSED.md):
-- running jokes. Counts content bigrams/trigrams per chat over a sliding window
-- (rows decay/expire at startup, see Database.decay_chat_hot_ngrams). An n-gram
-- is "hot" when its window count is a large share of its all-time count in
-- transitions/transitions1 — i.e. the chat started saying it recently. Hot
-- n-grams occasionally seed unprompted replies via the generator's seed API.
--
-- Privacy: keyed by raw chat_id to match the Markov model tables; per-chat
-- aggregate only (no author), built from the same normalized tokens as the
-- word model, wiped together with it in clear_chat. Bigrams store w3 = ''.
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them.

CREATE TABLE IF NOT EXISTS chat_hot_ngrams (
    chat_id    INTEGER NOT NULL,
    w1         TEXT NOT NULL,
    w2         TEXT NOT NULL,
    w3         TEXT NOT NULL DEFAULT '',
    cnt        INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_id, w1, w2, w3)
);
