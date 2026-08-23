-- 022_phrase_ngrams  (PRE, M3R-000 — change derive-phrase-index)
--
-- Storage for the content phrase index: per-chat adjacent bigrams and trigrams
-- of words, with their all-time counts. Nothing reads it yet — the phrase route
-- (M3R-210) arrives as its own change behind its own gate, and this migration
-- must leave generation byte-identical.
--
-- Shape and key mirror `chat_hot_ngrams` (bigrams stored with w3 = '') so the
-- two phrase-shaped tables read the same way. The lifecycle mirrors
-- `chat_verbatim_ngrams` instead: message retention does not trim this table
-- and no decay is applied to it. That difference is the whole point — the hot
-- table's count is a WINDOW count and its stale rows are halved and purged
-- (`DecayableCountsRepo._decay_stale`), which is why the all-time count could
-- not simply become another column there.
--
-- Deliberately NOT backfilled, unlike migration 016. There the backfill had no
-- alternative — window 4-grams are unrecoverable otherwise — and 016 inlined a
-- copy of the tokenizer on purpose, because migrations are frozen history and
-- must not drift with the app code they backfilled for. Here the backfill and
-- the recurring rebuild are the SAME operation over `transitions`, so inlining
-- a copy of it would create exactly the drift 016 warns about. The first
-- background rebuild fills the table from the whole chain; until then it is
-- empty, which costs nothing while it has no readers.
--
-- WITHOUT ROWID for the same reason chat_verbatim_ngrams uses it: the row is
-- its key plus one integer, so a separate rowid would be pure overhead.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

CREATE TABLE IF NOT EXISTS chat_phrase_ngrams (
    chat_id INTEGER NOT NULL,
    w1      TEXT    NOT NULL,
    w2      TEXT    NOT NULL,
    w3      TEXT    NOT NULL DEFAULT '',
    cnt     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (chat_id, w1, w2, w3)
) WITHOUT ROWID;
