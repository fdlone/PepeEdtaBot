-- 019_markov_collocations  (Markov 2.0R, M2R-300/320 — TZ §10, §13)
--
-- Per-chat registry of word pairs that occur together far more often than their
-- individual frequencies predict. Association, not frequency: the chain already
-- knows what is common, and "common" is exactly what cannot tell a meme from a
-- filler word.
--
-- The registry only ever feeds SCORING (ADR-016). Collocations are never glued
-- into single tokens: a one-way tokenization change with no rebuild path
-- (ADR-005) would leave `KEK_LOL` and `кек`+`лол` coexisting forever, and no
-- retirement could clean that up. Retiring an entry here is safe by
-- construction — a scoring rule stops applying, and no stored text moves.
--
-- `meme_score` is stored alongside `pmi` even though TZ §13 lists only the
-- latter: the ranking, the /stats view and the manual top-20 export all read the
-- composite score, and recomputing it in three places would be three chances to
-- disagree about what the ranking was.
--
-- WITHOUT ROWID: the primary key is the whole row's identity and the table is
-- read by exact chat+pair lookup, the same shape as chat_verbatim_ngrams.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

CREATE TABLE IF NOT EXISTS markov_collocations (
    chat_id     INTEGER NOT NULL,
    left_token  TEXT    NOT NULL,
    right_token TEXT    NOT NULL,
    joint_count INTEGER NOT NULL DEFAULT 0,
    pmi         REAL    NOT NULL DEFAULT 0,
    meme_score  REAL    NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'candidate',
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (chat_id, left_token, right_token)
) WITHOUT ROWID;
