-- 020_reverse_index_and_token_df  (Markov 2.0R, M2R-400 — TZ §9.2, §9.3, ADR-012)
--
-- Data layer for statistical lexical anchoring. Nothing reads these structures
-- yet: seeded generation (M2R-410) arrives as its own change behind its own
-- gate, and this migration must leave generation byte-identical.
--
-- Reverse order-2 lookups are served by an INDEX over the existing forward
-- table, not by the `markov_reverse` duplicate sketched in TZ §13 (recorded
-- deviation — change markov2r-phase5-reverse-index, design D1). A reverse
-- transition IS the forward row read by (w2, w3): the index gives consistency
-- by construction (no second copy to drift), a fraction of the duplicate
-- table's storage, backfill as a single CREATE INDEX, and rollback as a single
-- DROP INDEX. Migration 008's rule is not violated: it removed indexes that
-- duplicated a PK prefix with no read benefit, while (chat_id, w2, w3) serves
-- a query the PK cannot.
--
-- `markov_token_df` is the incremental document-frequency record (+1 per
-- unique token per learned message). It exists because raw messages are
-- retention-trimmed: df can never be recomputed later, only accumulated.
-- Deliberately NOT backfilled from the retained message window — inventing
-- full-history frequencies from a trimmed window would be inventing history.
--
-- `n_docs` (total learned messages per chat) joins chat_model_volume, whose
-- row is already upserted on every learned message.
--
-- WITHOUT ROWID: read by exact chat+token lookup, same shape as
-- chat_verbatim_ngrams and markov_collocations.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

CREATE INDEX IF NOT EXISTS idx_transitions_reverse
    ON transitions (chat_id, w2, w3);

CREATE TABLE IF NOT EXISTS markov_token_df (
    chat_id       INTEGER NOT NULL,
    token         TEXT    NOT NULL,
    messages_seen INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (chat_id, token)
) WITHOUT ROWID;

ALTER TABLE chat_model_volume ADD COLUMN n_docs INTEGER NOT NULL DEFAULT 0;
