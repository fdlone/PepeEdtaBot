-- 013_drop_transitions1
--
-- The order-1 Markov chain was removed from generation: the 2026-07-09
-- default BACKOFF_MIN_ORDER=2 already forbade 1-gram walks (word salad per
-- eval_prod), and the knob itself is gone now that order 2 is the only floor.
-- The last reader of this table — the hot-ngram bigram hotness denominator —
-- now aggregates SUM(cnt) over transitions(w1, w2, *) instead, so the table
-- only added write amplification on every learned message.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

DROP INDEX IF EXISTS idx_transitions1_lookup;
DROP TABLE IF EXISTS transitions1;
