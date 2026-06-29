-- 008_drop_redundant_indexes
--
-- Audit finding D1 ([07_database], [09_performance] P4): these secondary
-- indexes duplicate a leftmost prefix of the table's PRIMARY KEY index (or,
-- for messages, a prefix of idx_messages_normalized_lookup). The query
-- planner can serve every real query from the PK autoindex / the kept
-- composite index, so these add only write amplification (extra b-trees
-- maintained on every learned message) with no read benefit.
--
-- Verified with EXPLAIN QUERY PLAN: after dropping all eight, every
-- repository query still SEARCHes via an index (no full table scan).
--
-- Kept on purpose:
--   idx_messages_normalized_lookup  -- (chat_id, normalized_text) verbatim-copy check
--   idx_pivo_daily_usage_day        -- (usage_day) range scan for retention cleanup
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

DROP INDEX IF EXISTS idx_starts_lookup;          -- == PK(chat_id,w1,w2)
DROP INDEX IF EXISTS idx_starts_chat_id;         -- prefix of PK(chat_id,w1,w2)
DROP INDEX IF EXISTS idx_starts3_chat_id;        -- prefix of PK(chat_id,w1,w2,w3)
DROP INDEX IF EXISTS idx_transitions_lookup;     -- prefix of PK(chat_id,w1,w2,w3)
DROP INDEX IF EXISTS idx_transitions3_lookup;    -- prefix of PK(chat_id,w1,w2,w3,w4)
DROP INDEX IF EXISTS idx_transitions1_lookup;    -- prefix of PK(chat_id,w1,w2)
DROP INDEX IF EXISTS idx_chat_members_chat_hash; -- prefix of PK(chat_hash,user_hash)
DROP INDEX IF EXISTS idx_messages_chat_id;       -- prefix of idx_messages_normalized_lookup
