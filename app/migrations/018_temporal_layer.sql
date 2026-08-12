-- 018_temporal_layer  (Markov 2.0R, M2R-200 — TZ §7 / §13)
--
-- Gives every transition and start a temporal record: when it was first and
-- last observed (the long layer's bookkeeping) and the exponentially decaying
-- short-layer pair (s_value, s_updated_at) defined in TZ §7.1.
--
-- Existing rows keep NULL timestamps ON PURPOSE. Historical observation times
-- are unrecoverable (rebuilding the chain is forbidden, ADR-005), and the audit
-- (§8) flagged the alternative — stamping every old row with the migration date
-- — as a systemic risk: two lines later in a report, a stamped date is
-- indistinguishable from a real observation. NULL cannot be mistaken for
-- history, so readers treat NULL first_seen as "predates the temporal record"
-- and are required to say so in any metric derived from it.
--
-- s_value defaults to 0 with a NULL s_updated_at, i.e. an empty short layer:
-- the chat's fresh-language memory starts accumulating from the first message
-- learned after this migration, not from a fabricated past.
--
-- Cost: ALTER TABLE ADD COLUMN is a metadata-only operation in SQLite and no
-- row is rewritten here.
--
-- The migration runner wraps this script in BEGIN/COMMIT; do not add them here.

ALTER TABLE starts ADD COLUMN first_seen    INTEGER;
ALTER TABLE starts ADD COLUMN last_seen     INTEGER;
ALTER TABLE starts ADD COLUMN s_value       REAL NOT NULL DEFAULT 0;
ALTER TABLE starts ADD COLUMN s_updated_at  INTEGER;

ALTER TABLE starts3 ADD COLUMN first_seen   INTEGER;
ALTER TABLE starts3 ADD COLUMN last_seen    INTEGER;
ALTER TABLE starts3 ADD COLUMN s_value      REAL NOT NULL DEFAULT 0;
ALTER TABLE starts3 ADD COLUMN s_updated_at INTEGER;

ALTER TABLE transitions ADD COLUMN first_seen   INTEGER;
ALTER TABLE transitions ADD COLUMN last_seen    INTEGER;
ALTER TABLE transitions ADD COLUMN s_value      REAL NOT NULL DEFAULT 0;
ALTER TABLE transitions ADD COLUMN s_updated_at INTEGER;

ALTER TABLE transitions3 ADD COLUMN first_seen   INTEGER;
ALTER TABLE transitions3 ADD COLUMN last_seen    INTEGER;
ALTER TABLE transitions3 ADD COLUMN s_value      REAL NOT NULL DEFAULT 0;
ALTER TABLE transitions3 ADD COLUMN s_updated_at INTEGER;
