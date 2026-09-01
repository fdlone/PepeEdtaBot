-- Глобальные метаданные обслуживания (spec maintenance-cadence).
-- Ключ last_flavor_decay_at держит момент последнего полного суточного пасса,
-- чтобы каденция переживала рестарт: монотонный таймер по построению не мог.
-- Таблица глобальная, не пер-чатовая, поэтому в clear_chat/forget_chat
-- намеренно не входит (ловушка §5 CLAUDE.md здесь неприменима).
CREATE TABLE IF NOT EXISTS maintenance_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
