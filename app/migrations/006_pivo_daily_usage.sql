CREATE TABLE pivo_daily_usage (
    chat_hash  TEXT    NOT NULL,
    user_hash  TEXT    NOT NULL,
    usage_day  TEXT    NOT NULL,
    used_count INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY(chat_hash, user_hash, usage_day)
);
CREATE INDEX idx_pivo_daily_usage_day ON pivo_daily_usage(usage_day)
