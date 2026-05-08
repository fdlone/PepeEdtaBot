CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL,
    author_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS starts (
    chat_id INTEGER NOT NULL,
    w1 TEXT NOT NULL,
    w2 TEXT NOT NULL,
    cnt INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2)
);

CREATE TABLE IF NOT EXISTS transitions (
    chat_id INTEGER NOT NULL,
    w1 TEXT NOT NULL,
    w2 TEXT NOT NULL,
    w3 TEXT NOT NULL,
    cnt INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2, w3)
);

CREATE TABLE IF NOT EXISTS starts3 (
    chat_id INTEGER NOT NULL,
    w1 TEXT NOT NULL,
    w2 TEXT NOT NULL,
    w3 TEXT NOT NULL,
    cnt INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2, w3)
);

CREATE TABLE IF NOT EXISTS transitions3 (
    chat_id INTEGER NOT NULL,
    w1 TEXT NOT NULL,
    w2 TEXT NOT NULL,
    w3 TEXT NOT NULL,
    w4 TEXT NOT NULL,
    cnt INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2, w3, w4)
);

CREATE TABLE IF NOT EXISTS transitions1 (
    chat_id INTEGER NOT NULL,
    w1 TEXT NOT NULL,
    w2 TEXT NOT NULL,
    cnt INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(chat_id, w1, w2)
);

CREATE TABLE IF NOT EXISTS pivo_chat_members (
    chat_hash TEXT NOT NULL,
    user_hash TEXT NOT NULL,
    encrypted_user_id TEXT NOT NULL,
    encrypted_username TEXT NOT NULL DEFAULT '',
    encrypted_display_name TEXT NOT NULL DEFAULT '',
    is_bot INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY(chat_hash, user_hash)
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_starts_lookup ON starts(chat_id, w1, w2);
CREATE INDEX IF NOT EXISTS idx_transitions_lookup ON transitions(chat_id, w1, w2);
CREATE INDEX IF NOT EXISTS idx_starts3_chat_id ON starts3(chat_id);
CREATE INDEX IF NOT EXISTS idx_transitions3_lookup ON transitions3(chat_id, w1, w2, w3);
CREATE INDEX IF NOT EXISTS idx_transitions1_lookup ON transitions1(chat_id, w1);
CREATE INDEX IF NOT EXISTS idx_pivo_chat_members_chat_hash ON pivo_chat_members(chat_hash)
