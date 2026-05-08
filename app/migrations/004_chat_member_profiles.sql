CREATE TABLE chat_member_profiles (
    chat_hash         TEXT    NOT NULL,
    user_hash         TEXT    NOT NULL,
    encrypted_user_id TEXT    NOT NULL,
    encrypted_payload TEXT    NOT NULL,
    key_version       INTEGER NOT NULL DEFAULT 1,
    consent_version   INTEGER NOT NULL,
    consented_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY(chat_hash, user_hash)
);
CREATE INDEX idx_chat_member_profiles_chat_hash ON chat_member_profiles(chat_hash)
