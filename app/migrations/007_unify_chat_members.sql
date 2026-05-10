-- Unify chat-member tables.
--
-- Replaces `pivo_chat_members` and the unused `chat_member_profiles`
-- with a single `chat_members` table that:
--   * keeps the same key derivation as pivo (HMAC of chat_id / user_id
--     under PIVO_HMAC_SECRET, Fernet over sha256(PIVO_ENCRYPTION_SECRET))
--     — so existing /pivo subscriptions remain readable without any
--     decrypt+re-encrypt step;
--   * drops the never-read `is_bot` column (bots are filtered in the
--     `/pivo_on` handler before we ever insert a row);
--   * is the canonical place for future features that need persistent
--     per-chat member state.
--
-- The previous `chat_member_profiles` infrastructure (MembersService,
-- MembersRepo, HKDF key derivation) was wired up for hypothetical future
-- domains that never materialised; dropping it removes ~150 lines of
-- code with no behavioural impact.

CREATE TABLE chat_members (
    chat_hash               TEXT NOT NULL,
    user_hash               TEXT NOT NULL,
    encrypted_user_id       TEXT NOT NULL,
    encrypted_username      TEXT NOT NULL DEFAULT '',
    encrypted_display_name  TEXT NOT NULL DEFAULT '',
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY(chat_hash, user_hash)
);

INSERT INTO chat_members(
    chat_hash, user_hash,
    encrypted_user_id, encrypted_username, encrypted_display_name,
    created_at, updated_at
)
SELECT
    chat_hash, user_hash,
    encrypted_user_id, encrypted_username, encrypted_display_name,
    created_at, updated_at
FROM pivo_chat_members;

DROP INDEX IF EXISTS idx_pivo_chat_members_chat_hash;
DROP TABLE pivo_chat_members;

DROP INDEX IF EXISTS idx_chat_member_profiles_chat_hash;
DROP TABLE IF EXISTS chat_member_profiles;

CREATE INDEX idx_chat_members_chat_hash ON chat_members(chat_hash);
