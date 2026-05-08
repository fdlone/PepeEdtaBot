"""Бизнес-логика для хранения профилей участников с шифрованием и opt-in."""
from __future__ import annotations

import hashlib
import hmac
import json

from aiogram.types import User
from cryptography.fernet import Fernet

from app.repositories.members_repo import MembersRepo
from app.security.key_derivation import derive_fernet_key, derive_key

_KEY_VERSION = 1
_HMAC_DOMAIN = "members:hmac"
_ENCRYPTION_DOMAIN = "members:encryption"


class MembersService:
    """
    Хранит профили участников в chat_member_profiles.

    Ключи выводятся через HKDF из существующих PIVO_*_SECRET —
    новых переменных окружения не требуется.
    pivo продолжает использовать сырые секреты напрямую (PivoSecurity не изменён).
    """

    def __init__(
        self,
        repo: MembersRepo,
        pivo_hmac_secret: str,
        pivo_encryption_secret: str,
    ) -> None:
        self._repo = repo
        self._hmac_key = derive_key(pivo_hmac_secret, _HMAC_DOMAIN)
        self._fernet = Fernet(derive_fernet_key(pivo_encryption_secret, _ENCRYPTION_DOMAIN))

    # --- публичный API ---

    async def record_consent(
        self,
        chat_id: int,
        user: User,
        payload: dict[str, object],
        consent_version: int,
    ) -> None:
        """Сохраняет или обновляет профиль участника (upsert)."""
        await self._repo.upsert(
            chat_hash=self._hmac(chat_id),
            user_hash=self._hmac(user.id),
            encrypted_user_id=self._encrypt(user.id),
            encrypted_payload=self._encrypt_json(payload),
            key_version=_KEY_VERSION,
            consent_version=consent_version,
        )

    async def get_profile(
        self, chat_id: int, user_id: int
    ) -> dict[str, object] | None:
        """Возвращает расшифрованный payload или None если участник не найден."""
        row = await self._repo.get(self._hmac(chat_id), self._hmac(user_id))
        if row is None:
            return None
        return self._decrypt_json(str(row["encrypted_payload"]))

    async def revoke(self, chat_id: int, user_id: int) -> None:
        """Удаляет профиль участника."""
        await self._repo.remove(self._hmac(chat_id), self._hmac(user_id))

    # --- приватные helpers ---

    def _hmac(self, value: int | str) -> str:
        return hmac.new(self._hmac_key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()

    def _encrypt(self, value: int | str) -> str:
        return self._fernet.encrypt(str(value).encode("utf-8")).decode("utf-8")

    def _encrypt_json(self, data: dict[str, object]) -> str:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        return self._fernet.encrypt(raw).decode("utf-8")

    def _decrypt_json(self, token: str) -> dict[str, object]:
        result: dict[str, object] = json.loads(
            self._fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        )
        return result
