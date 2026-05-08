from __future__ import annotations

import base64
import hashlib
import hmac
import html
import random
from dataclasses import dataclass
from typing import Optional

from cryptography.fernet import Fernet

from pivo_templates import PIVO_TEMPLATES


PIVO_FALLBACK_MENTIONS = "Господа дегенераты"
PIVO_PRIVACY_MESSAGE = """Для /pivo я храню только тех пользователей, которые сами включились командой /pivo_on.

Эти данные нужны только для того, чтобы упоминать тебя в шуточном созыве в Discord.

Чувствительная информация хранится в защищённом виде.

Удалить себя из списка можно в любой момент командой:
/pivo_off"""


@dataclass(slots=True)
class PivoMember:
    encrypted_user_id: str
    encrypted_username: str
    encrypted_display_name: str
    is_bot: bool


class PivoSecurity:
    def __init__(self, hmac_secret: str, encryption_secret: str) -> None:
        self._hmac_secret = hmac_secret.encode("utf-8")
        key = base64.urlsafe_b64encode(
            hashlib.sha256(encryption_secret.encode("utf-8")).digest()
        )
        self._fernet = Fernet(key)

    def hmac_value(self, value: int | str) -> str:
        payload = str(value).encode("utf-8")
        return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()

    def encrypt_value(self, value: int | str) -> str:
        return self._fernet.encrypt(str(value).encode("utf-8")).decode("utf-8")

    def decrypt_value(self, value: str) -> str:
        return self._fernet.decrypt(value.encode("utf-8")).decode("utf-8")


def normalize_username(username: Optional[str]) -> str:
    if not username:
        return ""

    normalized = username.strip()
    if not normalized:
        return ""

    if normalized.startswith("@"):
        return normalized
    return f"@{normalized}"


def display_name_from_user(user: object) -> str:
    full_name = str(getattr(user, "full_name", "") or "").strip()
    if full_name:
        return full_name

    first_name = str(getattr(user, "first_name", "") or "").strip()
    last_name = str(getattr(user, "last_name", "") or "").strip()
    fallback = " ".join(part for part in (first_name, last_name) if part).strip()
    return fallback or "участник"


def build_pivo_mention(member: PivoMember, security: PivoSecurity) -> str:
    if member.is_bot:
        return ""

    username = normalize_username(security.decrypt_value(member.encrypted_username))
    if username:
        return html.escape(username, quote=False)

    user_id = security.decrypt_value(member.encrypted_user_id)
    display_name = security.decrypt_value(member.encrypted_display_name) or "участник"
    safe_user_id = html.escape(user_id, quote=True)
    safe_display_name = html.escape(display_name, quote=False)
    return f'<a href="tg://user?id={safe_user_id}">{safe_display_name}</a>'


def collect_pivo_mentions(
    members: list[PivoMember],
    caller_user_id: int,
    security: PivoSecurity,
) -> list[str]:
    caller_hash = security.hmac_value(caller_user_id)
    mentions: list[str] = []

    for member in members:
        user_id = security.decrypt_value(member.encrypted_user_id)
        if security.hmac_value(user_id) == caller_hash:
            continue
        mention = build_pivo_mention(member, security)
        if mention:
            mentions.append(mention)
    return mentions


def build_pivo_mentions(
    members: list[PivoMember],
    caller_user_id: int,
    security: PivoSecurity,
) -> str:
    mentions = collect_pivo_mentions(members, caller_user_id, security)

    if not mentions:
        return PIVO_FALLBACK_MENTIONS
    return " ".join(mentions)


def get_random_pivo_message(mentions: str) -> str:
    template = random.choice(PIVO_TEMPLATES)
    return template.format(mentions=mentions).strip()
