from __future__ import annotations

import base64
import hashlib
import hmac
import html
from dataclasses import dataclass

from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PIVO_FALLBACK_MENTIONS = "Господа дегенераты"
PIVO_PRIVACY_MESSAGE = """Для /pivo я храню только тех пользователей, которые сами
включились командой /pivo_on.

Эти данные нужны только для того, чтобы упоминать тебя в шуточном созыве в Discord.

Чувствительная информация хранится в защищённом виде.

Удалить себя из списка можно в любой момент командой:
/pivo_off"""


@dataclass(slots=True)
class PivoMember:
    encrypted_user_id: str
    encrypted_username: str
    encrypted_display_name: str
    # HMAC подписчика — то же значение, что лежит ключом строки в базе. Нужен,
    # чтобы исключить вызывающего из списка, не расшифровывая его user_id.
    user_hash: str = ""


# Доменная метка ключа шифрования подписок. Отдельная метка — отдельный ключ:
# компрометация одного назначения не должна давать ключа к другому (ср.
# ``logging:chat_id`` в app/log_masking.py).
_FERNET_DOMAIN = b"pivo:fernet"


def _derive_fernet_key(secret: str) -> bytes:
    """Ключ Fernet из секрета конфигурации через HKDF-SHA256."""
    derived = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=_FERNET_DOMAIN,
    ).derive(secret.encode("utf-8"))
    return base64.urlsafe_b64encode(derived)


def _legacy_fernet_key(secret: str) -> bytes:
    """Прежняя схема: ключ — просто SHA-256 от секрета.

    Оставлена только для чтения уже сохранённых строк. Одиночное хеширование
    не создаёт барьера для перебора: секрет — парольная фраза из конфигурации,
    и подбор по украденной базе идёт со скоростью хеширования.
    """
    return base64.urlsafe_b64encode(
        hashlib.sha256(secret.encode("utf-8")).digest()
    )


class PivoSecurity:
    """HMAC-анонимизация ключей и шифрование персональных полей подписчика.

    Шифрование идёт набором ключей: первый (HKDF) шифрует, второй (прежняя
    схема) только читает. Поэтому смена схемы не требует ни остановки бота, ни
    единовременной миграции таблицы — строки переходят на новый ключ при
    очередной перезаписи профиля (``/pivo_on``, суточное обновление).
    """

    def __init__(self, hmac_secret: str, encryption_secret: str) -> None:
        self._hmac_secret = hmac_secret.encode("utf-8")
        self._fernet = MultiFernet(
            [
                Fernet(_derive_fernet_key(encryption_secret)),
                Fernet(_legacy_fernet_key(encryption_secret)),
            ]
        )

    def hmac_value(self, value: int | str) -> str:
        payload = str(value).encode("utf-8")
        return hmac.new(self._hmac_secret, payload, hashlib.sha256).hexdigest()

    def encrypt_value(self, value: int | str) -> str:
        return self._fernet.encrypt(str(value).encode("utf-8")).decode("utf-8")

    def decrypt_value(self, value: str) -> str:
        return self._fernet.decrypt(value.encode("utf-8")).decode("utf-8")


def normalize_username(username: str | None) -> str:
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


# How a subscriber ended up mentioned. Kept as plain strings so they can go
# straight into an aggregate log line without carrying anything personal.
MENTION_BY_ID = "by_id"
MENTION_BY_USERNAME = "by_username"
MENTION_SKIPPED = "skipped"
# Не путь упоминания, а счётчик подписчиков, не поместившихся в предел fanout.
# Живёт в том же агрегате: тому, кто читает лог или отчёт, важно видеть, что
# упомянуты не все, ровно там же, где он видит остальные числа.
MENTION_TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class PivoMentionResult:
    """One subscriber's mention plus how it was built.

    ``text`` is the HTML fragment that goes into the message; ``path`` and
    ``reason`` exist so ``/pivo`` can log an aggregate and ``/pivo_check``
    can explain a single subscriber, neither of which may leak the
    decrypted values themselves.
    """

    text: str
    path: str
    reason: str
    # Подпись без разметки — для диагностики, которая не должна пинговать.
    label: str = ""


def _usable_user_id(raw: str) -> str:
    """Return the user id if it is shaped like one, else an empty string.

    Telegram user ids are positive integers. Anything else in the column is
    corrupt rather than merely unusual, so it must not be interpolated into
    a ``tg://user?id=`` link.
    """
    candidate = raw.strip()
    return candidate if candidate.isdigit() else ""


def build_pivo_mention(
    member: PivoMember,
    security: PivoSecurity,
    *,
    mention_by_id: bool = True,
) -> PivoMentionResult:
    # Bots are rejected up-front in the `/pivo_on` handler, so anything
    # that reached the table is a regular user. The previous explicit
    # `is_bot` short-circuit was redundant and is gone with that column.
    username = normalize_username(security.decrypt_value(member.encrypted_username))
    user_id = _usable_user_id(security.decrypt_value(member.encrypted_user_id))

    if mention_by_id and user_id:
        # O2: the href carries the id-based mention (the server turns it into
        # a `text_mention` entity, which needs no name matching); the label
        # keeps the plain `@username` that used to be the whole mechanism.
        # So the worst case of this path equals the old behaviour instead of
        # replacing it — see design.md, D1.
        display_name = (
            security.decrypt_value(member.encrypted_display_name) or "участник"
        )
        label = username or display_name
        safe_user_id = html.escape(user_id, quote=True)
        safe_label = html.escape(label, quote=False)
        return PivoMentionResult(
            text=f'<a href="tg://user?id={safe_user_id}">{safe_label}</a>',
            path=MENTION_BY_ID,
            reason="ссылка по user_id",
            label=label,
        )

    if username:
        return PivoMentionResult(
            text=html.escape(username, quote=False),
            path=MENTION_BY_USERNAME,
            reason="ручка выключена" if not mention_by_id else "нет user_id",
            label=username,
        )

    if user_id:
        # Knob off and no username: the id link with a display name is what
        # this subscriber always got, so keep it rather than dropping them.
        display_name = (
            security.decrypt_value(member.encrypted_display_name) or "участник"
        )
        safe_user_id = html.escape(user_id, quote=True)
        safe_display_name = html.escape(display_name, quote=False)
        return PivoMentionResult(
            text=f'<a href="tg://user?id={safe_user_id}">{safe_display_name}</a>',
            path=MENTION_BY_ID,
            reason="нет ника, ссылка по user_id",
            label=display_name,
        )

    return PivoMentionResult(text="", path=MENTION_SKIPPED, reason="нет user_id и ника")


def collect_pivo_mentions(
    members: list[PivoMember],
    caller_user_id: int,
    security: PivoSecurity,
    *,
    mention_by_id: bool = True,
) -> list[PivoMentionResult]:
    """Упоминания подписчиков чата, кроме самого вызывающего.

    Вызывающий отсеивается по хранимому ``user_hash`` — тому самому значению,
    которым строка ключуется в базе. Расшифровка ``encrypted_user_id`` ради
    этого сравнения не нужна: она делается только там, где идентификатор
    действительно идёт в ссылку упоминания.
    """
    caller_hash = security.hmac_value(caller_user_id)
    mentions: list[PivoMentionResult] = []

    for member in members:
        if member.user_hash and hmac.compare_digest(member.user_hash, caller_hash):
            continue
        if not member.user_hash:
            # Совместимость с вызывающими, которые не передают хеш: тогда
            # остаётся прежний путь через расшифровку.
            user_id = security.decrypt_value(member.encrypted_user_id)
            if hmac.compare_digest(security.hmac_value(user_id), caller_hash):
                continue
        mentions.append(
            build_pivo_mention(member, security, mention_by_id=mention_by_id)
        )
    return mentions


def count_mention_paths(results: list[PivoMentionResult]) -> dict[str, int]:
    """Aggregate mention paths for logging — numbers only, nothing personal."""
    counts = {MENTION_BY_ID: 0, MENTION_BY_USERNAME: 0, MENTION_SKIPPED: 0}
    for result in results:
        counts[result.path] += 1
    return counts
