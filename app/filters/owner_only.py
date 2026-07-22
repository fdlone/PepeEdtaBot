from __future__ import annotations

from aiogram.filters import Filter
from aiogram.types import Message

from app.config.settings import Settings


async def is_owner(message: Message, settings: Settings) -> bool:
    if message.from_user is None or settings.owner_id is None:
        return False
    return message.from_user.id == settings.owner_id


class OwnerOnly(Filter):
    """Restricts a command to ``OWNER_ID``, unlike ``AdminOrOwner`` no chat
    admin ever qualifies. Used for commands whose effect is process-global
    (see O5 in docs/OPEN.md): a chat admin from any group the bot is added
    to must not be able to trigger them.
    """

    async def __call__(self, message: Message, settings: Settings) -> bool:
        return await is_owner(message, settings)
