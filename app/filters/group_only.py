from __future__ import annotations

from aiogram.enums import ChatType
from aiogram.filters import Filter
from aiogram.types import Message


class GroupOnly(Filter):
    async def __call__(self, message: Message) -> bool:
        return message.chat.type in {ChatType.GROUP, ChatType.SUPERGROUP}
