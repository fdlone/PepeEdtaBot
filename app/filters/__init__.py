from aiogram import F
from aiogram.enums import ChatType

from .admin_or_owner import AdminOrOwner, is_admin_or_owner
from .owner_only import OwnerOnly, is_owner

# Пропускает только группы и супергруппы; личка и каналы отсекаются.
GROUP_ONLY = F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP})

# Пропускает только личку: для команд, чей ответ не должен иметь пути в группу
# (например, документ с корпусом сообщений в /db_snapshot).
PRIVATE_ONLY = F.chat.type == ChatType.PRIVATE

__all__ = [
    "GROUP_ONLY",
    "PRIVATE_ONLY",
    "AdminOrOwner",
    "OwnerOnly",
    "is_admin_or_owner",
    "is_owner",
]
