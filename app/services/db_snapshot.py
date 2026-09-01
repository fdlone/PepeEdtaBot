"""Доставка снимка БД владельцу в личку (спека db-snapshot-delivery).

Снимок снимается через ``VACUUM INTO``: в WAL-режиме голое копирование файла
может отдать рваное состояние, а VACUUM INTO даёт консистентный однофайловый
снимок без -wal/-shm. Отправка — тем же ЛС-каналом, что отчёты /pivo и алерты
обслуживания; любой отказ здесь стоит предупреждения в логе, но не запуска.
"""
from __future__ import annotations

import asyncio
import gzip
import logging
import os
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aiogram.types import FSInputFile

if TYPE_CHECKING:
    from aiogram import Bot

    from app.config.settings import Settings

logger = logging.getLogger("chat_markov")

# Лимит Bot API на sendDocument. Файл крупнее заменяется текстовым
# предупреждением: молча исчезнувший бэкап хуже сообщения о размере.
TELEGRAM_DOCUMENT_LIMIT_BYTES = 50 * 1024 * 1024


async def make_snapshot(db_path: str) -> str | None:
    """Снять сжатый снимок БД во временный каталог.

    Возвращает путь к ``markov-<UTC-метка>.db.gz`` или ``None``, если файла БД
    ещё нет (первый запуск — не ошибка). Временный каталог удаляет
    ``send_packed_snapshot``; при исключении здесь он вычищается сразу.
    """
    if not os.path.exists(db_path):
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    tmp_dir = tempfile.mkdtemp(prefix="pepe_db_snapshot_")
    packed_path = os.path.join(tmp_dir, f"markov-{stamp}.db.gz")

    def _work() -> None:
        raw_path = os.path.join(tmp_dir, "snapshot.db")
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("VACUUM INTO ?", (raw_path,))
        finally:
            conn.close()
        with open(raw_path, "rb") as src, gzip.open(packed_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.remove(raw_path)

    try:
        await asyncio.to_thread(_work)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return packed_path


async def send_packed_snapshot(bot: Bot, chat_id: int, packed_path: str) -> None:
    """Отправить снятый снимок документом; всегда убрать временный каталог.

    Файл крупнее лимита Telegram заменяется предупреждением с фактическим
    размером. Ошибки отправки пробрасываются — решают вызывающие: старт
    гасит их в warning, хендлер команды отвечает владельцу.
    """
    try:
        size = os.path.getsize(packed_path)
        if size > TELEGRAM_DOCUMENT_LIMIT_BYTES:
            await bot.send_message(
                chat_id,
                "Снимок БД снят, но не отправлен: "
                f"{size} байт при лимите {TELEGRAM_DOCUMENT_LIMIT_BYTES}. "
                "Нужен другой канал доставки.",
            )
            return
        await bot.send_document(
            chat_id,
            FSInputFile(packed_path),
            caption="Снимок БД (gzip), снят до миграций текущего старта.",
        )
    finally:
        shutil.rmtree(os.path.dirname(packed_path), ignore_errors=True)


async def startup_snapshot(settings: Settings) -> str | None:
    """Снимок на старте, до ``db.init()``: гейты ручки и владельца, ошибки — в warning.

    До миграций — потому что это точка отката и вход правила «миграцию
    проверять на копии прода»; после init снимок отражал бы уже новую сборку.
    """
    if not settings.db_snapshot_to_owner or settings.owner_id is None:
        return None
    try:
        return await make_snapshot(settings.db_path)
    except Exception:
        logger.warning("db snapshot on startup failed", exc_info=True)
        return None


async def send_startup_snapshot(bot: Bot, owner_id: int, packed_path: str) -> None:
    """Фоновая доставка стартового снимка: не бросает, чтобы не убить задачу."""
    try:
        await send_packed_snapshot(bot, owner_id, packed_path)
    except Exception:
        # Типичный случай — владелец не начинал диалог с ботом (то же
        # ограничение, что у отчётов /pivo): предупреждаем и живём дальше.
        logger.warning("db snapshot delivery failed", exc_info=True)
