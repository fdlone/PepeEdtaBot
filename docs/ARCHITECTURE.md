# Архитектура Проекта

## Обзор
- `main.py`: runtime бота, хендлеры, политика триггеров, runtime-команды конфигурации.
- `db.py`: схема SQLite, миграция, атомарные записи, статистика/запросы.
- `markov.py`: токенизация и генерация variable-order (`3 -> 2 -> 1`).
- `text_utils.py`: пайплайн очистки текста.
- `settings.py`: загрузка и валидация `.env`.

## Модель Данных
- `messages`: сырые входящие сообщения для аудита/отладки.
- `starts3` / `transitions3`: основная триграммная модель.
- `starts` / `transitions`: биграммный fallback-слой.
- `transitions1`: униграммный fallback-слой.

## Runtime-Управление
Все изменения через runtime-команды действуют только в текущем процессе и сбрасываются после перезапуска.
Права: `OWNER_ID` или администратор чата.
- `/set reply_probability ...`
- `/set min_cooldown_sec ...`
- `/set min_tokens_for_model ...`
- `/set max_reply_chars ...`
- `/set normalize_lower ...`
- `/set typing_min_ms ...`
- `/set typing_max_ms ...`
- `/set randomness_strength ...`
- `/set repetition_penalty_strength ...`
- `/set markov_order ...`
- `/set enable_backoff ...`
- `/set backoff_min_order ...`
- `/set use_reply_context ...`
- `/set reply_context_max_tokens ...`
- `/set reply_context_last_tokens ...`
- `/set reply_context_bias ...`
- `/set reply_context_start_bias ...`
- `/set reply_context_only_for_replies ...`
- `/set reply_context_include_current_message ...`

## Контекст Reply
- Reply-контекст не требует отдельной БД и не меняет схему SQLite.
- Текст `reply_to_message` очищается и токенизируется тем же пайплайном, что и обычные сообщения.
- Контекст влияет на выбор старта генерации и на веса переходов между токенами.
- Если контекст не подходит к накопленной модели чата, генератор откатывается к базовой Markov-логике.

## Безопасность
- SQL-запросы параметризованы.
- `.env` игнорируется git.
- Защита от дубликатов не даёт отправлять дословные копии сообщений.
