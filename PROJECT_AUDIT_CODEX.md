# PROJECT_AUDIT_CODEX

Дата актуализации: 2026-05-09  
Текущая ветка: `codex/pivo-daily-quota`  
Базовый документ аудита: `PROJECT_AUDIT.md`

## Current status

`PROJECT_AUDIT.md` остается главным общим аудитом проекта. По нему рефакторинговая ветка `refactor/structure` уже признана готовой к merge: P0 закрыты, P1-блокеры B1/B2/B3 закрыты, оставшийся долг относится к P2/P3.

Этот файл используется как рабочий tracker Codex по текущей ветке `codex/pivo-daily-quota`. Старые выводы по `refactor/structure` больше не дублируются здесь; актуальные детали по ним нужно смотреть в `PROJECT_AUDIT.md`.

## Current branch audit: daily quota for /pivo

### Implemented

- Добавлена таблица `pivo_daily_usage` через миграцию `006_pivo_daily_usage.sql`.
- Daily quota хранится отдельно от `pivo_chat_members` по ключу `(chat_hash, user_hash, usage_day)`.
- Обычный пользователь может вызвать `/pivo` 1 раз в сутки.
- Admin/owner может вызвать `/pivo` 3 раза в сутки.
- Runtime cooldown снова подключен через `ThrottlingMiddleware`.
- `/pivo` и `/clear` ограничены cooldown 3600 секунд на `(chat_id, user_id, command)`.
- Если `/pivo` quota уже списана, но сборка или отправка ответа падает, quota откатывается через refund.
- Для `pivo_daily_usage` добавлен retention cleanup: строки старше 7 дней удаляются при `Database.init()`.
- Daily quota reset остается по UTC. Это осознанное продуктовое решение и не считается текущей проблемой.
- `main.py` wiring вынесен в `configure_dispatcher()` и покрыт smoke-test: dispatcher data, routers и middleware проверяются без запуска Telegram polling.
- `/pivo` subscribe → call quota → unsubscribe flow покрыт сервисным тестом на реальной временной SQLite базе.

### Security/privacy notes

- В `pivo_daily_usage` не хранятся raw Telegram IDs.
- Используются `chat_hash` и `user_hash`, как и в остальном `/pivo` контуре.
- `usage_day` хранит только дату использования quota.
- Retention на 7 дней ограничивает долгосрочный рост таблицы и снижает объем вспомогательных usage-данных.

### Resolved during this session

- Закрыто: потерянный runtime cooldown после удаления `ThrottlingMiddleware` из `main.py`.
- Закрыто: риск потери дневной quota при ошибке после списания и до успешной отправки `/pivo`.
- Закрыто: бесконечный рост `pivo_daily_usage` без cleanup/retention.
- Принято: UTC reset для daily quota остается без изменений.

## Changed files in current work

- `main.py`
- `db.py`
- `app/handlers/pivo.py`
- `app/repositories/pivo_usage_repo.py`
- `app/services/pivo_service.py`
- `tests/test_db_logic.py`
- `tests/test_handlers.py`
- `tests/test_main.py`
- `tests/test_pivo.py`
- `PROJECT_AUDIT_CODEX.md`

## Tests/checks run

- `python -m unittest tests.test_pivo tests.test_handlers tests.test_db_logic tests.test_main -v` — 53 tests OK.
- `python -m unittest tests.test_db_logic -v` — 13 tests OK.
- `python -m unittest tests.test_main tests.test_pivo -v` — 17 tests OK.
- `python -m unittest discover tests -v` — 198 tests OK.
- `python -m ruff check app/ tests/` — passed.
- `python -m mypy app/` — passed, 30 source files.

## Not run / limitations

- Live Telegram smoke test не выполнялся.
- Docker build не выполнялся.
- GitHub Actions на удаленном runner после текущих локальных изменений не проверялся.

## Remaining work

- По текущему `/pivo` daily quota flow локальных P1/P2-блокеров не осталось.
- Локальные smoke/E2E-like проверки, рекомендованные для текущей ветки, добавлены.
- Для полного production confidence желательно отдельно выполнить live smoke в Telegram.
- При необходимости перед merge можно запустить Docker build и дождаться удаленного CI.
- Общий P2/P3 техдолг проекта остается в `PROJECT_AUDIT.md`, разделы 14-15.
