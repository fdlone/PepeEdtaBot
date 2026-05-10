# PROJECT_AUDIT_CODEX

Дата аудита: 2026-05-10  
Аудитор: Codex  
Область: фактическое состояние репозитория `E:\test\PepeEdtaBot` на ветке `main` без изменения `PROJECT_AUDIT.md`

## 1. Executive summary

Проект в целом находится в хорошем состоянии. Архитектура уже не монолитная: слои `handlers / services / repositories / middlewares / migrations` действительно выделены, мигратор версионирован, тестовый контур широкий, `ruff` и `mypy` зелёные. Критических дефектов уровня P0/P1 по коду я не нашёл.

Главная оставшаяся проблема не в runtime-коде, а в консистентности продукта и документации:

1. `/pivo`-команды не ограничены групповыми чатами, хотя проект и документация описывают их как групповой сценарий.
2. `README.md` заметно отстал от реального security/privacy-устройства `/pivo` и логирования.
3. `docs/ARCHITECTURE.md` тоже частично устарел по тестовой статистике.
4. Quickstart в `README.md` ведёт на незалоченные зависимости, тогда как CI и Docker живут на `requirements.lock`.

Итоговая оценка:

- Код/архитектура: `хорошо`
- Тестовое покрытие базовых сценариев: `хорошо`
- Документация: `средне`
- Операционная консистентность: `выше среднего`, но с хвостом по `/pivo`-scope и docs drift

## 2. Что было проверено

### Репозиторий и состояние

- Ветка: `main`
- `git status`: рабочее дерево чистое на момент начала аудита
- В репозитории присутствуют локальные runtime-артефакты:
  - `data/markov.db`
  - `data/markov.db.backup-before-migration-007`
  - корневой `markov.db`
- `.env` локально указывает `DB_PATH=data/markov.db`

### Фактические проверки

Запущено через системный Python 3.12:

- `python -m unittest discover tests`  
  Результат: `Ran 199 tests ... OK`
- `python -m ruff check app/ tests/`  
  Результат: `All checks passed!`
- `python -m mypy app/`  
  Результат: `Success: no issues found in 27 source files`

### Что дополнительно сверено вручную

- `main.py`, `db.py`, `settings.py`, `config_registry.py`, `runtime_config.py`, `runtime_state.py`
- `app/handlers/*`
- `app/services/pivo_service.py`
- `app/repositories/chat_members_repo.py`, `pivo_usage_repo.py`
- `app/infrastructure/migrator.py`
- `pivo.py`, `text_utils.py`
- `README.md`, `docs/ARCHITECTURE.md`, `PROJECT_AUDIT.md`
- `Dockerfile`, `docker-entrypoint.sh`, `compose.yaml`
- `tests/test_main.py`, `tests/test_migrator.py`, `tests/test_log_masking.py`

## 3. Подтверждённые сильные стороны

### Архитектура

- `main.py` действительно выполняет роль compose root, а не god-file.
- DI через `Dispatcher` собран прозрачно и без лишних контейнеров.
- Runtime-изменяемые настройки сведены в единый `config_registry.py`; это реальное улучшение по сравнению с дублированием в нескольких местах.

### База и миграции

- Миграции действительно версионированы и применяются через `schema_migrations`.
- `.sql`-миграции реально обёрнуты в `BEGIN; ... COMMIT;` через `executescript`, то есть заявленная атомарность не фиктивна.
- В `tests/test_migrator.py` есть не только happy-path, но и проверки legacy-схем и rollback на half-failed `.sql`.

### Privacy / security

- `messages.text` в текущей схеме не хранится, используется только `normalized_text`.
- `author_id` анонимизируется.
- `/pivo`-данные хранятся через `chat_hash` / `user_hash` и зашифрованные payload-поля.
- `chat_id` для learning-логов реально маскируется через `app/log_masking.py`.

### Качество инженерной базы

- `199` тестов действительно проходят локально.
- `ruff` clean.
- `mypy app/` clean.
- CI-конфиг соответствует заявленному базовому набору проверок.

## 4. Findings

### F1. `/pivo`-контур не ограничен group/supergroup, хотя по смыслу проект и команды описаны как групповые

Серьёзность: `P3`  
Статус: `open`

#### Факт

В [`app/handlers/pivo.py`](app/handlers/pivo.py) на хендлерах `/pivo`, `/pivo_on`, `/pivo_off`, `/pivo_privacy` нет фильтра `GroupOnly()`:

- [`app/handlers/pivo.py:21`](app/handlers/pivo.py#L21)
- [`app/handlers/pivo.py:70`](app/handlers/pivo.py#L70)
- [`app/handlers/pivo.py:95`](app/handlers/pivo.py#L95)
- [`app/handlers/pivo.py:112`](app/handlers/pivo.py#L112)

При этом проект позиционируется как бот для группового чата, а `README.md` прямо ведёт пользователя в group-oriented сценарий:

- [`README.md:3`](README.md#L3)
- [`README.md:112`](README.md#L112)

#### Риск

- Пользователь может подписаться через `/pivo_on` в личке, что создаст запись в `chat_members` для приватного чата.
- `/pivo` в личке тоже будет работать в терминах квоты и сборки сообщения, хотя продуктовый смысл у этого сомнительный.
- Это поведение не выглядит осознанно документированным и не покрыто как явный контракт.

#### Оценка

Это не security-авария и не runtime-crash, но это реальная продуктовая неконсистентность между кодом и ожидаемой моделью использования.

#### Рекомендация

Если DM-сценарий не нужен:

- добавить `GroupOnly()` ко всем `/pivo*` handlers;
- добавить regression-тесты на отказ в private chat;
- синхронизировать help/README.

Если DM-сценарий нужен:

- это надо явно задокументировать как supported behavior.

### F2. `README.md` устарел по криптографии `/pivo` и privacy/logging

Серьёзность: `P3`  
Статус: `open`

#### Факт

`README.md` всё ещё утверждает, что `/pivo` опирается на HKDF-домены:

- [`README.md:9`](README.md#L9)
- [`README.md:146`](README.md#L146)

Но фактическая реализация в [`pivo.py`](pivo.py) другая:

- HMAC считается напрямую от `PIVO_HMAC_SECRET`: [`pivo.py:42`](pivo.py#L42)
- Fernet-ключ строится как `sha256(PIVO_ENCRYPTION_SECRET)`: [`pivo.py:35`](pivo.py#L35)

Также `README.md` говорит, что повышенные уровни логирования могут печатать raw `chat_id`:

- [`README.md:149`](README.md#L149)

Но текущий learning-path уже использует `mask_chat_id(...)`:

- [`app/handlers/learning.py:11`](app/handlers/learning.py#L11)
- [`app/handlers/learning.py:89`](app/handlers/learning.py#L89)
- [`app/handlers/learning.py:105`](app/handlers/learning.py#L105)
- [`app/handlers/learning.py:238`](app/handlers/learning.py#L238)

#### Риск

- Оператор читает неверное описание хранения секретов и логов.
- Документация расходится с кодом именно в security/privacy-части, а это самый нежелательный тип docs drift.

#### Рекомендация

- переписать `README.md` под фактическую модель `PivoSecurity`;
- убрать утверждение о raw `chat_id` из privacy-блока;
- оставить HKDF только там, где он реально используется: log masking.

### F3. `docs/ARCHITECTURE.md` устарел по числу тестов

Серьёзность: `P3`  
Статус: `open`

#### Факт

`docs/ARCHITECTURE.md` утверждает, что в проекте `208 unit-тестов`:

- [`docs/ARCHITECTURE.md:235`](docs/ARCHITECTURE.md#L235)

Фактический прогон дал:

- `Ran 199 tests in 27.263s`

`PROJECT_AUDIT.md` в этом месте уже актуален, а `docs/ARCHITECTURE.md` нет.

#### Риск

Небольшой, но это явный маркер того, что часть документации обновляется несинхронно.

#### Рекомендация

- синхронизировать численные метрики между `README.md`, `docs/ARCHITECTURE.md`, `PROJECT_AUDIT.md`;
- если не хочется постоянно править числа, убрать хрупкие метрики из архитектурного документа или пометить их как approximate.

### F4. Quickstart в `README.md` не воспроизводит тот же dependency set, что CI и Docker

Серьёзность: `P3`  
Статус: `open`

#### Факт

Quickstart предлагает ставить:

- [`README.md:14`](README.md#L14) → `pip install -r requirements.txt`

Но production/CI завязаны на:

- `requirements.lock`
- `requirements-dev.txt`
- [`Dockerfile`](Dockerfile)
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

#### Риск

- локальная разработка по README может идти на другой резолюции пакетов, чем CI/Docker;
- отладка «у меня локально работает / в CI нет» становится вероятнее.

#### Рекомендация

Минимум:

- в quickstart явно разделить `runtime install` и `dev install`;
- для dev по умолчанию рекомендовать `pip install -r requirements-dev.txt`.

Опционально:

- добавить отдельный reproduceable runtime-start через `requirements.lock`.

## 5. Что в `PROJECT_AUDIT.md` подтверждено, а что нет

### Подтверждено

- Ветка `main`
- `199` тестов
- `ruff` clean
- `mypy app/` clean на `27 source files`
- слоистая архитектура
- мигратор и atomic `.sql`
- `chat_members` как текущая таблица `/pivo`
- log masking через `app/log_masking.py`

### Частично / с оговорками

- Тезис «backlog пуст» я бы формально не повторял:
  - по коду критичного долга действительно не видно;
  - но docs drift и неконсистентность `/pivo`-scope означают, что маленький backlog всё же остался.

### Не подтверждено в рамках этой сессии

- live smoke в Telegram я не проводил;
- Docker build/runtime здесь не проверялся;
- удалённый GitHub Actions не перепроверялся после моего запуска локальных проверок;
- фактическое содержимое текущей SQLite runtime-базы я не ревизовал на уровне данных, только на уровне файлов и кода.

## 6. Текущее состояние файлов и структуры

Подтверждено по дереву проекта:

- `app/` содержит `27` Python-модулей
- `tests/` содержит `13` test-файлов
- `main.py` и `db.py` по масштабу соответствуют описанию рефакторинга
- `Dockerfile`, `docker-entrypoint.sh`, `compose.yaml` присутствуют
- миграции `001` ... `007` на месте

Отдельно:

- в `data/` лежит живая БД и backup перед migration 007;
- это нормально для локального runtime-state, но важно помнить, что аудит кода и аудит содержимого прод-данных не одно и то же.

## 7. Приоритет действий

### Рекомендую сделать в первую очередь

1. Определиться, должен ли `/pivo` работать в личке.
2. После решения либо добавить `GroupOnly()`, либо явно задокументировать DM-support.
3. Синхронизировать `README.md` с текущей security/privacy-моделью.
4. Обновить `docs/ARCHITECTURE.md` по счётчикам тестов.
5. Подправить quickstart на dependency parity с CI/Docker.

### Что можно не трогать срочно

- реестр runtime-настроек;
- мигратор;
- базовую `/pivo`-quota-логику;
- log masking;
- test/ruff/mypy контур.

## 8. Вердикт

Проект выглядит живым, поддерживаемым и инженерно заметно более зрелым, чем типичный «бот на одном файле». Основные прошлые риски действительно закрыты. На текущем состоянии я не вижу причин считать проект нестабильным или опасным для продолжения разработки.

Но говорить, что аудит полностью закрыт и больше ничего не осталось, пока рано. Оставшиеся проблемы небольшие, но реальные:

- одна продуктовая неконсистентность в `/pivo`-scope;
- несколько явных разъездов документации с кодом;
- слабая воспроизводимость quickstart относительно CI/prod.

Если эти пункты закрыть, проект можно считать действительно аккуратно приведённым в порядок.
