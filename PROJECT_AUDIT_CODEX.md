# PROJECT_AUDIT_CODEX

## 2026-05-12 — Audit tasklist remediation PR 1

Scope:
- AUD-009: upgrade `cryptography` to a fixed release and regenerate `requirements.lock`.
- AUD-001: add `/pivo` explicit mention limit.
- AUD-001: add `/pivo` subscriber fanout limit.
- CA-F1 / AUD-005: verify `GroupOnly()` on all `/pivo*` handlers in `main`.
- AUD-005 / CA-F7: mirror local ignore patterns into `.dockerignore`.
- CA-F2: keep README security/privacy wording aligned with current `/pivo` implementation.

Implemented:
- Added `PIVO_EXPLICIT_MENTIONS_LIMIT` and `PIVO_SUBSCRIBER_FANOUT_LIMIT` to config.
- Enforced explicit mention and subscriber fanout limits inside `PivoService`.
- Updated `/pivo` handler flow to surface limit errors before quota consumption.
- Updated `requirements.txt` and `requirements.lock` to `cryptography==46.0.7`.
- Mirrored `db_prod_copy/`, `.test_tmp/`, and `Screenshot_*.jpg` into `.dockerignore`.
- Added the new limit env vars to `.env.example`.
- Added regression tests for explicit mentions, subscriber fanout, and handler limit rejection.
- Verified `GroupOnly()` is already applied to every `/pivo*` handler and marked that task complete.
- Updated README to mention the configurable `/pivo` limits.

Checks:
- `.\.venv\Scripts\python.exe -m unittest tests.test_pivo tests.test_handlers -v` — passed.
- `.\.venv\Scripts\python.exe -m ruff check app/ tests/` — passed.
- `.\.venv\Scripts\python.exe -m mypy app/` — passed, 29 source files.
- `.\.venv\Scripts\python.exe -m unittest discover tests -v` — passed, 252 tests.
- `.\.venv\Scripts\python.exe -m pip check` — passed.
- `.\.venv\Scripts\python.exe -m bandit -r app main.py db.py settings.py pivo.py markov.py -x tests` — completed with 47 Low findings, 0 Medium, 0 High. Existing Low findings remain in `db.py`, `markov.py`, `app/handlers/_helpers.py`, `app/services/learning_service.py`, `app/services/pivo_message_builder.py`, and `app/handlers/learning.py`.
- `.\.venv\Scripts\safety.exe check -r requirements.lock` — no known vulnerabilities found.
- `.\.venv\Scripts\python.exe -X utf8 -m pip_audit -r requirements.lock --format json` — no vulnerabilities reported.

Deferred:
- AUD-004 runbook items for log rotation, WAL checkpointing, and backup/restore were left for a later PR because they are larger operational work and not a safe add-on to the current security/dependency batch.

## 2026-05-12 — Security and stability audit report

Mode: audit plus dev-tooling update. Production bot code, tests, runtime
configs, and deployment files were not changed. Dev-only security tools were
added to `requirements-dev.txt`.

Created:
- `PROJECT_SECURITY_STABILITY_AUDIT.md`

Changed:
- `requirements-dev.txt` — added `bandit`, `pip-audit`, and `safety`.

Scope:
- code security;
- Telegram bot abuse/security;
- secrets/configuration;
- CPU/RAM/disk/network load risks;
- long-running process stability;
- database/storage;
- logging/observability;
- deployment/operations;
- tests and static checks.

Summary:
- Overall risk: `MEDIUM` for the current controlled small-group deployment;
  `HIGH` before expanding to larger or less trusted chats.
- No tracked `.env`, Telegram token, SQLite DB, WAL/SHM files, local
  `db_prod_copy/`, screenshots, or obvious hardcoded production secrets were
  found.
- Top finding: `/pivo` explicit mentions are not capped, so a regular group
  user can create several high-fanout mention messages per day.
- Dependency finding: Safety reports 3 vulnerabilities in locked
  `cryptography==45.0.7` (CVE-2026-34073, CVE-2026-26007,
  CVE-2026-39892). Recommended next fix: upgrade `cryptography`, regenerate
  `requirements.lock`, and rerun the full check set.
- Other important findings: unbounded runtime dictionaries, full-message
  prefix-cache rebuilds for novelty checks, unbounded SQLite model/WAL growth,
  weaker `.dockerignore` compared with `.gitignore`, weak production
  observability/log rotation guidance.

Checks:
- `.\.venv\Scripts\python.exe --version` — Python 3.14.0.
- `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` —
  installed dev security tools successfully.
- `.\.venv\Scripts\python.exe -m unittest discover tests -v` — 244 tests OK.
- `.\.venv\Scripts\python.exe -m ruff check app/ tests/` — clean.
- `.\.venv\Scripts\python.exe -m mypy app/` — clean, 29 source files.
- `.\.venv\Scripts\python.exe -m pip check` — no broken requirements.
- `.\.venv\Scripts\python.exe -m bandit -r app main.py db.py settings.py pivo.py markov.py -x tests`
  — completed with 47 Low findings, 0 Medium, 0 High. Findings are
  `assert` usage, non-cryptographic `random`, and intentional `except/pass`
  around Telegram chat action.
- `.\.venv\Scripts\safety.exe check -r requirements.lock` — completed with
  3 vulnerabilities in `cryptography==45.0.7`.
- `.\.venv\Scripts\python.exe -m pip list --outdated` — completed; several
  packages have newer releases, including `aiogram`, `cryptography`, and
  `pydantic`.

Not run:
- `pip-audit` verdict — tool is installed, but the check did not complete in
  this environment. Attempts against PyPI timed out after 300 seconds; OSV
  backend timed out after 180 seconds. Earlier attempts also exposed local
  Windows cache/encoding issues, worked around with local cache and
  `PYTHONUTF8=1`.
- Docker build — Docker CLI is not installed on this machine.
- GitHub PR creation through `gh` — GitHub CLI is not installed on this
  machine.

Workflow note:
- The audit branch `audit-security-stability-review` was created from the
  currently checked out `fix-short-reply-policy` branch, not directly from
  `origin/main`. Choose the PR base carefully to avoid mixing the audit report
  with unrelated branch changes.

## 2026-05-11 — Action plan execution log

### AUD-001 deferred by workflow decision

User clarified that `D:\test\PepeEdtaBot` is a local working/test checkout, not
the production Docker build environment. Local Docker build from this folder is
not needed and is not possible on the current machine.

Decision:
- Do not change `.dockerignore` now.
- Do not delete, move, or rename local working artifacts.
- Keep the finding documented as relevant only if local Docker builds from this
  workspace become part of the workflow later.

Files changed:
- `PROJECT_AUDIT_ACTION_PLAN.md`
- `PROJECT_AUDIT_CODEX.md`

### AUD-002 completed by explicit local ignore policy

User chose Option B for `db_prod_copy/`: keep the folder in the local
working/test checkout, do not delete or move its contents, and make the local
ignore policy explicit.

Change:
- Added `db_prod_copy/` to `.gitignore`.
- Marked `AUD-002` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

No data contents were inspected. No files in `db_prod_copy/` were deleted,
moved, renamed, or added to git.

### AUD-003 left as manual local secret follow-up

User chose Option A for local `.env`: report only, do not modify the file.

Decision:
- Keep `.env` unchanged because it is a secret-bearing local file.
- Track the item as `Needs decision` in `PROJECT_AUDIT_ACTION_PLAN.md`.
- Missing keys were considered by name only; secret values were not printed,
  copied, or changed.

Update:
- Local `PIVO_HMAC_SECRET` and `PIVO_ENCRYPTION_SECRET` were later generated
  and inserted into the git-ignored `.env` without printing their values.
- User added a new `BOT_TOKEN`.
- `load_settings()` now succeeds with `DB_PATH=markov.db`, `BOT_TOKEN`,
  `OWNER_ID`, and `/pivo` secrets set.
- Marked `AUD-003` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

### AUD-004 completed by syncing existing `.venv`

User chose Option B for local `.venv`: install dev dependencies into the
existing environment instead of recreating it.

Command:
- `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`

Notes:
- The first sandboxed attempt timed out.
- The approved rerun completed successfully.
- Runtime packages were synchronized to `requirements.lock` where needed.
- `ruff 0.15.12` and `mypy 2.0.0` were installed.
- No `requirements*.txt` files were changed.

Checks after install:
- `.\.venv\Scripts\python.exe -m ruff check app/ tests/` — passed.
- `.\.venv\Scripts\python.exe -m mypy app/` — passed, 27 source files.
- `.\.venv\Scripts\python.exe -m unittest discover tests -v` — 199 tests OK.

### AUD-005 completed with group-only `/pivo` contract

User chose Option A: the bot is for group chats, and private chat command
handling is not an active product feature.

Changes:
- Added `GroupOnly()` to all `/pivo*` handlers.
- Added a router-registration regression test ensuring every `/pivo*` handler
  is protected by `GroupOnly`.
- Updated `README.md` to state that commands are intended for group chats and
  `/pivo*` commands are available only in groups/supergroups.
- Marked `AUD-005` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

Checks after change:
- `.\.venv\Scripts\python.exe -m unittest tests.test_handlers tests.test_filters -v`
  — 53 tests OK.
- `.\.venv\Scripts\python.exe -m ruff check app/ tests/` — passed.
- `.\.venv\Scripts\python.exe -m mypy app/` — passed, 27 source files.
- `.\.venv\Scripts\python.exe -m unittest discover tests -v` — 200 tests OK.

### AUD-006 completed with minimal README crypto/privacy correction

User chose Option A: minimal factual correction.

Changes:
- README stack wording now says Fernet is used for `/pivo`, while HKDF is used
  for `chat_id` log masking.
- README privacy wording now describes `/pivo` as HMAC-SHA256 under
  `PIVO_HMAC_SECRET` plus Fernet key derived as SHA-256 of
  `PIVO_ENCRYPTION_SECRET`.
- README no longer broadly warns that debug logging prints raw `chat_id`; it
  states that learning logs mask `chat_id` and new log messages should not add
  raw IDs.
- Marked `AUD-006` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

### AUD-007 completed by splitting runtime and development setup

User chose Option B.

Changes:
- README quickstart now labels `requirements.txt` as the minimal local runtime
  install path.
- README now explicitly recommends `requirements-dev.txt` for development and
  local checks because it installs `requirements.lock` and CI tools.
- No dependency files were changed.
- Marked `AUD-007` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

### AUD-008 completed by removing brittle test-count wording

User chose Option A.

Changes:
- Replaced the stale exact `208 unit-тестов` metric in
  `docs/ARCHITECTURE.md` with stable wording.
- `PROJECT_AUDIT.md` was not edited.
- Marked `AUD-008` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

### AUD-009 completed by confirming local-only `AGENTS.md`

User chose Option C: `AGENTS.md` should be local-only rather than a repository
document.

Verification:
- `git ls-files AGENTS.md` returned no tracked path.
- `git check-ignore -v AGENTS.md` confirmed `.gitignore` ignores the file.
- `git rm --cached AGENTS.md` had no tracked path to remove, so no index change
  was needed.

Result:
- No file deletion or rename was performed.
- `AGENTS.md` remains local-only in this checkout.
- Marked `AUD-009` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

### AUD-010 completed with targeted screenshot ignore

User chose Option B: keep the local screenshot workflow and add a targeted
ignore pattern.

Changes:
- Added `Screenshot_*.jpg` to `.gitignore`.
- Marked `AUD-010` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

No screenshot files were deleted, moved, renamed, or added to git.

### AUD-011 completed with CI Docker build smoke

User chose Option B and asked to fix the related Docker docs drift immediately.

Changes:
- Added a separate `docker-build` job to `.github/workflows/ci.yml`.
- The job runs `docker build -t pepe-bot:latest .` on GitHub Actions.
- It does not start the bot, call Telegram APIs, or require `.env`/secrets.
- Updated README Docker commands from `pepe-edta-bot` to `pepe-bot` /
  `pepe-bot:latest`, matching `compose.yaml`.
- Marked `AUD-011` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

Local Docker build was not run because Docker build is not available/needed on
the current machine.

### AUD-012 deferred until real DB work

User chose Option A: defer the `db.py` stats/clear helper refactor until the
next real DB-related feature or bugfix.

Decision:
- Do not refactor DB code as a standalone audit cleanup.
- Keep public DB behavior unchanged.
- Marked `AUD-012` as Deferred in `PROJECT_AUDIT_ACTION_PLAN.md`.

### AUD-013 completed by extending README tests section

User chose Option B: fold local environment verification guidance into the
existing README tests section.

Changes:
- README now states that local verification should use the same dependency set
  and commands as CI.
- README notes that CI also has a Docker build smoke job without starting the
  bot.
- Marked `AUD-013` as Done in `PROJECT_AUDIT_ACTION_PLAN.md`.

## 2026-05-11 — Deep project audit and maintenance action plan

Аудитор: Codex  
Область: фактическое состояние репозитория `D:\test\PepeEdtaBot` на ветке `main` после синхронизации с `origin/main`.  
Режим: audit-only; исходный код, зависимости, runtime-конфигурация и `PROJECT_AUDIT.md` не изменялись.  
Связанный план: `PROJECT_AUDIT_ACTION_PLAN.md`.

### Executive summary

Проект находится в рабочем и в целом поддерживаемом состоянии: слои `handlers / services / repositories / infrastructure / migrations` реально выделены, миграции версионированы, есть CI, Docker, lock-файл, unit-тесты и базовая privacy-модель для `/pivo`.

Критических P0-проблем в коде или репозитории не найдено. Главные риски текущего состояния:

1. В локальной рабочей папке присутствует потенциально чувствительный runtime-state: `.env`, корневой `markov.db`, `db_prod_copy/markov.db`, `db_prod_copy/*.db-shm/*.db-wal`. Эти файлы не tracked, но требуют аккуратной ручной политики хранения.
2. `.dockerignore` не исключает локальные артефакты `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg`, поэтому они могут попасть в Docker build context.
3. `README.md`, `docs/ARCHITECTURE.md` и `PROJECT_AUDIT.md` частично расходятся с текущей реализацией и проверками.
4. `/pivo`-команды всё ещё не ограничены `GroupOnly()`, хотя продукт и документация описывают групповой сценарий.
5. Локальное `.venv` не соответствует `requirements.lock`, поэтому локальные проверки в нём не полностью воспроизводят CI/Docker.

### Checks performed

- `git status --short --branch` — ветка `main`, синхронизирована с `origin/main`; untracked: `Screenshot_1.jpg`, `Screenshot_2.jpg`.
- `git ls-files` — tracked-файлы сверены.
- `git ls-files --others --exclude-standard` — видимые неигнорируемые untracked: только `Screenshot_1.jpg`, `Screenshot_2.jpg`.
- `git check-ignore -v AGENTS.md Screenshot_1.jpg Screenshot_2.jpg db_prod_copy\markov.db .test_tmp\tmp_196kqx7` — подтверждено, что `AGENTS.md`, `*.db`, `.test_tmp/` игнорируются `.gitignore`; screenshots не игнорируются.
- `rg --files` и рекурсивный file inventory — структура проекта проверена.
- `rg -n "TODO|FIXME|HACK|XXX|password|secret|token|api[_-]?key|BOT_TOKEN|PIVO_|OWNER_ID"` — в коде не найдено TODO/FIXME/HACK и явных committed-секретов; совпадения в документации и примерах проверялись отдельно.
- `.\.venv\Scripts\python.exe -m unittest discover tests` — `Ran 199 tests ... OK`.
- `.\.venv\Scripts\python.exe -m ruff check app/ tests/` — не выполнено: в `.venv` нет `ruff`.
- `.\.venv\Scripts\python.exe -m mypy app/` — не выполнено: в `.venv` нет `mypy`.
- `.\.venv\Scripts\python.exe -m pip list` — локальный `.venv` отличается от `requirements.lock`.

### Technology and architecture snapshot

- Python Telegram bot на `aiogram v3`.
- SQLite + `aiosqlite`; WAL включается в `Database.init()`.
- `cryptography.Fernet` используется для payload `/pivo`; HMAC-SHA256 используется для `chat_hash`/`user_hash`.
- HKDF сейчас реально используется в `app/log_masking.py` для маскирования `chat_id`, но не для `/pivo` encryption/HMAC.
- `main.py` — compose root: `Settings`, `Database`, `MarkovGenerator`, `PivoService`, `LearningService`, `Dispatcher`.
- `db.py` — фасад и cross-domain транзакции; репозитории вынесены в `app/repositories`.
- Legacy/root modules всё ещё существуют (`markov.py`, `bot_messages.py`, `bot_policy.py`, `pivo.py`, `pivo_templates.py`, `settings.py`, `runtime_*`) и частично исключены из strict mypy через `pyproject.toml`.

### Status of previous `PROJECT_AUDIT_CODEX.md` findings

| Previous finding | Current status | Note |
|---|---|---|
| F1 `/pivo` не ограничен group/supergroup | Still open | `app/handlers/pivo.py` содержит handlers без `GroupOnly()` на строках `21`, `70`, `95`, `112`. |
| F2 `README.md` устарел по `/pivo` crypto/privacy/logging | Still open | `README.md` всё ещё говорит про HKDF-домены для `/pivo` и raw `chat_id` при DEBUG, что не соответствует текущему коду. |
| F3 `docs/ARCHITECTURE.md` устарел по числу тестов | Still open | Документ говорит `208 unit-тестов`, фактически локально прошло `199`. |
| F4 Quickstart ставит `requirements.txt`, а CI/Docker живут на lock | Still open | Quickstart по-прежнему предлагает `pip install -r requirements.txt`; dev/CI путь ниже в README использует `requirements-dev.txt`. |

### Discrepancies found in `PROJECT_AUDIT.md`

`PROJECT_AUDIT.md` был прочитан и намеренно не изменялся.

Подтверждено:

- общая слоистая архитектура соответствует текущей структуре;
- `199` тестов — подтверждено локальным `unittest`;
- миграции `001`...`007` присутствуют;
- `chat_members` является текущей таблицей `/pivo`;
- `messages.text` удалён, `author_id` анонимизируется;
- CI, Dockerfile, lock-файл, `.gitattributes` существуют.

Неактуально или спорно:

- В разделе `0.1` есть фраза «Все 200 тестов зелёные локально», но актуальный прогон дал `199`.
- В разделе `0` заявлен пустой P0/P1/P2/P3 backlog; текущий аудит нашёл новый backlog P1/P2/P3, прежде всего docs drift, Docker context hygiene, local env drift и `/pivo` scope.
- В Docker-сниппете раздела `9.3` указан `ENTRYPOINT ["/app/docker-entrypoint.sh"]`, а фактический Dockerfile использует `/usr/local/bin/docker-entrypoint.sh`.
- Исторические ссылки на ветку `codex/pivo-daily-quota` остаются как история, но для новых веток использовать `codex` нельзя по текущим правилам пользователя.
- Документ в целом точнее `README.md` по HKDF: он правильно уточняет, что `PivoSecurity` не использует HKDF. Однако верхний стек всё ещё формулирует `cryptography (Fernet, HKDF)` слишком широко.

### Findings — current audit

#### CA-2026-05-11-F1. `/pivo` scope не зафиксирован как явный контракт

Priority: P2  
Category: architecture / product-consistency  
Status: open

`/pivo`, `/pivo_on`, `/pivo_off`, `/pivo_privacy` зарегистрированы без `GroupOnly()`. Learning и часть admin/common команд имеют явную group-модель, а `/pivo` — нет. Это не выглядит security-инцидентом, но мешает предсказуемому развитию продукта.

Рекомендация: принять продуктовые решение: либо добавить `GroupOnly()` и тесты на private chat, либо задокументировать DM-support как поддерживаемый сценарий.

#### CA-2026-05-11-F2. Documentation drift в security/privacy-части README

Priority: P2  
Category: docs / security  
Status: open

`README.md` утверждает, что `/pivo` выводит ключи через HKDF-домены `members:hmac` / `members:encryption`. Фактически `pivo.py` использует прямой HMAC от `PIVO_HMAC_SECRET` и Fernet-ключ как `sha256(PIVO_ENCRYPTION_SECRET)`. Также README предупреждает о raw `chat_id` при DEBUG, хотя learning-path маскирует `chat_id` через `mask_chat_id(...)`.

Рекомендация: синхронизировать README с фактической реализацией и оставить HKDF только для log-masking.

#### CA-2026-05-11-F3. `docs/ARCHITECTURE.md` содержит устаревшую test metric

Priority: P3  
Category: docs  
Status: open

Документ говорит `208 unit-тестов`, но текущий локальный прогон на Python 3.14 дал `199 tests OK`. Хрупкие числовые метрики быстро устаревают.

Рекомендация: либо обновить число, либо заменить на менее хрупкую формулировку.

#### CA-2026-05-11-F4. Quickstart не воспроизводит CI/Docker dependency set

Priority: P2  
Category: dependencies / docs  
Status: open

`README.md` в быстром старте использует `pip install -r requirements.txt`, тогда как CI/Docker используют `requirements-dev.txt`/`requirements.lock`. Это допустимо для runtime-range install, но не как основной путь для новой разработки.

Рекомендация: в README явно разделить `runtime quickstart` и `development setup`; для разработки по умолчанию рекомендовать `pip install -r requirements-dev.txt`.

#### CA-2026-05-11-F5. Local `.venv` drift relative to `requirements.lock`

Priority: P2  
Category: dependencies / devops  
Status: open

`pip list` в `.venv` отличается от `requirements.lock`: например, локально `aiogram 3.25.0`, `aiohttp 3.13.3`, `python-dotenv 1.2.1`, а lock содержит `aiogram 3.27.0`, `aiohttp 3.13.5`, `python-dotenv 1.2.2`. Из-за этого `unittest` проверяет не тот же набор пакетов, что Docker/CI.

Рекомендация: пересоздать `.venv` или переустановить зависимости из `requirements-dev.txt`; добавить в docs короткую команду проверки соответствия.

#### CA-2026-05-11-F6. `ruff` и `mypy` отсутствуют в локальном `.venv`

Priority: P2  
Category: devops / dependencies  
Status: open

`python -m ruff` и `python -m mypy` не запускаются в текущем `.venv`, хотя CI их использует. Это признак неполного dev setup.

Рекомендация: установить dev-зависимости из `requirements-dev.txt` и после этого повторить `ruff`/`mypy`.

#### CA-2026-05-11-F7. Docker build context может включать локальные артефакты

Priority: P1  
Category: devops / filesystem / security  
Status: open

`.dockerignore` исключает `.env`, `.venv`, `.idea`, caches, `markov.db`, `docs`, `tests`, но не исключает `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg`. Эти файлы не tracked, но при локальном `docker build` могут быть отправлены в build context и потенциально попасть в слой `COPY . .`, если не отфильтрованы.

Рекомендация: добавить в `.dockerignore` локальные DB/backups/temp/screenshot-паттерны; перед сборкой проверить `docker build` context.

#### CA-2026-05-11-F8. Local production DB copy in workspace

Priority: P1  
Category: filesystem / security  
Status: open

В рабочей папке есть `db_prod_copy/markov.db` (~9.5 MB) плюс WAL/SHM. Файлы игнорируются за счёт `*.db`, `*.db-shm`, `*.db-wal`, но сама папка не описана в `.gitignore`/`.dockerignore` явно. По имени это копия production DB; даже после анонимизации часть данных может быть чувствительной.

Рекомендация: хранить такие копии вне repo workspace или в явно игнорируемой local-only директории; перед удалением/переносом проверить, не нужна ли копия для миграционного smoke.

#### CA-2026-05-11-F9. Root `markov.db` remains local runtime/test artifact

Priority: P2  
Category: filesystem / repository hygiene  
Status: open

Корневой `markov.db` присутствует локально и описан в README как локальная тестовая база. Он игнорируется `*.db`, но всё равно загрязняет рабочую папку и может путать новые сценарии, где runtime DB по умолчанию `data/markov.db`.

Рекомендация: либо перенести тестовые DB в `data/`/local-only storage, либо оставить как документированный local artifact; не удалять без ручного подтверждения.

#### CA-2026-05-11-F10. `Screenshot_1.jpg` and `Screenshot_2.jpg` are untracked and not ignored

Priority: P2  
Category: git / filesystem  
Status: open

`git status` показывает два untracked screenshot-файла. Они не попадают под `.gitignore`, и могут случайно попасть в commit или Docker context.

Рекомендация: определить, нужны ли они как документация/issue evidence; если нет — удалить вручную или добавить локальный ignore-паттерн для `Screenshot_*.jpg`.

#### CA-2026-05-11-F11. Local `.env` appears outdated relative to `.env.example`

Priority: P2  
Category: devops / configuration  
Status: open

Без раскрытия значений проверены только ключи: локальный `.env` содержит старые комментарии (`/seed`) и не содержит видимых новых ключей вроде `PIVO_HMAC_SECRET`, `PIVO_ENCRYPTION_SECRET`, `MAX_REPLY_TOKENS`, `REPETITION_PENALTY_STRENGTH`, `LOG_LEVEL`, `BOT_TEXT_ALIASES`. Такой `.env` может не запустить текущую версию приложения.

Рекомендация: вручную синхронизировать локальный `.env` с `.env.example`, сохранив реальные секреты.

#### CA-2026-05-11-F12. `AGENTS.md` is tracked while `.gitignore` marks it local

Priority: P3  
Category: git / repository hygiene  
Status: open

`.gitignore` содержит `AGENTS.md` как local agent instructions, но `git ls-files` показывает `AGENTS.md` как tracked. Это не runtime-баг, но правило вводит в заблуждение: новые изменения в tracked ignored-файле всё равно попадут в git status.

Рекомендация: решить, должен ли `AGENTS.md` быть частью репозитория. Если да — убрать его из `.gitignore`; если нет — отдельным решением удалить из индекса.

#### CA-2026-05-11-F13. `db.py` still has dense cross-domain SQL and repetitive stats queries

Priority: P3  
Category: architecture / code-quality  
Status: open

`db.py` уже стал фасадом, но `save_message_and_update_model`, `get_stats`, `clear_chat` всё ещё содержат много прямого SQL и повторяющихся `COUNT/SUM` запросов. Это не срочно, но расширение статистики или новой модели может быть труднее, чем в repository-style коде.

Рекомендация: при следующем архитектурном проходе вынести stats/clear helpers или добавить маленькие internal query helpers без изменения поведения.

#### CA-2026-05-11-F14. Tests are broad but mostly unit-level; no automated Docker/runtime smoke

Priority: P3  
Category: tests / devops  
Status: open

Unit suite сильная и проходит, но текущая локальная проверка не покрывает Docker build, container startup, real Telegram smoke, dependency sync и реальные secrets/env validation.

Рекомендация: добавить безопасный smoke checklist и, по возможности, CI job для Docker build без запуска бота.

### File and folder classification

Actively used:

- `main.py`, `db.py`, `markov.py`, `bot_policy.py`, `bot_messages.py`, `pivo.py`, `pivo_templates.py`, `settings.py`, `runtime_config.py`, `runtime_state.py`, `config_registry.py`, `text_utils.py`
- `app/handlers`, `app/services`, `app/repositories`, `app/filters`, `app/middlewares`, `app/infrastructure`, `app/migrations`
- `tests/`, `tests/fixtures/legacy_real_schema.sql`
- `requirements.txt`, `requirements.lock`, `requirements-dev.txt`, `pyproject.toml`
- `Dockerfile`, `docker-entrypoint.sh`, `compose.yaml`, `.github/workflows/ci.yml`
- `README.md`, `docs/ARCHITECTURE.md`, `.env.example`

Auxiliary but valid:

- `seed_db.py`, `seed_diverse.py` — local/smoke data seeders.
- `PROJECT_AUDIT.md` — main historical audit; read-only for this task.
- `PROJECT_AUDIT_CODEX.md` — working audit log.
- `AGENTS.md` — currently tracked repo instruction file, despite `.gitignore` ambiguity.

Temporary/local/legacy candidates:

- `.env` — local secret-bearing config, ignored; do not commit.
- `.venv/` — local virtualenv, ignored; currently dependency-drifted.
- `.idea/` — local IDE config, ignored.
- `.test_tmp/` — old temp directories, ignored by git but not dockerignored.
- `markov.db` — local SQLite DB, ignored.
- `db_prod_copy/` — local DB copy, ignored only by file extension; likely sensitive.
- `Screenshot_1.jpg`, `Screenshot_2.jpg` — untracked, not ignored.
- `__pycache__/` — generated cache.

### Deletion / relocation candidates

| Path | Why it looks removable or relocatable | Confidence | Check before deletion |
|---|---|---|---|
| `Screenshot_1.jpg` | Untracked screenshot, not referenced by docs/code. | Medium | Ask whether it is needed for issue evidence or documentation. |
| `Screenshot_2.jpg` | Same as above. | Medium | Ask whether it is needed for issue evidence or documentation. |
| `.test_tmp/` | Old local temp directories from February; ignored by git. | High | Ensure no running test/process uses it. |
| `__pycache__/` | Generated Python cache. | High | None beyond confirming no process is running. |
| `markov.db` | Local SQLite DB; README says local test DB, but runtime default is `data/markov.db`. | Medium | Confirm it is not the only useful local training/test dataset. |
| `db_prod_copy/` | Looks like production DB copy; should not live in repo workspace. | Medium | Confirm backup/retention requirements; move securely before deletion. |
| `.idea/` | Local IDE metadata. | Medium | Keep if user wants local IDE settings; do not commit. |
| `.venv/` | Recreateable virtualenv and currently out of sync. | Medium | Confirm no local-only packages are intentionally installed. |

### Security notes

- No committed `.env` or obvious real token was found in tracked files.
- `.env.example` contains placeholders and generation guidance, acceptable for repo.
- SQL statements appear parameterized.
- `/pivo` payload encryption and HMAC are present, but README must describe the actual scheme precisely.
- `chat_id` log masking is present in learning logs. `/pivo` logs do not print raw IDs.
- Local DB copies and `.env` remain the main operational privacy risk in this workspace.

### Recommended next steps

1. Address P1 hygiene/security: `.dockerignore` for local DB/temp/screenshot artifacts; decide where `db_prod_copy/` belongs.
2. Sync local `.venv` and `.env` with project files.
3. Resolve `/pivo` group-vs-DM product contract.
4. Fix README and architecture docs drift.
5. Add Docker build smoke and dependency sync checks when convenient.

---

## Historical audit snapshot — 2026-05-10

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
