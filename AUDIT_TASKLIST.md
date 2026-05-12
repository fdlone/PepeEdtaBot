# Таск лист по результатам аудита безопасности и стабильности

> Составлен на основе `PROJECT_SECURITY_STABILITY_AUDIT.md` и `PROJECT_AUDIT_CODEX.md`  
> Дата: 2026-05-12  
> Ветка аудита: `audit-security-stability-review`

---

## 🔴 Критично (делать сейчас)

- [x] **[AUD-009]** Обновить `cryptography` до незатронутой версии, пересобрать `requirements.lock`, перезапустить полный чек-сет (`ruff`, `mypy`, `unittest`, `safety`)
- [x] **[AUD-001]** Добавить лимит на число явных упоминаний в `/pivo` — отклонять или усекать вызов, если explicit mentions превышают N; добавить тесты
- [x] **[AUD-001]** Добавить лимит на subscriber fanout в `/pivo` — аналогично, с пользовательским сообщением при усечении

---

## 🟠 Высокий приоритет

- [x] **[CA-F1 / AUD-005]** Добавить `GroupOnly()` ко всем `/pivo*` хендлерам + regression-тест на отказ в приватном чате *(выполнено в codex-логе — проверено, в `main`)*
- [x] **[AUD-005 / CA-F7]** Зеркалировать паттерны из `.gitignore` в `.dockerignore`: `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg`
- [x] **[CA-F2]** Синхронизировать `README.md` — убрать HKDF из описания `/pivo`, исправить privacy-блок про raw `chat_id` в логах
- [x] **[AUD-004 / AUD-006]** Добавить операционный runbook — ротация логов, WAL checkpoint, резервное копирование и восстановление БД

---

## 🟡 Средний приоритет

- [x] **[AUD-002]** Добавить TTL/LRU для `RuntimeState` и `ThrottlingMiddleware` — очищать неактивные ключи по чату/пользователю
- [x] **[AUD-003]** Ограничить prefix-cache — инкрементальное хранение хешей префиксов или сэмплирование последних N строк вместо полного `fetchall()`
- [ ] **[AUD-007]** Добавить централизованный error middleware для Telegram API — логирование и политика на outage/rate-limit
- [x] **[CI]** Добавить `bandit` и `safety` в CI workflow (`ci.yml`) — инструменты уже в `requirements-dev.txt`, но не запускаются в пайплайне
- [ ] **[CA-F11 / AUD-008]** Синхронизировать локальный `.env` с `.env.example` вручную, не публикуя значения
- [x] **[CA-F5 / CA-F6]** Синхронизировать `.venv` через `pip install -r requirements-dev.txt` или пересоздать окружение
- [x] **[CA-F3]** Обновить `docs/ARCHITECTURE.md` — убрать или заменить хрупкую метрику числа тестов
- [x] **[CA-F4]** Разделить quickstart в `README.md` на runtime-install и dev-install; рекомендовать `requirements-dev.txt` для разработки
- [ ] **[CA-F8]** Решить судьбу `db_prod_copy/` — вынести за пределы workspace или добавить явную запись в `.gitignore`/`.dockerignore`

---

## 🔵 Низкий приоритет / технический долг

- [ ] **[AUD-004]** Определить политику retention для таблиц `messages` и transitions — лимит строк на чат, документация по VACUUM/compaction
- [ ] **[AUD-011]** Заменить `assert` в DB/service-коде на явные runtime-исключения (Bandit Low findings)
- [x] **[CA-F14 / AUD-011]** Добавить Docker build job в CI без запуска бота *(выполнено в codex-логе — проверено, в `main`)*
- [ ] **[AUD-010]** Рассмотреть отдельный `LOG_MASKING_SECRET` — только если нужна стабильная корреляция логов при ротации `PIVO_HMAC_SECRET`
- [ ] **[AUD-006]** Улучшить Docker healthcheck — сейчас проверяет только запуск интерпретатора, не polling и не БД
- [ ] **[AUD-006]** Добавить structured logging / метрики — рестарты контейнера, RSS, ошибки Telegram API, счётчик `/pivo`
- [ ] **[CA-F10]** Удалить или заигнорировать `Screenshot_*.jpg` — сейчас untracked и не в `.gitignore`
- [ ] **[CA-F13 / AUD-012]** Рефакторинг stats/clear helpers в `db.py` — выделить query helpers; делать при следующем DB-related изменении

---

## ✅ Проверить статус (помечены как выполненные в codex-логе)

- [x] **[CA-F1 / AUD-005]** `GroupOnly()` на `/pivo*` — влито ли в `main`?
- [x] **[CA-F14]** Docker build job в CI — влито ли в `main`?
- [x] **[CA-F9]** Корневой `markov.db` — статус resolved, перепроверить
- [x] **[CA-F12]** `AGENTS.md` — статус resolved, перепроверить

---

## Справка по ID находок

| ID | Источник | Краткое описание |
|---|---|---|
| AUD-001 | Security Audit | `/pivo` explicit mentions не ограничены |
| AUD-002 | Security Audit | Unbounded runtime dictionaries |
| AUD-003 | Security Audit | Prefix cache полностью перестраивается на каждом новом сообщении |
| AUD-004 | Security Audit | Нет политики retention/compaction для БД |
| AUD-005 | Security Audit | `.dockerignore` слабее `.gitignore` |
| AUD-006 | Security Audit | Нет ротации логов, метрик, осмысленного healthcheck |
| AUD-007 | Security Audit | Нет централизованной политики ошибок Telegram API |
| AUD-008 | Security Audit | Локальный `.env` не синхронизирован с `.env.example` |
| AUD-009 | Security Audit | `cryptography==45.0.7` — 3 CVE по данным Safety |
| AUD-010 | Security Audit | Log masking привязан к `PIVO_HMAC_SECRET` |
| AUD-011 | Security Audit | Bandit: 47 Low findings (`assert`, non-crypto `random`) |
| CA-F1 | Codex Audit | `/pivo*` без `GroupOnly()` |
| CA-F2 | Codex Audit | README: неверное описание HKDF и `chat_id` в логах |
| CA-F3 | Codex Audit | `docs/ARCHITECTURE.md`: устаревшее число тестов |
| CA-F4 | Codex Audit | Quickstart не воспроизводит CI/Docker dependency set |
| CA-F5 | Codex Audit | Локальный `.venv` дрейфует от `requirements.lock` |
| CA-F6 | Codex Audit | `ruff` и `mypy` отсутствуют в локальном `.venv` |
| CA-F7 | Codex Audit | Docker build context включает локальные артефакты |
| CA-F8 | Codex Audit | `db_prod_copy/` в рабочем workspace |
| CA-F9 | Codex Audit | Корневой `markov.db` как локальный артефакт |
| CA-F10 | Codex Audit | `Screenshot_*.jpg` untracked и не в `.gitignore` |
| CA-F11 | Codex Audit | Локальный `.env` устарел |
| CA-F12 | Codex Audit | `AGENTS.md` tracked, хотя `.gitignore` помечает его как local |
| CA-F13 | Codex Audit | `db.py`: dense cross-domain SQL в stats/clear helpers |
| CA-F14 | Codex Audit | Нет автоматического Docker build smoke в CI |
