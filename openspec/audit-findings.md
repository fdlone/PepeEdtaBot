# Аудит PepeEdtaBot — фаза 1 (инвентаризация)

Дата: 2026-08-06. Коммит: `bd8d9d4`.
Аудит проведён «с нуля» по текущему коду: прошлые отчёты (`docs/OPEN.md`,
`docs/CLOSED.md`, `openspec/changes/archive/**`) не читались и не использовались.

Базовая линия перед изменениями (проверено):

- `ruff check app/ tests/ tools/ main.py` — clean
- `mypy app/` (strict, 69 файлов) — clean
- `python -m unittest discover tests` — 878 тестов, OK (~198 c)

Оси: **A** — упрощение/оптимизация кода, **B** — безопасность, **C** —
производительность, **D** — код-ревью (корректность, качество, тесты).

---

## B. Безопасность

| ID | Ось | Файл:строка | Описание | Severity | Рекомендация | Change |
|----|-----|-------------|----------|----------|--------------|--------|
| B1 | B | `app/handlers/pivo.py:137-155`, `main.py:55-57` | `/pivo` намеренно исключён из throttling, а отказ по суточной квоте отвечается **сообщением на каждый вызов**. Участник может слать `/pivo` бесконечно и получать бесконечный поток ответов бота — та самая амплификация трафика и риск flood-ban, от которых защищают cooldown'ы. Квота ограничивает полезные вызовы, но не отказы. | **critical** | Ограничить именно *уведомление об отказе* (одно на окно, как `notify_cooldown_sec` в `ThrottlingMiddleware`), дальше — тишина. Альтернатива: вернуть `/pivo` в throttling с `notify_on_throttle`. | `security-pivo-abuse-limits` |
| B2 | B | `app/handlers/pivo.py:120-131`, `app/services/pivo_service.py:188-213` | Сообщение `/pivo` собирается **до** списания квоты: чтение `chat_members` + по 3 Fernet-decrypt на подписчика выполняются даже для вызова, который будет отклонён. Вместе с B1 даёт дешёвый способ грузить бота криптой и БД. | high | Списывать квоту до `build_call_message`; при последующем сбое возврат уже реализован (`refund_daily_call_quota`). | `security-pivo-abuse-limits` |
| B3 | B | `app/domain/pivo.py:33-36` | Ключ Fernet выводится как `base64(sha256(PIVO_ENCRYPTION_SECRET))` — без KDF, соли и растяжения. Секрет проверяется только на длину ≥16 и не-placeholder, т.е. допускается низкоэнтропийная парольная фраза, перебираемая по украденной БД. Рядом в проекте есть корректный пример (`log_masking` использует HKDF-SHA256 с доменной меткой). | high | Перейти на HKDF-SHA256 с меткой `pivo:fernet` и версией ключа. **Breaking**: нужна миграция (пере-шифровать `chat_members` или `MultiFernet` со старым ключом). Вынесено в спорные — решение за владельцем. | `security-pivo-crypto` (спорный) |
| B4 | B/C | `app/domain/pivo.py:172-182`, `app/repositories/chat_members_repo.py:88-105` | `collect_pivo_mentions` расшифровывает `encrypted_user_id` каждого подписчика и заново считает HMAC, чтобы сравнить с вызывающим, — хотя `user_hash` (тот же HMAC) уже лежит в строке и просто не выбирается запросом. Сравнение хешей — обычный `==` (не `hmac.compare_digest`). | medium | Выбирать `user_hash` в `list_members`, сравнивать через `hmac.compare_digest`. Побочный эффект — минус один decrypt+HMAC на подписчика. | `security-pivo-crypto` |
| B5 | B | `app/services/pivo_message_builder.py:304-315`, `app/services/pivo_parser.py:38-61` | Аргументы `/pivo` (`target`) не имеют ограничения длины. Пользователь отправляет `/pivo <~4000 символов>` — итоговое сообщение превышает лимит Telegram (4096), отправка падает всегда. Квота при этом возвращается, т.е. приём повторяем без ограничений. | medium | Ограничить `target` (например, 200 символов) и `planned_time` на входе парсера; при превышении — усечение или понятный отказ. | `security-pivo-abuse-limits` |
| B6 | B | `app/services/pivo_service.py:209-213` | Если в чате подписалось больше `PIVO_SUBSCRIBER_FANOUT_LIMIT` (по умолчанию 20) человек, `/pivo` **навсегда** отдаёт `PivoCallLimitError` для всех: команда ломается, и починить её нельзя ничем, кроме `/clear confirm`. Отписать другого нельзя. Лимит задуман как защита от fanout, а работает как self-DoS. | medium | Усекать список до лимита (с пометкой «и ещё N»), а не падать; либо проверять лимит при `/pivo_on`. Изменение наблюдаемого поведения — см. спорные. | `security-pivo-abuse-limits` |
| B7 | B | `tests/test_log_masking.py`, `tests/test_error_handler.py`, `tests/test_migrator.py`, `tests/fixtures/legacy_real_schema.sql`, `docs/CLOSED.md`, `tools/sweeps/*_results.jsonl` | Реальный prod `chat_id` (`-1001147461458`, совпадает с основным чатом в `db_prod_copy`) закоммичен в репозиторий — ~280 вхождений, из них 267 в отслеживаемых `tools/sweeps/*.jsonl`. Это прямо противоречит инварианту, ради которого написан `log_masking`: сырой идентификатор чата не должен утекать наружу. | medium | Заменить на синтетические id в тестах/доках, вычистить или разотслеживать sweep-результаты. | `privacy-scrub-prod-chat-id` |
| B8 | B | `.gitignore` | `scrin/` (скриншоты реального чата) исключён только из `.dockerignore`; в `.gitignore` есть лишь `Screenshot_*.jpg`. Сейчас каталог не отслеживается, но одна `git add -A` — и содержимое чата в истории. | low | Добавить `scrin/` в `.gitignore`. | `privacy-scrub-prod-chat-id` |
| B9 | B | `app/handlers/common.py`, `app/handlers/admin.py:65-79` | `/config` и `/config full` не имеют ни `GroupOnly`, ни проверки прав: любой участник любого чата (и личка) видит полную runtime-конфигурацию. Секретов там нет, но это раскрытие настроек модерации (шанс ответа, кулдауны, лимиты). | low | Решение владельца: оставить как есть либо закрыть `AdminOrOwner`. | опционально |
| B10 | B | `Dockerfile`, `docker-entrypoint.sh`, `.github/workflows/ci.yml`, `requirements.lock` | **Проверено, замечаний нет.** Образ копирует белый список файлов; entrypoint стартует root'ом только чтобы сделать `chown /app/data`, затем `runuser -u bot`; healthcheck открывает БД read-only URI; зависимости запинены; в CI есть bandit (medium/medium) и pip-audit по lock-файлу. | info | — | — |
| B11 | B | `app/repositories/*`, `app/config/registry.py` | **Проверено, замечаний нет.** Все SQL-запросы параметризованы; f-string используется только для литеральных имён таблиц/колонок из кода (помечено `nosec B608`). `/set` резолвит ключ через словарь спеков, значение — через типизированный парсер с диапазоном, NaN/inf отбиты. HTML в `/pivo` экранируется на всех путях (упоминания, target, время). Инъекций не найдено. | info | — | — |

## C. Производительность

| ID | Ось | Файл:строка | Описание | Severity | Рекомендация | Change |
|----|-----|-------------|----------|----------|--------------|--------|
| C1 | C | `app/repositories/markov_repo.py:140-161`, `app/handlers/learning.py:287` | `get_token_volume` на **каждое сообщение** делает `SELECT SUM(cnt) FROM transitions3 WHERE chat_id=?` (30 907 строк на живом чате в prod-копии). При этом инкрементальная таблица `chat_model_volume` уже создана миграцией 009 и поддерживается на записи — её просто не читают на чтении. | **high** | Читать `chat_model_volume`; полный `SUM` оставить как fallback/бэкфилл. | `perf-model-volume` |
| C2 | C | `app/services/learning_service.py:53-69, 226-231` | `_invalidate_text_cache` сбрасывает **пять** кэшей чата на каждое выученное сообщение. Следующий ответ пересобирает всё заново, синхронно в event loop: 1000 нормализованных текстов, IDF по этим же 1000 сообщений (токенизация + стемминг каждого), весь кумулятивный индекс 4-грамм, частотный словарь (`GROUP BY` по 30k строк). В активном чате почти каждый ответ идёт по «холодному» пути. | **high** | Инкрементальное обновление вместо сброса (добавить вклад нового сообщения), либо разделить инвалидацию по потребителям и отложить перестроение. | `perf-learning-caches` |
| C3 | C | `app/infrastructure/database.py:631-643`, `app/services/learning_service.py:153-169`, `app/migrations/016_verbatim_ngrams.py` | `get_verbatim_ngram_index` читает **всю** `chat_verbatim_ngrams` чата и строит `frozenset` кортежей. Таблица по замыслу не подрезается ретенцией (растёт вечно): на prod-копии — порядок 20–25 тыс. строк, и это только начало жизни чата. Вместе с C2 — полная пересборка на каждый ответ. | **high** | Кэшировать по-настоящему (не сбрасывать целиком — досыпать новые 4-граммы), либо перенести проверку в SQL (`EXISTS` по окнам кандидата), либо ввести ретенцию/сжатие индекса. | `perf-learning-caches` |
| C4 | C | `app/services/learning_service.py:53-69` | Пять словарей кэшей `LearningService` ключуются `chat_id` и **не ограничены** ни TTL, ни числом чатов — в отличие от соседних состояний (`RuntimeState.prune_inactive`, `ThrottlingMiddleware._prune_state`, `_admin_cache`). Чат, который замолчал, держит свой индекс 4-грамм и частотный словарь до рестарта. | medium | Тот же паттерн вытеснения, что у соседей (TTL + max_chats). | `perf-learning-caches` |
| C5 | C | `app/services/pivo_service.py:63, 101-103` | `PivoService._refreshed_on` (дроссель обновления профиля) растёт неограниченно: ключи `(chat_hash, user_hash)` никогда не удаляются, записи прошлых дней не чистятся. Ключи приходят снаружи (любой участник любого чата). | medium | Чистка по смене дня или общий паттерн вытеснения. | `perf-learning-caches` |
| C6 | C | `app/core/markov.py:560, 609, 636` | `weighted_next_choice` / `weighted_start2_choice` / `weighted_start3_choice` **сортируют** входной список на каждом вызове. Для стартов это 4 482 строки (prod-копия) × 10 попыток генерации на ответ, плюс вызовы из прыжков. Списки уже лежат в кэше `_cache_starts3`/`_cache_starts2` и меняются только при обучении. | medium | Сортировать один раз при заполнении кэша; в функциях выбора полагаться на отсортированность. | `perf-generation-hotpath` |
| C7 | C | `app/core/slot_mutation.py:196-236` | `pick_replacement` линейно обходит **весь** частотный словарь чата на каждый слот-кандидат и для выживших дёргает pymorphy3. Отбор по двухбуквенному окончанию делается уже внутри цикла — индекс `ending → слова` снял бы ~99% итераций. | medium | Предпостроить индекс по окончанию при загрузке частот (одноразово на чат). | `perf-generation-hotpath` |
| C8 | C | `app/repositories/chat_hot_ngrams_repo.py:44-110` | Вторая ветка `get_hot` агрегирует `transitions` по всему чату (`GROUP BY chat_id, w1, w2`) при каждом вызове; вызывается на seed-путях и при чтении входов слот-мутаций. | medium | Ограничить подзапрос присутствующими в окне биграммами либо материализовать агрегат. | `perf-generation-hotpath` |
| C9 | C/A | `app/infrastructure/database.py:602-629`, `app/presentation/bot_messages.py:88-89` | `Database.get_stats` выполняет 7 агрегатов по чату, а `/stats` печатает **только** `volume`. Шесть запросов — чистая выброшенная работа; вдобавок `volume2`/`volume3` уже лежат в `chat_model_volume`. | medium | Свести к одному чтению `chat_model_volume` (остальное — по требованию). | `perf-model-volume` |
| C10 | C/A | `app/core/candidate_scorer.py:366-377`, `app/core/response_generator.py:389-414` | `score_candidate` считает `context_relevance`, но `_build_score` — единственный продовый путь — всегда перезаписывает это поле `idf_context_relevance`. Вычисление выбрасывается для каждого кандидата. | low | Не считать поле там, где оно заведомо будет заменено. | `simplify-dead-code` |
| C11 | C | `app/core/markov.py:274-287, 434-472` | `longest_shared_run` — O(n·m·k) — вызывается в двух гейтах на каждую завершённую попытку генерации. При текущих длинах (≤45 токенов) это терпимо, но алгоритм квадратичный по построению. | low | Заменить на суффиксное сравнение по хешам/динамику O(n·m). | `perf-generation-hotpath` |
| C12 | C | `app/infrastructure/database.py:197-214`, migrations 001/008 | **Проверено, замечаний нет.** Индексы соответствуют запросам (после миграции 008 лишние сняты), WAL + `busy_timeout` + `wal_autocheckpoint` настраиваются из конфига, ретенция `messages` и суточные decay'и на месте, все обращения к SQLite — через `aiosqlite` (отдельный поток), общая блокировка одна. N+1 в горячем пути не найдено. | info | — | — |

## A. Упрощение и оптимизация кода

| ID | Ось | Файл:строка | Описание | Severity | Рекомендация | Change |
|----|-----|-------------|----------|----------|--------------|--------|
| A1 | A | `app/infrastructure/database.py:316-592` | `Database` — фасад на 666 строк, из которых ~40 методов — однострочные делегаты в репозитории (`get_starts`, `record_chat_emojis`, `decay_*`, …). Слой не добавляет ни логики, ни инвариантов, только `_require`. | medium | Оставить в `Database` соединение, миграции, репозитории и кросс-доменные операции (`save_message_and_update_model`, `clear_chat`, `get_stats`); потребителей перевести на `db.markov.*` / `db.messages.*`. Тестируемость не страдает — репозитории уже отдельные. | `simplify-db-facade` |
| A2 | A | `app/core/markov.py:14`, `app/core/context_state_matcher.py:6`, `app/infrastructure/database.py:43-64` | Нарушение границ слоёв: `app/core/*` импортирует `app.infrastructure.database`. Из-за возникающего цикла в `database.py` **продублированы** `PUNCT_SET`, `VERBATIM_NGRAM_SIZE` и функция построения 4-грамм (с комментарием «importing it would cycle»); третья копия — в миграции 016. | medium | Ввести Protocol-порт (`MarkovReadPort`) в `app/core`, инфраструктуру подставлять в `main`. Дубли констант убрать. | `simplify-layering` |
| A3 | A | `app/handlers/learning.py:181-607` | `on_text_message` — 430 строк, до 7 уровней вложенности, две вложенные async-функции, `try/finally` вокруг всей логики ответа. Хендлер содержит политику ответа, mood/rhythm, seed'ы, quirks, rare events и обучение. | medium | Вынести конвейер ответа в сервис (`ReplyPipeline`), хендлер оставить тонким. Крупный рефакторинг — см. спорные. | `simplify-learning-handler` |
| A4 | A | см. описание | **Мёртвый код:** `MIN_LEARN_MESSAGE_CHARS` (`learning.py:62`) не используется нигде; `Database.message_exists` + `MessagesRepo.exists` — только в тестах; `MarkovGenerator.generate_text` — только в тестах; `reply_humanized` — только в тестах; `build_pivo_message` — только в тестах; `ContextStateMatch.similarity` всегда `1.0` и участвует только в сортировке по себе; `learning_service.content_ngram_windows` используется только из `tools/eval_prod.py`. | low | Удалить или явно пометить как API для тестов/инструментов. | `simplify-dead-code` |
| A5 | A | `app/domain/pivo.py:11`, `app/services/pivo_message_builder.py:30, 350` | Одна и та же строка `"Господа дегенераты"` объявлена в двух модулях, и `_build_notification_line` сравнивает значение со **своей** копией: расхождение констант тихо сломает подавление notification-строки. | low | Один источник (`app/domain/pivo.py`). | `simplify-dead-code` |
| A6 | A | `app/core/markov.py:152-176` | `GenerationTrace` (dataclass) и `_GenerationAttempt` (NamedTuple) — два типа с одинаковыми восемью полями; конструирование `_GenerationAttempt` идёт позиционно на 5 сайтах, что и требует комментария про порядок аргументов. | low | Один тип либо `_GenerationAttempt(text, trace)`. | `simplify-dead-code` |
| A7 | A | `pyproject.toml`, `app/config/registry.py` | **Проверено, замечаний нет.** `ignore_errors`/legacy-исключений в mypy нет — весь `app.*` уже strict. Все 64 runtime-ключа реестра реально читаются (мёртвых ручек нет). Дублирования парсинга конфига между `settings.py`/`runtime_state.py`/`runtime_config.py` нет — единый реестр. | info | — | — |

## D. Код-ревью (корректность, качество, тесты)

| ID | Ось | Файл:строка | Описание | Severity | Рекомендация | Change |
|----|-----|-------------|----------|----------|--------------|--------|
| D1 | D | `app/config/runtime_state.py:121-140, 214-221` | `effective()` возвращает shallow-copy, и инвариант «всё изменяемое состояние общее» держится только для словарей. Единственное изменяемое **скалярное** поле, `_cleanup_tick`, инкрементится на копии-однодневке: для чатов с override'ами счётчик никогда не доходит до 64. Проверено экспериментально: 100 вызовов `note_chat_activity` через view оставили `base._cleanup_tick == 0`. Следствие — TTL-вытеснение неактивных чатов с этого пути не запускается, работает только страховка по `runtime_state_max_chats`. | medium | Инкрементить тик на базовом объекте (хранить его в общем изменяемом контейнере) либо запретить мутацию скаляров у view. Нужен регрессионный тест. | `fix-runtime-overlay-tick` |
| D2 | D | `app/infrastructure/database.py:560-578` | `decay_flavor_stats_if_due` обновляет `_last_flavor_decay_monotonic` **до** запуска decay'ев: исключение внутри отложит следующую попытку на сутки, а не на ближайшее сообщение. | low | Обновлять отметку после успешного выполнения. | `fix-runtime-overlay-tick` |
| D3 | D | `app/core/markov.py:1292` | `_finalize_attempt` токенизирует готовый текст без `normalize_lower`, тогда как весь остальной конвейер работает по флагу чата. Гейты формы (`is_low_diversity_reply`, `is_context_heavy_reply`) сравнивают регистрозависимые токены — в case-preserved профиле «Слово» и «слово» считаются разными. | low | Передавать флаг чата либо явно зафиксировать инвариант тестом. | `fix-runtime-overlay-tick` |
| D4 | D | `app/core/response_generator.py:165-184, 320-328` | Кандидат нормализуется двумя разными способами: `sanitize_text().lower()` для сравнения с текущим сообщением и `normalize_reply_for_repeat()` для анти-повтора; обе нормализации считаются в одном вызове. Легко разъезжается при правках. | low | Один нормализатор с явными вариантами. | `simplify-dead-code` |
| D5 | D | `.github/workflows/ci.yml:41` | `mypy` в CI проверяет только `app/`; `tests/` и `tools/` не типизируются (при том что ruff их проверяет). Ошибка в тестовом хелпере не ловится статически. | low | Расширить mypy на `tests/`/`tools/` (можно с ослабленным профилем). | `docs-refresh` / отдельный |
| D6 | D | `app/handlers/pivo.py:175-179` | Возврат квоты предусмотрен только для сбоя отправки; исключение в `record_pool_usage` (после доставки) уходит наверх без возврата и без пометки — сообщение доставлено, а вызов будет выглядеть как упавший. | low | Обернуть пост-обработку в собственный `try/except` с логированием. | `security-pivo-abuse-limits` |
| D7 | D | `README.md:10, 40-42` vs `app/infrastructure/database.py:194-196` | Заявление «no raw texts kept» / «сырые тексты не хранятся» неточно: `messages.normalized_text` хранит текст сообщения целиком (удалены только ссылки, `@mention`, PII-паттерны и схлопнуты пробелы) в пределах окна ретенции. Ниже, в разделе Privacy, README описывает это корректно — расходятся между собой два места одного документа. | low | Привести формулировку в шапке к фактическому поведению. | `docs-refresh` |
| D8 | D | `tests/` | **Проверено:** 878 тестов, покрытие с ratchet 87%, есть регрессии на маскирование логов, фильтры прав, валидацию ключей, миграции. **Пробелы:** нет тестов на (a) отсутствие флуда `/pivo` вне квоты (B1), (b) ограничение длины аргументов `/pivo` (B5), (c) поведение при переполнении fanout (B6), (d) TTL-вытеснение при активных override'ах (D1). | low | Добавить вместе с соответствующими фиксами. | по месту |

---

## Принятые решения по спорным находкам

Закрыты рекомендациями аудита (решение владельца от 2026-08-06 — «закрывай своими
рекомендациями»); зафиксированы в соответствующих предложениях.

1. **B1 — реакция на исчерпанную квоту `/pivo`** → уведомление об отказе
   дросселируется (одно на пару «участник — чат» в окно 60 с), дальше тишина.
   Объяснение сохраняется, амплификация снимается; исходная причина вывода
   `/pivo` из-под throttling (молчание вместо объяснения) не возвращается.
2. **B3 — схема вывода ключа Fernet** → HKDF-SHA256 с меткой `pivo:fernet`,
   чтение через `MultiFernet` (новый ключ пишет, прежний читает), перешифровка
   ленивая при обновлении профиля. Без остановки бота и без единовременной
   миграции таблицы. Дополнительно порог длины секретов поднимается до 32.
3. **B6 — переполнение fanout `/pivo`** → усечение списка до предела с пометкой
   в сообщении. Отказ сохраняется только для упоминаний, перечисленных самим
   вызывающим.
4. **A1/A3 — крупные рефакторинги** → выполняются, последними перед `docs-refresh`.
   `A1` — в ограниченном виде: в `Database` остаются соединение, миграции,
   настройки SQLite, доступ к репозиториям и кросс-доменные операции.
5. **B9 — доступность `/config`** → правами закрывается только полная форма
   (`/config full`), краткая остаётся доступной всем.
6. **D5 — охват mypy** → не расширяем: `mypy tests/ tools/` даёт 261 ошибку в 26
   файлах, это самостоятельная задача. Фиксируется в `docs-refresh` как принятое
   ограничение.

---

## Изменения (фаза 2) — созданы и валидны

Порядок: security → performance → simplification → docs.
Все 13 проходят `openspec validate <id> --strict`.

| # | change-id | Находки | Spec-дельты |
|---|-----------|---------|-------------|
| 1 | `security-pivo-abuse-limits` | B1, B2, B5, B6, D6 | NEW `pivo-call-limits`, MODIFIED `command-rate-limits` |
| 2 | `privacy-scrub-prod-chat-id` | B7, B8 | MODIFIED `log-privacy` |
| 3 | `security-pivo-crypto` | B3, B4 | NEW `pivo-identity-protection` |
| 4 | `restrict-config-full` | B9 | MODIFIED `chat-scoped-settings` |
| 5 | `perf-model-volume` | C1, C9 | — (`skip_specs`) |
| 6 | `perf-learning-caches` | C2, C3, C4, C5 | MODIFIED `in-memory-state-eviction` |
| 7 | `perf-generation-hotpath` | C6, C7, C8, C11 | — (`skip_specs`) |
| 8 | `fix-runtime-overlay-tick` | D1, D2, D3 | MODIFIED `in-memory-state-eviction`, `chat-scoped-settings` |
| 9 | `simplify-dead-code` | A4, A5, A6, C10, D4 | — (`skip_specs`) |
| 10 | `simplify-layering` | A2 | — (`skip_specs`) |
| 11 | `simplify-db-facade` | A1 | — (`skip_specs`) |
| 12 | `simplify-learning-handler` | A3 | — (`skip_specs`) |
| 13 | `docs-refresh` | D7, D5 + итоги | — (`skip_specs`) |
