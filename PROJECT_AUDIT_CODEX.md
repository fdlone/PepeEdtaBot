# PROJECT_AUDIT_CODEX

Дата аудита: 2026-05-08  
Ветка аудита: `refactor/structure` (`f523de7`)  
База сравнения: `origin/main` (`a261a6d`)  
Важно: файл `PROJECT_AUDIT.md` по просьбе не открывался и не читался. В diff он виден только как имя файла и статистика.

## 1. Резюме

Проект стал заметно взрослее относительно `origin/main`: монолитный `main.py` почти полностью разложен на handlers, filters, services, repositories и migration runner; появились versioned migrations, CI, ruff, mypy и широкий набор тестов. По фактическим проверкам текущая ветка выглядит рабочей: `python -m unittest discover tests -v` прошел 181 тест, `ruff` и `mypy app/` также прошли без ошибок.

Главные риски сейчас не в "код не запускается", а в эксплуатационных и продуктовых углах:

- часть отказов в админ-командах теперь молча фильтруется, тогда как в `main` пользователь получал понятный ответ;
- migration runner простой и пока достаточный для текущих миграций, но хрупкий для будущих SQL-файлов и частичных сбоев;
- новый `MembersService` и таблица `chat_member_profiles` хорошо протестированы, но не подключены к runtime-сценариям;
- `requirements.lock` добавлен, но не используется ни Dockerfile, ни CI, поэтому не дает реальной воспроизводимости;
- README содержит абсолютные ссылки на старый путь `D:/test/PepeEdtaBot`.

Общая оценка: хорошая рефакторинговая ветка, пригодная к дальнейшему слиянию после точечного исправления UX-регрессии по админ-командам и решения, что делать с lock-файлом и `MembersService`.

## 2. Что проверено

Команды:

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover tests -v
python -m ruff check app/ tests/
python -m mypy app/
```

Результаты:

- Unit tests: 181 тест, OK.
- Ruff: `All checks passed!`
- Mypy: `Success: no issues found in 29 source files`.

Git-сравнение:

- `origin/main...HEAD`: 48 файлов изменено, 3636 insertions, 864 deletions.
- Основной перенос: `main.py` уменьшен с 588 до 84 строк, большая часть логики вынесена в `app/`.
- `db.py` уменьшен с 627 до 373 строк и стал фасадом над репозиториями и миграциями.

## 3. Что стало лучше относительно `origin/main`

### 3.1 Архитектура стала намного чище

В `origin/main` почти вся orchestration-логика жила в `main.py`: handlers, permission checks, `/pivo`, `/set`, `/clear`, генерация ответов и humanized reply. В текущей ветке `main.py` отвечает в основном за wiring:

- создание `Database`, `MarkovGenerator`, `PivoService`, `LearningService`;
- регистрацию middleware;
- прокидывание зависимостей в dispatcher;
- подключение routers.

Это улучшение по поддерживаемости: теперь отдельные области можно читать и тестировать независимо:

- `app/handlers/*` — Telegram commands и message handling;
- `app/filters/*` — group/admin checks;
- `app/services/*` — бизнес-логика;
- `app/repositories/*` — доступ к SQLite;
- `app/infrastructure/migrator.py` — миграции.

### 3.2 Слой данных стал ближе к нормальной evolvable-схеме

В `origin/main` схема создавалась прямо в `Database.init()`. Сейчас появились versioned migrations:

- `001_initial.sql`;
- `002_normalize_messages_text_column.py`;
- `003_anonymize_authors.py`;
- `004_chat_member_profiles.sql`;
- `005_drop_messages_text.py`.

Это большой шаг вперед: схема теперь имеет историю, есть `schema_migrations`, а тесты проверяют fresh DB, existing DB, partial migration resume, real-schema fixture и сохранность Markov/pivo данных.

### 3.3 Privacy улучшена

Сильные улучшения:

- `messages.author_id` зануляется миграцией и новые сообщения пишутся с `author_id = 0`;
- `messages.text` удаляется миграцией `005`, остается `normalized_text`;
- `/pivo` продолжает хранить идентификаторы через HMAC и Fernet;
- для `MembersService` добавлен HKDF-domain separation, чтобы новый домен не переиспользовал raw pivo-ключи напрямую.

Относительно `main` это лучше: в старой схеме raw text сохранялся в `messages.text`, а новая ветка явно уводит проект к data minimization.

### 3.4 Тестовое покрытие стало существенно лучше

Добавлены тесты для:

- filters и throttling middleware;
- handlers;
- migration runner и совместимости с legacy DB;
- MembersRepo/MembersService/key derivation;
- LearningService dedup;
- runtime config и DB logic.

Это, пожалуй, самый сильный практический плюс ветки: рефактор не просто разложил файлы, он закрепил поведение тестами.

### 3.5 CI и статический контроль появились там, где их не было

Добавлен `.github/workflows/ci.yml`:

- Python matrix: 3.12, 3.13, 3.14;
- ruff;
- mypy;
- unittest discovery.

`pyproject.toml` включает ruff и строгий mypy для `app.*`. Legacy-модули пока выведены в `ignore_errors`, что прагматично для поэтапного рефакторинга.

### 3.6 Генерация стала устойчивее к дословным повторам

`LearningService.is_duplicate()` добавил проверку префиксов сохраненных сообщений перед отправкой сгенерированного текста. Относительно `main`, где retry был только на пустой результат, это улучшает качество ответов и снижает риск "бот повторил обучающую фразу почти дословно".

### 3.7 /pivo стал чище по структуре

Логика `/pivo_on`, `/pivo_off`, сборки упоминаний и расшифровки участников вынесена в `PivoService`. Это снижает нагрузку на handler и делает код ближе к тестируемому бизнес-слою.

## 4. Что стало хуже или рискованнее относительно `origin/main`

### 4.1 Админ-команды теперь молча игнорируются при отсутствии прав

Файл: `app/filters/admin_or_owner.py`, строки 14-26.  
Файл: `app/handlers/admin.py`, строки 43, 100, 140.

В `origin/main` `/set`, `/setprob` и `/clear` вручную проверяли права и отвечали пользователю:

- для `/set` и `/setprob`: "Команда доступна OWNER_ID и администраторам чата.";
- для `/clear`: "Недостаточно прав. Нужен OWNER_ID или права админа чата."

В текущей ветке эти команды завязаны на `AdminOrOwner()` filter. Если filter возвращает `False`, handler просто не вызывается. Пользователь без прав не получает никакого объяснения.

Это UX-регрессия относительно `main`. Особенно неприятно для `/clear` и `/set`, потому что пользователь может думать, что бот сломан или не видит команды.

Рекомендация: добавить fallback-handler для этих command names после защищенных handlers или вынести проверку прав обратно в handler с явным ответом. Можно сохранить filter для "разрешенного" пути, но нужен отдельный denied response.

### 4.2 Migration runner хрупок для будущих SQL-миграций

Файл: `app/infrastructure/migrator.py`, строки 58-71.

Текущий SQL splitter:

```python
return [s.strip() for s in sql.split(";") if s.strip()]
```

Для текущих простых SQL-файлов этого достаточно, и тесты это подтверждают. Но для будущих миграций он сломается на:

- `;` внутри строковых литералов;
- triggers/views;
- сложных SQL blocks;
- комментариях и edge cases SQLite grammar.

Также `run()` применяет migration и записывает ее в `schema_migrations` после `_apply()`, но не оборачивает каждую миграцию в явную транзакцию с rollback на ошибке. Для DDL в SQLite это часто работает терпимо, но как infrastructure-код это слабое место.

Рекомендация: для `.sql` использовать `conn.executescript(sql)` внутри явной транзакции/rollback или оставить `.sql` только для простейших DDL и зафиксировать это правилом в документации.

### 4.3 `MembersService` выглядит как подготовленный, но не включенный функционал

Файл: `app/services/members_service.py`.  
Файл: `main.py`, строки 20-48.

`MembersService` хорошо сделан и протестирован, но runtime wiring его не создает и handlers его не используют. Поиск по проекту показывает usage только в tests. При этом миграция `004_chat_member_profiles.sql` уже добавляет таблицу в прод-схему.

Это не баг исполнения, но архитектурный долг: в проекте появилась новая доменная зона хранения consent/profile, которая пока не участвует в пользовательских сценариях. Относительно `main` это "хуже" в смысле большей поверхности поддержки без runtime-ценности.

Рекомендация: либо подключить сценарий, для которого нужен `MembersService`, либо отложить таблицу/сервис до реального использования.

### 4.4 Dedup фильтр намеренно вероятностный и может пропустить near-copy

Файл: `app/services/learning_service.py`, строки 36-56.

Комментарий прямо говорит, что случайная длина префикса может то отклонить, то пропустить одно и то же сообщение. Это лучше, чем отсутствие dedup в `main`, но как guard от копирования обучающих сообщений он нестрогий:

- если сохраненное сообщение имеет первые 3-4 токена, а candidate длиннее и random выбрал 5, совпадение может не сработать;
- поведение становится недетерминированным, что усложняет диагностику "почему бот иногда повторяет".

Рекомендация: если цель privacy/anti-copy, проверять все префиксы от 3 до `min(5, len(tokens))` и отклонять при любом совпадении. Если цель именно вариативность, оставить как есть, но назвать это не защитой от копий, а эвристикой снижения повторов.

### 4.5 `requirements.lock` не используется как lock-файл

Файл: `requirements.lock`, строки 1-24.  
Файл: `.github/workflows/ci.yml`, install step использует `requirements-dev.txt`.  
Файл: `Dockerfile`, install step использует `requirements.txt`.

Ветка добавила `requirements.lock`, но:

- CI ставит `requirements-dev.txt`;
- Docker ставит `requirements.txt`;
- lock не содержит `mypy` и `ruff`, хотя CI зависит от них через dev requirements;
- нет инструкции, чем и когда обновлять lock.

Итог: файл называется lock, но не закрепляет реальные окружения проекта. Это может ввести в заблуждение и создать drift между локальной разработкой, CI и Docker.

Рекомендация: выбрать одну стратегию:

- либо использовать lock в Docker/CI;
- либо переименовать его в runtime constraints;
- либо удалить до внедрения нормального процесса lock/update.

### 4.6 README содержит абсолютные ссылки на старый путь

Файл: `README.md`, строки 47 и 54.

Ссылки ведут на:

- `/D:/test/PepeEdtaBot/compose.yaml`;
- `/D:/test/PepeEdtaBot/.env.example`.

В текущем workspace проект лежит в `D:\MyProject\PepeEdtaBot`, а в GitHub такие ссылки тоже не будут полезными. Это мелкий, но видимый documentation regression.

Рекомендация: заменить на относительные ссылки `compose.yaml` и `.env.example`.

### 4.7 Throttling молча выбрасывает команды

Файл: `app/middlewares/throttling.py`, строки 35-37.

Middleware сейчас silently drops throttled `/pivo` и `/clear`. Для `/pivo` это приемлемо, потому что команда шумная. Для `/clear` это менее очевидно: если админ случайно повторил команду, бот не объяснит, что действует cooldown.

Относительно `main` это новый behavior. Не критично, но стоит осознанно решить: silent drop или короткий ответ "слишком часто".

## 5. Архитектурная оценка

### Границы модулей

Границы в целом хорошие:

- `main.py` больше не знает детали команд;
- handlers тонкие;
- services держат доменную логику;
- repositories инкапсулируют SQL;
- `Database` сохраняет старый публичный API через делегаты, что снижает риск для `MarkovGenerator` и legacy tests.

Оставшийся компромисс: `Database` все еще содержит крупные cross-domain операции `save_message_and_update_model()`, `get_stats()`, `clear_chat()`. Это нормально для текущего размера проекта, но если рефактор продолжать, `save_message_and_update_model()` логически просится в отдельный Markov write service/repo.

### Dependency injection

Aiogram dispatcher data используется аккуратно:

- `dp["db"]`;
- `dp["generator"]`;
- `dp["pivo_service"]`;
- `dp["learning_service"]`;
- `dp["state"]`;
- `dp["settings"]`;
- `dp["bot_username"]`;
- `dp["bot_id"]`.

Это делает handlers тестируемыми без запуска Telegram polling. По сравнению с closure-based handlers в `main` это лучше.

### Legacy compatibility

Сильная сторона текущей ветки: очень много тестов на старые схемы. Особенно полезны:

- legacy fixture;
- проверки сохранности row counts;
- backfill `normalized_text`;
- удаление `text`;
- anonymize `author_id`;
- partial migration resume.

## 6. Security/privacy

Плюсы:

- меньше raw PII в `messages`;
- `/pivo` opt-in;
- encrypted member data;
- HMAC для chat/user identifiers;
- HKDF domain separation для нового members-домена;
- `.env.example` хорошо предупреждает про секреты.

Риски:

- Fernet/HMAC secrets при смене ломают доступ к старым подпискам, это документировано;
- нет rotation/migration механизма для `key_version`, хотя поле уже заложено;
- `chat_member_profiles` добавлен, но runtime consent flow пока отсутствует.

Практический вывод: privacy стало лучше, но ключевая версия `key_version` пока скорее задел, чем работающий rotation design.

## 7. DB и миграции

Схема выглядит последовательной:

- Markov tables сохранены;
- `messages` минимизирована до `chat_id`, `author_id=0`, `normalized_text`, `created_at`;
- pivo table сохранена отдельно;
- members table добавлена отдельно.

Индексы покрывают основные lookups:

- `messages(chat_id)`;
- `messages(chat_id, normalized_text)`;
- `starts/transitions lookup`;
- `pivo_chat_members(chat_hash)`;
- `chat_member_profiles(chat_hash)`.

Основной технический риск: migration runner, описанный выше. Для текущего набора миграций он проходит тесты, но его лучше укрепить до появления более сложных SQL-файлов.

## 8. Тесты

Текущая тестовая база сильная для такого проекта. Особенно хорошо покрыто:

- DB schema и migrations;
- Markov behavior;
- runtime config validation;
- pivo crypto/mention behavior;
- Telegram handlers через mocks;
- filters and throttling.

Что еще стоит добавить:

- тест, что unauthorized `/set`, `/setprob`, `/clear` дают явный отказ, если такое поведение вернуть;
- тест на wiring `main.py`/routers хотя бы smoke-level, чтобы все expected dependencies действительно регистрируются;
- тест на deterministic strict dedup, если решите поменять вероятностный prefix check;
- integration smoke для Docker/entrypoint, если проект деплоится контейнером.

## 9. Документация и DevEx

Хорошо:

- README объясняет команды, Docker, pivo secrets, group privacy;
- `.env.example` подробный;
- CI простой и понятный;
- dev requirements включают runtime requirements через `-r requirements.txt`.

Нужно поправить:

- README absolute links;
- README говорит `Python 3.14`, а CI поддерживает 3.12/3.13/3.14. Лучше написать `Python 3.12+` или явно объяснить baseline;
- `requirements.lock` должен либо использоваться, либо не называться lock;
- нет команды "как запустить проверки локально" в README.

## 10. Приоритеты исправлений

### P1

1. Вернуть понятный отказ для unauthorized `/set`, `/setprob`, `/clear`.
2. Определиться с судьбой `requirements.lock`: использовать или убрать/переименовать.

### P2

1. Укрепить migration runner: transaction/rollback и более надежное применение SQL.
2. Исправить README absolute links.
3. Решить, нужен ли `MembersService` в этой ветке; если нужен, подключить runtime flow.

### P3

1. Сделать dedup неслучайным, если он считается privacy/security guard.
2. Добавить smoke-тест на wiring приложения.
3. Рассмотреть user feedback для throttled `/clear`.

## 11. Итоговая оценка по сравнению с `origin/main`

Стало лучше:

- поддерживаемость;
- тестируемость;
- privacy;
- миграционная история;
- CI/quality gates;
- separation of concerns;
- готовность к дальнейшему развитию.

Стало хуже или спорнее:

- UX отказа в админ-командах;
- больше инфраструктурной сложности;
- появился неиспользуемый members-домен;
- lock-файл пока не выполняет роль lock-файла;
- часть документации содержит stale local paths.

Финальный вывод: ветка `refactor/structure` является качественным шагом вперед относительно `origin/main`, но перед merge я бы исправил P1 и хотя бы README links. Остальное можно вести как follow-up задачи, потому что текущие тесты и статические проверки зеленые.
