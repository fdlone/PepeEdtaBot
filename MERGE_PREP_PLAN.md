# План подготовки к merge `refactor/structure` → `main`

Этот документ — оперативный план для закрытия трёх P1-блокеров, выявленных во второй редакции [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md). После выполнения всех шагов и зелёного CI ветка готова к merge.

После выполнения файл удаляется (как `REFACTOR_PLAN.md` после фаз 1–6).

---

## Обзор блокеров

| # | Блокер | Файл(ы) | Усилия | Риск |
|---|---|---|---|---|
| **B1** | UX-регрессия: unauthorized админ-команды молча игнорируются | `app/handlers/admin.py`, `tests/test_handlers.py` | ~40 мин | низкий |
| **B2** | `requirements.lock` декоративен, не используется | `Dockerfile`, `requirements-dev.txt`, `requirements.lock` (header); `.github/workflows/ci.yml` — не меняется, косвенно использует lock через `requirements-dev.txt` | ~25 мин | низкий |
| **B3** | README: 2 абсолютные ссылки `/D:/test/...` | `README.md` | ~5 мин | нулевой |

**Итого:** ~70 минут чистого времени + ~10 минут на CI.

**Порядок выполнения:** B3 → B2 → B1. Самые лёгкие/безрисковые сначала; B1 содержит изменения в логике handlers + тесты — оставляем напоследок, чтобы CI отрабатывал на каждом промежуточном коммите.

**Каждый блокер — отдельный коммит** с зелёным CI. Не объединять — это даёт чистый bisect-friendly след в истории.

---

## B3 — README absolute links (~5 мин)

### Проблема

[`README.md:47`](README.md):
```markdown
Файл конфигурации: [compose.yaml](/D:/test/PepeEdtaBot/compose.yaml)
```

[`README.md:54`](README.md):
```markdown
Все параметры перечислены в [.env.example](/D:/test/PepeEdtaBot/.env.example).
```

Обе ссылки ведут на чужую машину пользователя; в GitHub UI они сломаны.

### Решение

Заменить на относительные. GitHub автоматически разрешит их в blob-URL'ы.

### Изменения

```diff
-Файл конфигурации: [compose.yaml](/D:/test/PepeEdtaBot/compose.yaml)
+Файл конфигурации: [compose.yaml](compose.yaml)
```

```diff
-Все параметры перечислены в [.env.example](/D:/test/PepeEdtaBot/.env.example).
+Все параметры перечислены в [.env.example](.env.example).
```

### Acceptance

- `git grep "/D:/test/" README.md` → пусто.
- В GitHub Web UI обе ссылки открываются (не нужно проверять, доверяем тому что относительные ссылки работают).

### Коммит

```
docs(readme): fix two stale absolute links to compose.yaml and .env.example

Ссылки указывали на путь /D:/test/PepeEdtaBot/, который существовал
только на машине автора первой версии README. В GitHub Web UI они
выглядели как сломанные ссылки. Заменено на относительные пути.

Closes B3 from MERGE_PREP_PLAN.md.
```

---

## B2 — Подключить `requirements.lock` к Dockerfile и CI (~25 мин)

### Проблема

[`requirements.lock`](requirements.lock) создан в фазе 6, но не используется ни одним консумером:
- `Dockerfile` ставит `requirements.txt` (диапазоны);
- CI через `requirements-dev.txt` ставит `requirements.txt` (диапазоны);
- В lock'е нет `mypy`/`ruff`, без которых CI падает.

Результат — формальная галочка без воспроизводимости.

### Решение

**Вариант A (выбран):** lock становится единственным источником **runtime**-зависимостей. Стратегия — простая: `pip freeze` в чистом venv, без `pip-tools`/`pip-compile`. Это **прагматично для размера проекта**, но честно говоря, имеет ограничения, которые нужно осознавать:

- header регенерируется вручную после `pip freeze` (хрупко, легко забыть);
- нет hash-pin'а, нет автоматической верификации совместимости при апгрейде;
- dev-tools (`mypy`, `ruff`) **остаются диапазонами** в `requirements-dev.txt` — закреплён только runtime, не tooling.

Если в будущем потребуется более строгая воспроизводимость, можно перейти на `pip-tools` (`requirements.in` → `pip-compile` → `requirements.txt`) или `uv lock`. Пока — `pip freeze` достаточно.

Что делаем:
- `Dockerfile` устанавливает `requirements.lock` (production-runtime, всё закреплено).
- `requirements-dev.txt` ссылается на `requirements.lock` и добавляет `ruff>=0.15.0` / `mypy>=2.0.0` диапазонами (CI и локальная разработка).
- `.github/workflows/ci.yml` **не меняется** — он уже ставит `requirements-dev.txt`, который теперь косвенно использует lock.
- В шапке `requirements.lock` — комментарий о том, что это `pip freeze`-snapshot и как его регенерировать.
- `requirements.txt` остаётся как «верхнеуровневые диапазоны» — справочный файл для разработчика, желающего увидеть прямые зависимости без транзитивных.

**Вариант B и C** (переименовать или удалить) отвергнуты: P1-5 в исходном аудите явно требовал воспроизводимости; имеющийся lock-файл достаточно подключить к консумерам, чтобы он начал выполнять свою роль для runtime — это закрывает 80% риска при минимальных усилиях.

### Изменения

#### `Dockerfile`

```diff
 FROM python:3.14-slim

 ENV PYTHONDONTWRITEBYTECODE=1
 ENV PYTHONUNBUFFERED=1

 WORKDIR /app

 RUN mkdir -p /app/data

-COPY requirements.txt .
-RUN pip install --no-cache-dir -r requirements.txt
+COPY requirements.lock .
+RUN pip install --no-cache-dir -r requirements.lock

 COPY . .

 CMD ["python", "main.py"]
```

#### `requirements-dev.txt`

```diff
--r requirements.txt
+-r requirements.lock
 mypy>=2.0.0
 ruff>=0.15.0
```

#### `requirements.lock` — добавить header

В самом начале файла, перед списком пакетов:

```
# Lock-файл runtime-зависимостей PepeEdtaBot.
# НЕ редактировать вручную: версии должны соответствовать `pip freeze`
# из чистого окружения с диапазонами из requirements.txt.
#
# Используется:
#   - Dockerfile (production-runtime)
#   - requirements-dev.txt → CI и локальная разработка
#
# Регенерация (нужен Python 3.14, как в Dockerfile):
#   python -m venv .venv-fresh
#   .venv-fresh/Scripts/activate    # Windows
#   . .venv-fresh/bin/activate      # Linux/macOS
#   pip install --upgrade pip
#   pip install -r requirements.txt
#   pip freeze > requirements.lock
#   # Восстановить этот заголовок (pip freeze его сотрёт).
#   # Убедиться что mypy/ruff не попали — они только в requirements-dev.txt.

aiofiles==25.1.0
...
```

### Тесты / валидация

Не требуются автоматизированные тесты — изменения в build-конфигурации, не в коде. Валидация:

1. **Локально:**
   ```bash
   python -m unittest discover tests   # 181 тест должно пройти
   pip install -r requirements-dev.txt  # должна установиться без ошибок
   ```

2. **CI:**
   - Запушить → проверить, что workflow проходит.
   - Шаг `pip install -r requirements-dev.txt` теперь ставит закреплённый runtime через lock + диапазоны для `ruff`/`mypy`. Резолвер всё ещё работает (для dev-tools), но runtime больше не требует его.

3. **Docker (опционально, если есть Docker под рукой):**
   ```bash
   docker build -t pepe-edta-bot:test .
   ```

### Acceptance

- `Dockerfile` содержит `RUN pip install --no-cache-dir -r requirements.lock` (вместо прежнего `requirements.txt`).
- `requirements-dev.txt` начинается с `-r requirements.lock` (вместо прежнего `-r requirements.txt`).
- В `requirements.lock` добавлен header-комментарий: характеристика стратегии (`pip freeze`-snapshot без `pip-tools`), куда подключён, как регенерировать.
- `requirements.txt` оставлен без изменений — остаётся справочным файлом верхнеуровневых диапазонов.
- CI зелёный (использует lock косвенно через `requirements-dev.txt`).
- Локальный прогон тестов зелёный.

### Коммит

```
build: wire requirements.lock into Dockerfile and CI

Ранее requirements.lock был создан, но не использовался: Dockerfile
ставил requirements.txt (диапазоны), CI — requirements-dev.txt, который
тоже ссылался на requirements.txt. В lock-файле отсутствовали mypy/ruff.
Воспроизводимость, которую обещал P1-5 из старого аудита, не работала.

- Dockerfile: requirements.txt → requirements.lock
- requirements-dev.txt: -r requirements.txt → -r requirements.lock
- requirements.lock: добавлен header с процедурой регенерации

Mypy/ruff остаются только в requirements-dev.txt, чтобы lock описывал
исключительно runtime.

Closes B2 from MERGE_PREP_PLAN.md.
```

---

## B1 — Вернуть явный отказ для unauthorized админ-команд (~40 мин)

### Проблема

`/set`, `/setprob`, `/clear` фильтруются `AdminOrOwner()` filter'ом. При `False` handler не вызывается → ответа нет → пользователь думает, что бот сломан.

В `main` (до рефакторинга) каждая команда отвечала текстом:
- `/set`, `/setprob`: «Команда доступна OWNER_ID и администраторам чата.»;
- `/clear`: «Недостаточно прав. Нужен OWNER_ID или права админа чата.»

### Решение

**Вариант A (выбран):** fallback-handler с тем же `Command(...)` без `AdminOrOwner()` после защищённого. Aiogram идёт по handlers Router'а в порядке регистрации; первый handler с прошедшими фильтрами выполняется. Если защищённый отказал — управление передаётся fallback'у.

Это:
- сохраняет архитектурное достижение фазы 5 (фильтры остаются декларативными);
- возвращает UX, идентичный `main`;
- не требует изменений в `AdminOrOwner` или `GroupOnly`.

**Вариант B** (вернуть проверку в handler) отвергнут: возвращает дублирование, которое мы убрали.

### Логика порядка регистрации

В одном `Router` aiogram перебирает handlers по порядку. **Проверено в исходниках aiogram** ([`aiogram/dispatcher/event/telegram.py:111-130`](https://github.com/aiogram/aiogram/blob/dev-3.x/aiogram/dispatcher/event/telegram.py)):

```python
async def trigger(self, event: TelegramObject, **kwargs: Any) -> Any:
    """
    Propagate event to handlers and stops propagation on first match.
    Handler will be called when all its filters are pass.
    """
    for handler in self.handlers:
        ...
        if result:                  # все фильтры прошли
            ...
            return await wrapped_inner(event, kwargs)
    return UNHANDLED
```

Поведение однозначно:
- handlers перебираются `for handler in self.handlers` — то есть в порядке регистрации;
- на первом, у которого `check()` вернул True (все фильтры прошли), выполняется handler и `return`;
- если ни один не прошёл — `UNHANDLED`.

Это значит: **порядок регистрации в `app/handlers/admin.py` критичен.** Защищённый `cmd_set` должен быть зарегистрирован **до** `cmd_set_denied`. Если поменять местами — fallback всегда будет выигрывать (у него меньше фильтров, проходит чаще), и админы не смогут пользоваться командами.

Для защиты от случайной перестановки — добавляется smoke-тест порядка handlers (см. секцию «Тесты» ниже).

Поведение для `/set`:

| Кто отправил | `GroupOnly` | `AdminOrOwner` | Защищённый `cmd_set` | Fallback `cmd_set_denied` |
|---|---|---|---|---|
| Админ в группе | ✅ | ✅ | **выполнен** | пропущен |
| Не-админ в группе | ✅ | ❌ | пропущен (fail filter) | ✅ → **выполнен** (отвечает «нет прав») |
| В личке | ❌ | — | пропущен | пропущен (GroupOnly fail) → молчание ✅ |

Личка остаётся тихой — там команда вообще не для контекста, ответом мог бы быть только «команда работает в группе», но это **новое** поведение, не часть требования. Для согласованности оставим silent (оригинальный `main` тоже молчал в личке для `/clear`, но болтал для `/set` — нет смысла воспроизводить эту инконсистентность).

### Изменения

#### `app/handlers/admin.py`

После каждого защищённого handler добавить fallback. Вспомогательная функция `_reply_no_permission` для общего сообщения `/set`/`/setprob`:

```diff
 router = Router(name="admin")
 logger = logging.getLogger("chat_markov")


 def _extract_command_arg(text: str) -> str:
     parts = text.split(maxsplit=1)
     return parts[1].strip() if len(parts) >= 2 else ""


+async def _reply_no_permission(message: Message, state: RuntimeState) -> None:
+    """Отказ для команд, требующих OWNER_ID или прав админа чата."""
+    await reply_humanized(
+        message,
+        "Команда доступна OWNER_ID и администраторам чата.",
+        state.typing_min_ms,
+        state.typing_max_ms,
+    )
+
+
 @router.message(Command("config"))
 async def cmd_config(message: Message, state: RuntimeState) -> None:
     ...
```

После `cmd_set`:

```diff
 @router.message(Command("set"), GroupOnly(), AdminOrOwner())
 async def cmd_set(message: Message, state: RuntimeState, settings: Settings) -> None:
     ...

+
+@router.message(Command("set"), GroupOnly())
+async def cmd_set_denied(message: Message, state: RuntimeState) -> None:
+    """Fallback: вызывается когда AdminOrOwner отказал в правах для /set."""
+    await _reply_no_permission(message, state)
+
+
 @router.message(Command("setprob"), GroupOnly(), AdminOrOwner())
 async def cmd_setprob(message: Message, state: RuntimeState, settings: Settings) -> None:
     ...
```

После `cmd_setprob`:

```diff
 @router.message(Command("setprob"), GroupOnly(), AdminOrOwner())
 async def cmd_setprob(message: Message, state: RuntimeState, settings: Settings) -> None:
     ...

+
+@router.message(Command("setprob"), GroupOnly())
+async def cmd_setprob_denied(message: Message, state: RuntimeState) -> None:
+    """Fallback: вызывается когда AdminOrOwner отказал в правах для /setprob."""
+    await _reply_no_permission(message, state)
+
+
 @router.message(Command("clear"), GroupOnly(), AdminOrOwner())
 async def cmd_clear(...):
     ...
```

После `cmd_clear` — отдельное сообщение, как в `main`:

```diff
 @router.message(Command("clear"), GroupOnly(), AdminOrOwner())
 async def cmd_clear(...):
     ...

+
+@router.message(Command("clear"), GroupOnly())
+async def cmd_clear_denied(message: Message, state: RuntimeState) -> None:
+    """Fallback: вызывается когда AdminOrOwner отказал в правах для /clear."""
+    await reply_humanized(
+        message,
+        "Недостаточно прав. Нужен OWNER_ID или права админа чата.",
+        state.typing_min_ms,
+        state.typing_max_ms,
+    )
```

### Тесты

#### `tests/test_handlers.py` — проверка текстов отказа

Три теста в `class TestAdminHandlers`:

```python
async def test_set_denied_replies_with_explanation(self) -> None:
    from app.handlers.admin import cmd_set_denied
    msg = _fake_message(text="/set foo bar")
    state = _fake_state()
    await cmd_set_denied(msg, state)
    msg.reply.assert_awaited_once()
    assert "OWNER_ID" in msg.reply.call_args[0][0]
    assert "админ" in msg.reply.call_args[0][0].lower()

async def test_setprob_denied_replies_with_explanation(self) -> None:
    from app.handlers.admin import cmd_setprob_denied
    msg = _fake_message(text="/setprob 0.5")
    state = _fake_state()
    await cmd_setprob_denied(msg, state)
    msg.reply.assert_awaited_once()
    assert "OWNER_ID" in msg.reply.call_args[0][0]

async def test_clear_denied_replies_with_explanation(self) -> None:
    from app.handlers.admin import cmd_clear_denied
    msg = _fake_message(text="/clear confirm")
    state = _fake_state()
    await cmd_clear_denied(msg, state)
    msg.reply.assert_awaited_once()
    text = msg.reply.call_args[0][0]
    assert "Недостаточно прав" in text
    # /clear должен иметь специфичное сообщение, не общее «доступна OWNER_ID...»
    assert "OWNER_ID" in text or "админ" in text.lower()
```

Эти тесты:
- покрывают фактические тексты (защищая от случайной правки);
- разделяют поведение `/clear` (специфичный текст) и `/set`/`/setprob` (общий текст);
- работают через прямой вызов handler-функций, как остальные тесты в файле.

#### `tests/test_handlers.py` — smoke-тест порядка handlers

Это **самый важный** тест в B1: он защищает от случайной перестановки handlers, которая бы сломала всю логику без видимых ошибок.

Добавить в `class TestAdminHandlers`:

```python
def test_admin_router_handler_registration_order(self) -> None:
    """
    Защита от случайной перестановки fallback-handlers.

    Порядок критичен: aiogram перебирает handlers Router'а в порядке
    регистрации и останавливается на первом, у которого все фильтры
    прошли. Если cmd_set_denied (без AdminOrOwner) окажется зарегистри-
    рован раньше cmd_set, то admin-команда никогда не выполнится для
    реальных админов — fallback всегда будет выигрывать.

    Проверяем, что для каждой защищённой команды защищённый handler
    зарегистрирован раньше fallback'а.
    """
    from app.handlers.admin import router

    # Извлекаем callbacks в порядке регистрации
    callbacks = [h.callback for h in router.message.handlers]
    names = [cb.__name__ for cb in callbacks]

    # Для каждой пары (защищённый, fallback) проверяем что защищённый
    # идёт раньше.
    pairs = [
        ("cmd_set", "cmd_set_denied"),
        ("cmd_setprob", "cmd_setprob_denied"),
        ("cmd_clear", "cmd_clear_denied"),
    ]
    for protected, fallback in pairs:
        assert protected in names, f"{protected} not registered"
        assert fallback in names, f"{fallback} not registered"
        assert names.index(protected) < names.index(fallback), (
            f"{fallback} зарегистрирован раньше {protected} — "
            f"fallback будет всегда выигрывать, защищённый handler "
            f"никогда не вызовется. Поменяйте порядок в admin.py."
        )
```

Тест работает через `router.message.handlers` (это публичное API aiogram, см. `TelegramEventObserver`); не запускает dispatcher, не требует Bot.

**Почему этого достаточно:** мы уже верифицировали в исходниках aiogram, что `for handler in self.handlers` идёт в порядке регистрации с `return` на первом match. Если порядок регистрации правильный — поведение фреймворка обеспечивает остальное. Тестировать сам цикл `for handler in...` — это тестировать чужую библиотеку.

### Что **не** тестируется (и почему)

Полный flow «aiogram dispatcher решает, какой handler вызвать на основе `AdminOrOwner=False`» — не тестируем. Это поведение фреймворка, оно покрывается:
- тестом `test_non_admin_blocked` в [`tests/test_filters.py`](tests/test_filters.py) (фильтр возвращает `False`);
- семантикой aiogram (в случае `False` filter handler пропускается, dispatcher переходит к следующему).

Мы тестируем только **что fallback-handler пишет правильный текст**. Это адекватный уровень для unit-тестов.

### Acceptance

- В `app/handlers/admin.py` есть три новых handler'а: `cmd_set_denied`, `cmd_setprob_denied`, `cmd_clear_denied`.
- Каждый зарегистрирован **после** защищённого с тем же `Command(...)` и `GroupOnly()` (проверяется новым smoke-тестом порядка).
- Четыре новых теста в `tests/test_handlers.py` зелёные (3 на тексты + 1 на порядок регистрации).
- Существующий `test_non_admin_blocked` в `test_filters.py` не сломан (он тестирует только сам фильтр).
- Полный прогон 185 тестов (181 + 4) зелёный.
- `mypy app/` зелёный.
- `ruff check app/ tests/` зелёный.

### Коммит

```
fix(admin): restore explicit denial reply for unauthorized commands

В фазе 5 admin-команды (/set, /setprob, /clear) были закрыты фильтром
AdminOrOwner. Если фильтр возвращал False, handler не вызывался —
пользователь без прав не получал никакого ответа. До рефакторинга
эти команды отвечали понятным текстом «Команда доступна OWNER_ID и
администраторам чата» и (для /clear) «Недостаточно прав. Нужен
OWNER_ID или права админа чата.»

Возвращаем исходное поведение через fallback-handlers с тем же Command
и GroupOnly, но без AdminOrOwner. Aiogram перебирает handlers Router-а
по порядку (verified по исходникам aiogram, telegram.py:111-130) и
останавливается на первом match: защищённый handler пропускается при
AdminOrOwner=False, дальше срабатывает fallback. /clear сохраняет
свой специфичный текст отказа.

В тестах добавлены четыре проверки:
- три на тексты denied-handlers;
- одна smoke-проверка порядка регистрации (защита от случайной
  перестановки, которая бы сломала всю логику без видимых ошибок).

Покрытие выросло до 185 тестов.

Closes B1 from MERGE_PREP_PLAN.md.
```

---

## Финальный чек-лист готовности к merge

После выполнения всех трёх блокеров:

| # | Проверка | Команда | Ожидание |
|---|---|---|---|
| 1 | Все тесты проходят | `python -m unittest discover tests` | `Ran 185 tests ... OK` |
| 2 | Ruff без замечаний | `python -m ruff check app/ tests/` | `All checks passed!` |
| 3 | Mypy без замечаний | `python -m mypy app/` | `Success: no issues found` |
| 4 | CI зелёный на 3.12/3.13/3.14 | GitHub Actions | все три прогонa зелёные |
| 5 | README не содержит абсолютных ссылок | `git grep "/D:/test" README.md` | пусто |
| 6 | `Dockerfile` устанавливает lock | `git grep "requirements.lock" Dockerfile` | `RUN pip install --no-cache-dir -r requirements.lock` |
| 7 | `requirements-dev.txt` ссылается на lock | первая строка файла | `-r requirements.lock` |
| 8 | `requirements.lock` имеет header | первые строки файла | комментарии с инструкцией регенерации |
| 9 | smoke-тест порядка handlers зелёный | имя теста | `test_admin_router_handler_registration_order` |
| 10 | `MERGE_PREP_PLAN.md` удалён | `ls MERGE_PREP_PLAN.md` | `not found` |
| 11 | `PROJECT_AUDIT.md` отражает закрытие блокеров | раздел 13 и таблица 15 | блокеры B1/B2/B3 помечены как закрытые с указанием коммитов |

После пунктов 1–10 — финальный коммит, который:

1. **Обновляет `PROJECT_AUDIT.md`** — не просто заменяет «не мёржить» на «можно», а **фиксирует факт закрытия каждого блокера**:
   - в разделе 13.2 каждая строка таблицы блокеров получает столбец «Закрыто в коммите» с SHA;
   - раздел 13.4 переписывается: «Блокеры B1/B2/B3 закрыты коммитами `<sha1>`, `<sha2>`, `<sha3>`. CI зелёный, 185 тестов проходят, ruff/mypy чистые. Ветка готова к merge.»;
   - в разделе 15 (сводная таблица) строки P1-A/P1-B/P1-C получают пометку «✅ закрыто в `<sha>`».
2. **Удаляет `MERGE_PREP_PLAN.md`** (этот файл).
3. **Сообщение коммита:**
   ```
   docs(audit): mark blockers B1/B2/B3 as closed; ready to merge

   Все три P1-блокера из второй редакции PROJECT_AUDIT.md закрыты:
   - B1 (UX-регрессия в админ-командах) — коммит <sha1>
   - B2 (requirements.lock не используется) — коммит <sha2>
   - B3 (README absolute links) — коммит <sha3>

   CI зелёный на 3.12/3.13/3.14, 185 тестов проходят,
   ruff и mypy без замечаний.

   Ветка refactor/structure готова к merge в main.
   MERGE_PREP_PLAN.md удалён как выполненный.
   ```

---

## Стратегия merge

После того как CI зелёный и чек-лист закрыт:

1. **PR в `main`** — описание содержит ссылки на финальный `PROJECT_AUDIT.md` и краткий список фаз.
2. **Тип merge — merge commit без squash.** 28+ маленьких коммитов фаз дают полезный bisect-friendly след; squash сжимает их в один и стирает контекст.
3. **После merge:**
   - Включить branch protection для `main` (require PR + CI).
   - Создать issues для оставшегося P2-долга из раздела 14 `PROJECT_AUDIT.md` (миграционный hardening, реестр настроек, Dockerfile-hardening, README-разделы, smoke-тест на wiring).
   - Создать GitHub Release / тег `v1.0.0-refactor` для фиксации точки.

---

**Дата создания:** 2026-05-08, после второй редакции `PROJECT_AUDIT.md`.
**Автор плана:** проектная сессия Claude (вторая ревизия).
**Удалить файл:** после выполнения всех трёх блокеров и финального коммита.
