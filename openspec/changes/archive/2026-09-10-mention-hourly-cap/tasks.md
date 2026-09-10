## 1. Гарды до починки

- [x] 1.1 `tests/test_reply_pipeline.py`: обращение при заполненном окне `recent_mention_reply_times[1]` (20 меток за последний час) понижается (`address_reply=False`), `telemetry.mentions_capped == 1`; при `mention_max_per_hour=0` не понижается и `mentions_capped == 0`; обращение, погашенное кулдауном, срез не считает. Проверка: тесты **падают** на текущем коде (`TypeError` по неизвестному полю / атрибуту)
- [x] 1.2 `tests/test_runtime_config.py` (или соседний тест состояния): `note_mention_reply` пишет метку в окно чата и режет старше часа; `forget_chat` забывает окно. Проверка: тест **падает** на текущем коде

## 2. Реализация

- [x] 2.1 Ручка `mention_max_per_hour` (`MENTION_MAX_PER_HOUR`, дефолт 20, `_int_in_range(0, 1000)`) в `registry.py` рядом с `mention_cooldown_sec`, поле в `RuntimeTunables` (`settings.py`), строка в `.env.example` с оговоркой про 0. Проверка: `tests/test_registry.py` и `tests/test_runtime_config.py` зелёные, `/set help` через `_DIALOGUE_HELP_KNOBS` показывает ручку (тест в `test_bot_messages.py`)
- [x] 2.2 `RuntimeState.recent_mention_reply_times`, заполнение и обрезка в `note_mention_reply`, `forget_chat`. Проверка: тест 1.2 зелёный
- [x] 2.3 Гейт в `observe` после кулдауна обращений: `within_hourly_cap(state.recent_mention_reply_times.get(chat_id, ()), now, state.mention_max_per_hour)`; при срезе — `telemetry.note_mention_capped()`. Проверка: тест 1.1 зелёный; существующие тесты кулдауна и знаменателя обращений не тронуты и зелёные
- [x] 2.4 Телеметрия: счётчик `mentions_capped`, в `snapshot()` ключ `mentions_capped` (число, не `None`); строка `/stats` дополнена «срезано пределом K» (design Р5). Проверка: тест `test_bot_messages.py` на строку «обращений» дополнен и зелёный; `tests/test_generation_telemetry.py` (если есть тест снапшота) — ключ присутствует

## 3. Документация

- [x] 3.1 `docs/GENERATION_PIPELINE.md` §1.2: абзац «Mention-cooldown» дополнен вторым пределом; `docs/GENERATION_MAP.md` §2.1 строка политики «отвечать ли» — добавить `mention_max_per_hour`. Проверка: `tests/test_doc_pointers.py` зелёный
- [x] 3.2 `docs/OPEN.md`: запись O15 снята из таблицы и раздела; `docs/CLOSED.md`: строка «O15 закрыт» с цифрами окна (29 / 100% / пик 21) и формой решения. Проверка: `grep O15 docs/OPEN.md` пусто, кроме ссылок на историю

## 4. Приёмка

- [x] 4.1 `ruff check`, `mypy app/`, весь `unittest` зелёные; покрытие не ниже `fail_under = 87`. Проверка: команды в `.venv`, вывод в отчёте
- [x] 4.2 `tests/test_generation_hash_baseline.py` зелёный — хеш не сдвинулся, как заявлено. Проверка: расхождение = стоп и отчёт владельцу
