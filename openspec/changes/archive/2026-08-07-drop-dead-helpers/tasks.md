## 1. Проверка перед удалением

- [x] 1.1 Убедиться, что `extract_best_seed` не вызывается: `grep -rn "extract_best_seed" app/ tests/ tools/ docs/ main.py README.md .env.example` даёт только строку определения в `app/core/markov.py`
- [x] 1.2 Убедиться, что `runtime_config.parse_bool` не вызывается из `app/`: единственные вхождения вне определения — импорт и один тест-метод в `tests/test_runtime_config.py`
- [x] 1.3 Убедиться, что `registry._parse_bool` остаётся покрытым: `tests/test_registry.py` тестирует и truthy/falsy, и отказ на мусоре

## 2. Удаление

- [x] 2.1 Удалить функцию `extract_best_seed` из `app/core/markov.py`; проверить, что импортированные ради неё имена (`PUNCT_SET`) продолжают использоваться в модуле, иначе снять и импорт — `PUNCT_SET` определён в самом модуле и используется ещё 9 раз, снимать нечего
- [x] 2.2 Удалить функцию `parse_bool` из `app/config/runtime_config.py`; снять `_parse_bool` из списка импортов, если после удаления он больше не нужен — снят, других потребителей в модуле нет
- [x] 2.3 Удалить `test_parse_bool_accepts_supported_values` и импорт `parse_bool` в `tests/test_runtime_config.py`

## 3. Правка докстрингов

- [x] 3.1 В `app/handlers/_helpers.py` исправить докстринг `reply_humanized_state`: ссылка на `reply_humanized` заменяется на существующую `reply_humanized_sequence`
- [x] 3.2 Там же исправить докстринг `reply_humanized_sequence_state` по той же причине
- [x] 3.3 **Отклонено при реализации.** Задача предлагала переключить `reply_humanized_state` на прямой вызов `reply_humanized_sequence`, убрав переход через `reply_humanized_sequence_state`. Переход действительно уходит, но вместе с ним извлечение тройки `typing_min_ms` / `typing_max_ms` / `typing_per_char_ms` из `runtime_state` дублируется в двух функциях вместо одной — сложности становится больше, а не меньше. Цепочка оставлена как была; вместо этого в докстринге `reply_humanized_sequence_state` зафиксировано, что она и есть единственная точка разрешения ручек.

## 4. Проверка

- [x] 4.1 `python -m ruff check app/ tests/ tools/ main.py` — чисто (All checks passed)
- [x] 4.2 `python -m mypy app/` — чисто (no issues in 71 source files)
- [x] 4.3 `python -m coverage run -m pytest && python -m coverage report` — 954 теста и 4574 сабтеста зелёные, покрытие 94% при пороге 87
- [x] 4.4 Убедиться, что кроме `tests/test_runtime_config.py` ни один тест не пришлось править: правка любого другого теста означает, что удалённое не было мёртвым, и задачу надо пересмотреть — `git diff --name-only -- tests/` даёт ровно один файл
