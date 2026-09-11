# Tasks — split-diversity-bonus-by-mode (O19, вариант 2)

## 1. До правок

- [x] 1.1 `hash-log.md`: хеш на синтетике и прод-копии 10.09 до правок; ожидание «сдвига нет» (гард генерирует только с контекстом, ctx-ручка 0)

## 2. Ручка и выбор по режиму

- [x] 2.1 `selection_diversity_bonus_noctx` в реестре (0..1, дефолт 0.2), `RuntimeTunables`, `.env.example`; фикстуры тестов; `tests/test_selection_knobs.py`: дефолты обеих ручек
- [x] 2.2 `generate_with_result`: выбор ручки по `request.context_tokens` (design D2); тесты уровня `ResponseGenerator`: с контекстом бонус ctx-ручки (0 → скоры не тронуты), без контекста — noctx-ручки (0.2 → компонент виден в скоре)

## 3. Приёмка

- [x] 3.1 `ruff`, `mypy app/`, полный `unittest`, покрытие ≥ храповика
- [x] 3.2 `generation_hash` после правок на обоих снимках, факт в `hash-log.md`
- [x] 3.3 Документы: OPEN (O19 → CLOSED с цифрами и решением), роадмап M3R-100, карта §2.1 (обе ручки), пайплайн §4 (5d), README `tools/eval` (C0 в noctx несёт бонус 0.2 — разрыв сопоставимости)
