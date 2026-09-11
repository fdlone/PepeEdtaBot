# Tasks — narrow-knob-domains (O22, группа 3)

## 1. Реестр

- [x] 1.1 Домены четырёх `*_slot_ratio` → 0..0.5, `hot_ngram_min_count` → 1..100; комментарии с причиной; `.env.example`
- [x] 1.2 Инвариант «бонус ≤ запас» в `validate_cross_fields`; тесты: на границе принимается, выше отвергается, поднятый запас поднимает потолок; `BOUNDED` в тестах реестра

## 2. Приёмка

- [x] 2.1 `ruff`, `mypy app/`, полный `unittest`; `generation_hash` на обоих снимках не сдвинут (`hash-log.md`)
- [x] 2.2 Карта §2.1 — семь строк с новыми доменами; спека `runtime-knob-validation` синхронизирована
