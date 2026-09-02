## Why

M3R-110 (состав пула в ctx-режиме) требует свипа трёх осей:
`reply_context_start_bias`, `context_anchor_splice_probability` и
`GENERATION_ATTEMPTS_WITH_CONTEXT`. Первые две — ручки реестра, третья —
константа модуля `response_generator.py` (`= 5`), а матрица `tools/eval`
переопределяет только ручки реестра. Свип третьей оси невозможен без ручки, а
именно она отвечает за «контекст молча выключается с шестой попытки» — 37% ctx-ответов
на копии 13.08, 20% в живом проде 01.09 (map §1.3, M3R-141).

## What Changes

- Константа становится ручкой реестра `generation_attempts_with_context`
  (env `GENERATION_ATTEMPTS_WITH_CONTEXT`, дефолт **5** — прежнее значение,
  диапазон 0..10 = бюджет попыток). Ноль — контекст не подаётся ни одной
  попытке (диагностический режим, эквивалент noctx на ctx-входе).
- `ResponseGenerator` читает значение из `runtime_state`; константа модуля
  остаётся документированным дефолтом и источником для тестов.
- Поведение при дефолте не меняется: `generation_hash` без сдвига на обоих
  снимках (ожидание записывается до правок).
- Ручка доступна `/set` (чат-скоуп), `/config`, матрицам eval.

## Capabilities

### New Capabilities
- `generation-context-attempts`: сколько попыток сборки пула получают
  контекст — ручка, а не константа; дефолт равен прежней константе.

### Modified Capabilities
<!-- нет -->

## Impact

- `app/config/registry.py`, `app/config/settings.py`, `.env.example`,
  `app/core/response_generator.py`, тесты.
- `docs/GENERATION_MAP.md` §2.1/§2.2, `docs/GENERATION_PIPELINE.md` §4.
