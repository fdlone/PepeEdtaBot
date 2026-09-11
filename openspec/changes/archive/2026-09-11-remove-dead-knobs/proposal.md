## Why

O22, группа 1 (решение владельца 2026-09-11 по переписи 2.0,
`eval_2026-09-11_knob-census.md`): четыре ручки и одна константа мертвы по
коду или инертны по цифрам, а код за ними — около двухсот строк, из которых
тень order-4 ещё и работает на горячем пути каждой генерации.

- `recent_reply_penalty_strength` — штраф за триграммы последних 20 ответов:
  inert в eval по построению, в проде с тёплой памятью среднее 0.004 при
  окне отбора 0.3 (map §3.11) — отбор не меняет никогда.
- `fuzzy_context_casefold` — casefold-тир каскада контекстных стартов и
  casefold-половина `ContextStateMatcher`: корпус учится в нижнем регистре,
  ярус не даёт стартов сверх exact; перепись — inert в обоих режимах.
- `markov_shadow_order4_enabled` — теневой селектор фазы 7: фаза закрыта
  2026-08-12 (0 выборов на 5937 шагов), замер после этого ничего не питал, а
  индекс окна строился и пополнялся на каждом сообщении.
- `auto_capitalize_replies` — выключена решением владельца, inert.
- `JUMP_MIN_TOKENS_BETWEEN` — условие никогда не срабатывает при
  `JUMP_MAX_PER_REPLY = 1`.

## What Changes

- Удалены ручки (реестр, `RuntimeTunables`, `.env.example`, `/config`) и код
  за ними: функции штрафа недавних в скорере и его слагаемое в
  `CandidateScore`; casefold-строители стартов, casefold-индекс матчера и
  счётчик casefold-матчей в трассе; модуль `shadow_order.py`, оконный индекс
  в `LearningService`, счётчики и строка `/stats`, метод протокола, гейт
  `phase7_order4` в отчёте и блок порогов (фаза закрыта, инструмент
  измерения снят); `capitalize_reply_sentences`; константа прыжка и
  бухгалтерия `last_jump_end`.
- Харнессы: `eval_generation` и `eval_prod` больше не считают casefold-матчи и
  тень; метрические базлайны `eval_generation` перегенерированы (у них
  сменился состав ключей и синтетический прогон вёл память ответов).
- **Поведение генерации не меняется:** `generation_hash` совпал с базлайном
  промоушена на обоих снимках (`hash-log.md`).

## Capabilities

### New Capabilities

_нет_

### Modified Capabilities
- `generation-telemetry`: требование о теневом селекторе order-4 снято.
- `generation-eval`: требование о гейте фазы 7 снято (фаза закрыта,
  инструмент удалён).

## Impact

- `app/core/{candidate_scorer,response_generator,markov,context_state_matcher,
  text,gen_trace_log,generation_telemetry}.py`, `app/services/learning_service.py`,
  `app/config/{registry,settings}.py`, `app/presentation/bot_messages.py`,
  `.env.example`; удалён `app/core/shadow_order.py`.
- `tools/eval_generation.py`, `tools/eval_prod.py`, `tools/eval/{report,run,
  knob_census}.py`, `tools/eval/eval_thresholds.yaml`, базлайны
  `tools/generation_baseline*.json`.
- Тесты удалённого (около 25) сняты; документы: карта §2.1/§2.2/§2.3/§3.11,
  пайплайн §4/§5/§6, роадмап M3R-150, STATUS (фаза 1), OPEN (отложенное).
