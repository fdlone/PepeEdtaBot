## Why

O22, группа 4 (решение владельца 2026-09-11, вариант 1). Перепись 2.0 нашла
скрытую связь: `hot_ngram_seed_chance` — не только шанс легаси-затравки
одной попытки, но и **гейт записи** горячего окна на пути обучения; обнулить
её значило перестать писать окно и выключить hot-маршрут данными.
Легаси-розыгрыш при включённом маршруте (дефолт с 11.09) **до прогулки не
доходит**: генератор сбрасывает `seed` после первой попытки, а первые попытки
занимает маршрут. Замер `eval_2026-09-11_legacy-seed-check_noctx.md` (арм с
розыгрышем 0 против дефолта 0.25, noctx, копия 10.09): все дельты ровно
0.000, побайтно те же ответы.

## What Changes

- Гейт записи горячего окна — `hot_ngram_slot_ratio > 0` (единственный
  читатель канала).
- Удалены `hot_ngram_seed_chance` (реестр, `RuntimeTunables`, `.env.example`,
  краткий `/config` — строка «подхват мемов чата» показывает
  `hot_ngram_slot_ratio`), `ReplyPipeline._hot_ngram_seed`, поле
  `GenerationRequest.seed` и его единственный потребитель; в харнессе —
  `draw_hot_seed`, `HOT_SEED_RNG_OFFSET`, поле `seed_drawn` записи
  (coverage гейта `l1_hot_channel` и раньше читал `start_source`).
- **Поведение не меняется:** ctx — `generation_hash` совпал на обоих снимках
  (`hash-log.md`); noctx — замер выше.

## Capabilities

### New Capabilities

_нет_

### Modified Capabilities
- `generation-hot-route`: ратио маршрута — также гейт записи горячего окна.
- `generation-eval`: требование о моделировании легаси-розыгрыша в харнессе
  снято.

## Impact

- `app/services/reply_pipeline.py`, `app/core/response_generator.py`,
  `app/config/{registry,settings}.py`, `.env.example`,
  `app/presentation/bot_messages.py`.
- `tools/eval/{run,metrics,report}.py`, `tools/eval/README.md`,
  `tools/eval/matrix_l1_grid.yaml` (комментарий), исторический
  `matrix_legacy_seed_check.yaml`; тесты конвейера, хендлеров, гигиены
  замеров и протокола; карта §1.1/§2.1, пайплайн §3.2.
