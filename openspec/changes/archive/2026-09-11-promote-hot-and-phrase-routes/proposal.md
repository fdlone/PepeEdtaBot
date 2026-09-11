## Why

Два гейта закрыты целиком 2026-09-11 (раунды связности зачтены с явным
отступлением решением владельца, `eval_2026-09-11_{l1-route-v5,route-selection}-round-verdict.md`):

- **hot-маршрут** (M3R-230, O18): `l1_hot_channel` — coverage 30.5%, meme
  rate +0.299\*, escape окна +0.091\*, copy/repetition/ECB на месте
  (`eval_2026-09-02_l1-route-verdict.md`, арм C7r40 = `hot_ngram_slot_ratio`
  0.4 при порогах 2 / 0.25), связность +3.3 п.п. к C0;
- **пара «фразовый маршрут + бонус различности»** (M3R-210/220, O20):
  `route_gate` в ctx — одиночные входы −15.9 п.п.\*, affinity без копий
  +0.040\*, copy −4.9 п.п.\*, ECB +0.163\*, p95 35 мс
  (`eval_2026-09-11_route-selection-verdict.md`, арм C11s), связность
  −6.7 п.п. при планке −10; в noctx must-improve недостижим по построению,
  остальное взято.

**Решение владельца 2026-09-11: включить всё, пару — в обоих режимах.**
Дефолты становятся измеренными конфигурациями армов; по §2 CLAUDE.md
хеш-ломающие правки собираются в один пакет с одной перезаморозкой.

## What Changes

- Дефолты реестра (семь ручек, все уже существуют):
  `hot_ngram_slot_ratio` 0 → **0.4**, `hot_ngram_min_count` 3 → **2**,
  `hot_ngram_recency_share` 0.5 → **0.25**, `phrase_slot_ratio` 0 → **0.4**,
  `phrase_min_count` 3 → **2**, `selection_diversity_bonus` 0 → **0.2**
  (noctx-ручка уже 0.2 с O19), `verbatim_recognized_unit` false → **true**.
  `.env.example` синхронно.
- Одна перезаморозка `tools/generation_hash_baseline.json` (синтетика) с
  причиной; сдвиг на прод-копии записывается в `hash-log.md`. Метрический
  базлайн `tools/generation_baseline.json` (`eval_generation`, читает
  дефолты реестра) перезаписывается тем же пакетом.
- Приёмочный замер **совместной** конфигурации до перезаморозки: пара и
  hot-маршрут в noctx вместе не измерялись (hot занимает 2 слота, фраза 2,
  обходу остаётся 1 при пуле 5). Арм «новые дефолты» против «прежних
  дефолтов» в обоих режимах; условия приёмки — инвариант ECB пула ≥ 4.0
  (§5 CLAUDE.md), copy/repetition не растут значимо, affinity без копий не
  падает значимо, p95 ≤ 150. Значимое ухудшение — стоп и отчёт, не
  промоушен.
- Документы: спеки четырёх capability (дефолты), OPEN (O18, O20 → CLOSED),
  роадмап, карта §2.1, пайплайн, README харнесса (C0 = дефолты реестра
  теперь несёт маршруты и бонус — разрыв сопоставимости всех будущих
  отчётов с прежними, явно).
- Прод: включение — рестартом; после первого дня проверить `/stats`
  (`route_*` для `hot`/`phrase`, `phrase_empty_rate`, `hot_ngram_empty_rate`)
  — действие владельца, вне заявки.

## Capabilities

### New Capabilities

_нет_

### Modified Capabilities
- `generation-hot-route`: дефолт ратио 0.4 при порогах 2 / 0.25; байт-в-байт с
  прежним поведением — при ратио 0.
- `generation-phrase-route`: дефолты 0.4 / порог 2; байт-в-байт — при ратио 0.
- `generation-selection-window`: ctx-бонус по умолчанию 0.2.
- `generation-verbatim-guard`: гард включён по умолчанию; требование о
  выключенном гарде сохраняется как свойство значения 0/false.

## Impact

- `app/config/registry.py`, `.env.example`, `tests/test_selection_knobs.py`,
  `tools/generation_hash_baseline.json`, `tools/generation_baseline.json`,
  `tools/eval/matrix_promotion_check.yaml` (приёмочный замер), README.
- Все прежние eval-отчёты сопоставимы между собой, но не с отчётами после
  этой даты: C0 сменил состав. Число версии набора промптов не меняется
  (набор тот же), меняется базлайн — фиксируется в README и CLOSED.
