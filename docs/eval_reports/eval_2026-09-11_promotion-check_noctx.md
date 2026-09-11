# Eval report 2026-09-11 snapshot=promotion-check prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `3d04dce`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **CP**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | CP | Δ CP vs C0 |
|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 1.000 [0.999, 1.000] | 0.999 [0.998, 1.000] | -0.001 [-0.001, -0.000] * |
| mean_response_length | 10.685 [10.460, 10.933] | 10.464 [10.251, 10.688] | -0.221 [-0.528, 0.098] |
| unique_token_ratio | 0.986 [0.984, 0.988] | 0.987 [0.986, 0.989] | 0.001 [-0.001, 0.003] |
| exact_context_copy_rate | 0.003 [0.001, 0.005] | 0.003 [0.001, 0.006] | 0.001 [-0.003, 0.005] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.034 [0.028, 0.041] | 0.086 [0.076, 0.095] | 0.051 [0.041, 0.062] * |
| context_affinity_without_copy | 0.035 [0.029, 0.041] | 0.085 [0.076, 0.095] | 0.051 [0.040, 0.062] * |
| seeded_present_rate | insufficient data | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — |
| historical_meme_rate | 0.083 [0.056, 0.112] | 0.160 [0.123, 0.197] | 0.077 [0.029, 0.125] * |
| structural_pool_ecb | 4.558 [4.527, 4.589] | 4.669 [4.645, 4.694] | 0.111 [0.070, 0.149] * |
| structural_window_escape | 3.298 [3.251, 3.346] | 3.327 [3.281, 3.369] | 0.029 [-0.037, 0.093] |

C0: distinct-2 = 0.710 (basis 14527), distinct-3 = 0.861 (basis 13027) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.1/22.6 ms; cache_hit_rate: 27%; mean normalized entropy: 0.248 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

CP: distinct-2 = 0.640 (basis 14196), distinct-3 = 0.785 (basis 12696) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 16.1/24.7 ms; cache_hit_rate: 32%; mean normalized entropy: 0.244 (branching 3.40); mean applied temperature: 2.62; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 1845 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.027 [0.021, 0.033] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.033 [0.027, 0.041] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.060 [0.038, 0.086] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.021 [0.017, 0.025] |
| CP | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.069 [0.058, 0.080] |
| CP | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.091 [0.076, 0.109] |
| CP | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.141 [0.108, 0.175] |
| CP | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.048 [0.040, 0.057] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.603 [0.592, 0.613] | 0.992 [0.987, 0.996] | 0.549 [0.524, 0.575] | 0.030 [0.023, 0.039] | 0.005 [0.001, 0.010] | 14.7 / 14.3 | F4_stale 1 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.096 [0.089, 0.102] | 0.426 [0.400, 0.453] | 0.280 [0.246, 0.313] | 0.023 [0.013, 0.037] | 0.000 [0.000, 0.000] | 13.1 / 15.9 | 0 |
| C0 | extension | 1500 | 0.301 [0.292, 0.311] | 0.847 [0.829, 0.865] | 0.397 [0.370, 0.423] | 0.045 [0.033, 0.059] | 0.000 [0.000, 0.000] | 15.0 / 12.9 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| CP | vanilla | 1500 | 0.161 [0.153, 0.169] | 0.613 [0.587, 0.638] | 0.277 [0.248, 0.307] | 0.029 [0.015, 0.046] | 0.012 [0.000, 0.027] | 16.8 / 16.2 | 0 |
| CP | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| CP | mutated | 1500 | 0.065 [0.061, 0.070] | 0.307 [0.286, 0.329] | 0.260 [0.223, 0.299] | 0.018 [0.011, 0.026] | 0.008 [0.000, 0.025] | 14.7 / 17.4 | 0 |
| CP | extension | 1500 | 0.085 [0.079, 0.091] | 0.369 [0.345, 0.393] | 0.253 [0.219, 0.291] | 0.031 [0.014, 0.054] | 0.000 [0.000, 0.000] | 18.1 / 15.7 | 0 |
| CP | hot | 1500 | 0.400 [0.400, 0.400] | 1.000 [1.000, 1.000] | 0.414 [0.391, 0.439] | 0.027 [0.021, 0.034] | 0.000 [0.000, 0.000] | 16.6 / — | 0 |
| CP | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| CP | phrase | 1500 | 0.289 [0.279, 0.299] | 0.758 [0.735, 0.783] | 0.320 [0.292, 0.345] | 0.261 [0.234, 0.289] | 0.003 [0.000, 0.008] | 16.4 / 17.1 | F4_stale 7, F6_structural_repeat 1 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate**: insufficient data — no route-gate arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 3.298 [3.251, 3.346] (min 2.0); pool ECB 4.558 [4.527, 4.589] (floor 4.0); pool ECB share 0.912 [0.905, 0.918] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:2%, 2:19%, 3:38%, 4:31%, 5:10%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[CP]**: insufficient data — window escape 3.327 [3.281, 3.369] (min 2.0); pool ECB 4.669 [4.645, 4.694] (floor 4.0); pool ECB share 0.934 [0.929, 0.939] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:1%, 2:17%, 3:39%, 4:33%, 5:10%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1328 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 22.6 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 8/18 memes reproduced (44%); C0 8/18 (44%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[CP]**: pass — 15/18 memes reproduced (83%); C0 8/18 (44%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
