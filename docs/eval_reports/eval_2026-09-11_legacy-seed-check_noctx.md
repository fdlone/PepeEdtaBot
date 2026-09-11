# Eval report 2026-09-11 snapshot=legacy-seed-check prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `08c433f`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **CL0**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | CL0 | Δ CL0 vs C0 |
|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.999 [0.998, 1.000] | 0.999 [0.998, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.464 [10.251, 10.688] | 10.464 [10.251, 10.688] | 0.000 [0.000, 0.000] |
| unique_token_ratio | 0.987 [0.986, 0.989] | 0.987 [0.986, 0.989] | 0.000 [0.000, 0.000] |
| exact_context_copy_rate | 0.003 [0.001, 0.006] | 0.003 [0.001, 0.006] | 0.000 [0.000, 0.000] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.086 [0.076, 0.095] | 0.086 [0.076, 0.095] | 0.000 [0.000, 0.000] |
| context_affinity_without_copy | 0.085 [0.076, 0.095] | 0.085 [0.076, 0.095] | 0.000 [0.000, 0.000] |
| seeded_present_rate | insufficient data | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — |
| historical_meme_rate | 0.160 [0.123, 0.197] | 0.160 [0.123, 0.197] | 0.000 [0.000, 0.000] |
| structural_pool_ecb | 4.669 [4.645, 4.694] | 4.669 [4.645, 4.694] | 0.000 [0.000, 0.000] |
| structural_window_escape | 3.327 [3.281, 3.369] | 3.327 [3.281, 3.369] | 0.000 [0.000, 0.000] |

C0: distinct-2 = 0.640 (basis 14196), distinct-3 = 0.785 (basis 12696) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 17.3/27.8 ms; cache_hit_rate: 32%; mean normalized entropy: 0.244 (branching 3.40); mean applied temperature: 2.62; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; hot-ngram seeds: 1845 draws, empty 0%; storage_delta: n/a.

CL0: distinct-2 = 0.640 (basis 14196), distinct-3 = 0.785 (basis 12696) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 17.3/26.6 ms; cache_hit_rate: 32%; mean normalized entropy: 0.244 (branching 3.40); mean applied temperature: 2.62; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; hot-ngram seeds: 1500 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.069 [0.058, 0.080] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.091 [0.076, 0.109] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.141 [0.108, 0.175] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.048 [0.040, 0.057] |
| CL0 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.069 [0.058, 0.080] |
| CL0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.091 [0.076, 0.109] |
| CL0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.141 [0.108, 0.175] |
| CL0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.048 [0.040, 0.057] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.161 [0.153, 0.169] | 0.613 [0.587, 0.638] | 0.277 [0.248, 0.307] | 0.029 [0.015, 0.046] | 0.012 [0.000, 0.027] | 18.3 / 17.9 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.065 [0.061, 0.070] | 0.307 [0.286, 0.329] | 0.260 [0.223, 0.299] | 0.018 [0.011, 0.026] | 0.008 [0.000, 0.025] | 16.3 / 19.0 | 0 |
| C0 | extension | 1500 | 0.085 [0.079, 0.091] | 0.369 [0.345, 0.393] | 0.253 [0.219, 0.291] | 0.031 [0.014, 0.054] | 0.000 [0.000, 0.000] | 19.9 / 17.1 | 0 |
| C0 | hot | 1500 | 0.400 [0.400, 0.400] | 1.000 [1.000, 1.000] | 0.414 [0.391, 0.439] | 0.027 [0.021, 0.034] | 0.000 [0.000, 0.000] | 18.1 / — | 0 |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | phrase | 1500 | 0.289 [0.279, 0.299] | 0.758 [0.735, 0.783] | 0.320 [0.292, 0.345] | 0.261 [0.234, 0.289] | 0.003 [0.000, 0.008] | 18.0 / 18.6 | F4_stale 7, F6_structural_repeat 1 |
| CL0 | vanilla | 1500 | 0.161 [0.153, 0.169] | 0.613 [0.587, 0.638] | 0.277 [0.248, 0.307] | 0.029 [0.015, 0.046] | 0.012 [0.000, 0.027] | 18.1 / 17.3 | 0 |
| CL0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| CL0 | mutated | 1500 | 0.065 [0.061, 0.070] | 0.307 [0.286, 0.329] | 0.260 [0.223, 0.299] | 0.018 [0.011, 0.026] | 0.008 [0.000, 0.025] | 15.6 / 18.8 | 0 |
| CL0 | extension | 1500 | 0.085 [0.079, 0.091] | 0.369 [0.345, 0.393] | 0.253 [0.219, 0.291] | 0.031 [0.014, 0.054] | 0.000 [0.000, 0.000] | 19.5 / 16.8 | 0 |
| CL0 | hot | 1500 | 0.400 [0.400, 0.400] | 1.000 [1.000, 1.000] | 0.414 [0.391, 0.439] | 0.027 [0.021, 0.034] | 0.000 [0.000, 0.000] | 17.8 / — | 0 |
| CL0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| CL0 | phrase | 1500 | 0.289 [0.279, 0.299] | 0.758 [0.735, 0.783] | 0.320 [0.292, 0.345] | 0.261 [0.234, 0.289] | 0.003 [0.000, 0.008] | 17.6 / 18.5 | F4_stale 7, F6_structural_repeat 1 |

## Gates

- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate**: insufficient data — no route-gate arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 3.327 [3.281, 3.369] (min 2.0); pool ECB 4.669 [4.645, 4.694] (floor 4.0); pool ECB share 0.934 [0.929, 0.939] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:1%, 2:17%, 3:39%, 4:33%, 5:10%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[CL0]**: insufficient data — window escape 3.327 [3.281, 3.369] (min 2.0); pool ECB 4.669 [4.645, 4.694] (floor 4.0); pool ECB share 0.934 [0.929, 0.939] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:1%, 2:17%, 3:39%, 4:33%, 5:10%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **performance.generation_p95**: pass — C0 p95 = 27.8 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 15/18 memes reproduced (83%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[CL0]**: pass — 15/18 memes reproduced (83%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
