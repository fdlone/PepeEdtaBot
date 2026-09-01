# Eval report 2026-09-02 snapshot=l1-route prompts=308b7deaea0f seeds=42,1337,2026 mode=noctx

Revision: `c16cc98`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C7r20**: 1500 generations
- **C7r40**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C7r20 | Δ C7r20 vs C0 | C7r40 | Δ C7r40 vs C0 |
|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | -0.000 [-0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.352 [10.138, 10.603] | 10.382 [10.149, 10.629] | 0.030 [-0.251, 0.334] | 10.527 [10.311, 10.747] | 0.175 [-0.124, 0.465] |
| unique_token_ratio | 0.988 [0.986, 0.989] | 0.987 [0.986, 0.989] | -0.000 [-0.002, 0.002] | 0.988 [0.986, 0.989] | 0.000 [-0.002, 0.002] |
| exact_context_copy_rate | 0.004 [0.001, 0.007] | 0.007 [0.003, 0.011] | 0.003 [-0.002, 0.007] | 0.003 [0.001, 0.007] | -0.001 [-0.005, 0.003] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.023 [0.019, 0.027] | 0.026 [0.021, 0.031] | 0.004 [-0.003, 0.010] | 0.021 [0.017, 0.025] | -0.002 [-0.007, 0.003] |
| context_affinity_without_copy | 0.023 [0.019, 0.027] | 0.025 [0.020, 0.030] | 0.003 [-0.003, 0.009] | 0.020 [0.017, 0.024] | -0.002 [-0.008, 0.003] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.019 [0.008, 0.035] | 0.013 [0.003, 0.027] | -0.005 [-0.021, 0.013] | 0.021 [0.008, 0.037] | 0.003 [-0.016, 0.024] |
| structural_pool_ecb | 4.536 [4.507, 4.565] | 4.545 [4.514, 4.575] | 0.009 [-0.033, 0.049] | 4.515 [4.485, 4.545] | -0.021 [-0.064, 0.017] |
| structural_window_escape | 2.863 [2.813, 2.914] | 2.943 [2.895, 2.991] | 0.081 [0.009, 0.153] * | 2.977 [2.925, 3.027] | 0.114 [0.035, 0.185] * |

C0: distinct-2 = 0.703 (basis 14028), distinct-3 = 0.857 (basis 12528) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 16.0/26.5 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.52); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C7r20: distinct-2 = 0.681 (basis 14073), distinct-3 = 0.829 (basis 12573) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 16.0/25.9 ms; cache_hit_rate: 29%; mean normalized entropy: 0.246 (branching 3.39); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 1845 draws, empty 0%; storage_delta: n/a.

C7r40: distinct-2 = 0.645 (basis 14291), distinct-3 = 0.785 (basis 12791) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.6/23.9 ms; cache_hit_rate: 34%; mean normalized entropy: 0.252 (branching 3.41); mean applied temperature: 2.67; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 1845 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.031] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.017 [0.013, 0.022] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.040] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.031] |
| C7r20 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.024 [0.017, 0.032] |
| C7r20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.013 [0.003, 0.027] | 0.000 [0.000, 0.000] | 0.017 [0.012, 0.024] |
| C7r20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.044 [0.025, 0.065] |
| C7r20 | topical | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.022 [0.017, 0.027] |
| C7r40 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.030] |
| C7r40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.016 [0.011, 0.021] |
| C7r40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.021 [0.009, 0.035] |
| C7r40 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.023 [0.020, 0.027] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.590 [0.579, 0.602] | 0.989 [0.984, 0.994] | 0.559 [0.532, 0.582] | 0.022 [0.017, 0.028] | 0.007 [0.002, 0.014] | 16.6 / 17.0 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.100 [0.094, 0.106] | 0.447 [0.423, 0.472] | 0.273 [0.242, 0.307] | 0.028 [0.019, 0.038] | 0.000 [0.000, 0.000] | 14.8 / 18.1 | 0 |
| C0 | extension | 1500 | 0.310 [0.299, 0.321] | 0.843 [0.823, 0.863] | 0.386 [0.358, 0.413] | 0.022 [0.015, 0.030] | 0.000 [0.000, 0.000] | 16.9 / 15.2 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | vanilla | 1500 | 0.466 [0.457, 0.475] | 0.971 [0.962, 0.979] | 0.451 [0.424, 0.476] | 0.021 [0.015, 0.028] | 0.012 [0.005, 0.021] | 16.6 / 16.8 | 0 |
| C7r20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.437 [0.411, 0.460] | 0.272 [0.240, 0.307] | 0.033 [0.016, 0.054] | 0.006 [0.000, 0.017] | 14.7 / 18.1 | 0 |
| C7r20 | extension | 1500 | 0.237 [0.228, 0.246] | 0.767 [0.745, 0.787] | 0.327 [0.299, 0.352] | 0.035 [0.023, 0.051] | 0.003 [0.000, 0.008] | 17.1 / 15.1 | 0 |
| C7r20 | hot | 1500 | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 0.193 [0.172, 0.213] | 0.017 [0.012, 0.024] | 0.000 [0.000, 0.000] | 16.6 / — | 0 |
| C7r40 | vanilla | 1500 | 0.325 [0.317, 0.334] | 0.912 [0.899, 0.925] | 0.322 [0.298, 0.346] | 0.023 [0.015, 0.031] | 0.007 [0.000, 0.016] | 15.2 / 15.9 | 0 |
| C7r40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r40 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.445 [0.419, 0.470] | 0.282 [0.249, 0.316] | 0.016 [0.011, 0.022] | 0.000 [0.000, 0.000] | 13.3 / 16.8 | 0 |
| C7r40 | extension | 1500 | 0.174 [0.166, 0.181] | 0.649 [0.624, 0.673] | 0.271 [0.244, 0.301] | 0.028 [0.018, 0.042] | 0.000 [0.000, 0.000] | 15.9 / 14.1 | 0 |
| C7r40 | hot | 1500 | 0.400 [0.400, 0.400] | 1.000 [1.000, 1.000] | 0.405 [0.381, 0.431] | 0.016 [0.013, 0.020] | 0.003 [0.000, 0.008] | 15.3 / — | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel[C7r20]**: insufficient data — coverage: seeded starts 14.1% of 1500 (seeds drawn 345; floor 5%); historical_meme_rate Δ -0.005 [-0.021, 0.013]; copy Δ 0.003 [-0.002, 0.007]; repetition Δ 0.000 [0.000, 0.000]; p95 25.9 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — historical_meme_rate did not rise significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel[C7r40]**: insufficient data — coverage: seeded starts 29.9% of 1500 (seeds drawn 345; floor 5%); historical_meme_rate Δ 0.003 [-0.016, 0.024]; copy Δ -0.001 [-0.005, 0.003]; repetition Δ 0.000 [0.000, 0.000]; p95 23.9 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — historical_meme_rate did not rise significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.863 [2.813, 2.914] (min 2.0); pool ECB 4.536 [4.507, 4.565] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C7r20]**: insufficient data — window escape 2.943 [2.895, 2.991] (min 2.0); pool ECB 4.545 [4.514, 4.575] (floor 4.0); pool ECB share 0.909 [0.903, 0.915] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:7%, 2:27%, 3:36%, 4:24%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C7r40]**: insufficient data — window escape 2.977 [2.925, 3.027] (min 2.0); pool ECB 4.515 [4.485, 4.545] (floor 4.0); pool ECB share 0.903 [0.897, 0.909] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:7%, 2:25%, 3:37%, 4:25%, 5:6%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.003] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1520 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 26.5 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7r20]**: pass — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7r40]**: pass — 5/27 memes reproduced (19%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
