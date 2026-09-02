# Eval report 2026-09-01 snapshot=l1-grid prompts=308b7deaea0f seeds=42,1337,2026 mode=noctx

Revision: `cd6e1c2`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C7a**: 1500 generations
- **C7b**: 1500 generations
- **C7c**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C7a | Δ C7a vs C0 | C7b | Δ C7b vs C0 | C7c | Δ C7c vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.352 [10.138, 10.603] | 10.297 [10.091, 10.542] | -0.055 [-0.127, 0.019] | 10.337 [10.129, 10.582] | -0.015 [-0.129, 0.093] | 10.291 [10.079, 10.533] | -0.061 [-0.182, 0.056] |
| unique_token_ratio | 0.988 [0.986, 0.989] | 0.988 [0.986, 0.989] | 0.000 [-0.000, 0.001] | 0.988 [0.987, 0.990] | 0.001 [-0.000, 0.002] | 0.987 [0.986, 0.989] | -0.000 [-0.001, 0.001] |
| exact_context_copy_rate | 0.004 [0.001, 0.007] | 0.005 [0.001, 0.008] | 0.001 [0.000, 0.002] | 0.005 [0.001, 0.009] | 0.001 [0.000, 0.003] | 0.005 [0.002, 0.009] | 0.001 [0.000, 0.003] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] |
| cycle_detection_rate | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.023 [0.019, 0.027] | 0.023 [0.019, 0.027] | 0.000 [-0.001, 0.002] | 0.023 [0.019, 0.028] | 0.001 [-0.001, 0.003] | 0.024 [0.020, 0.029] | 0.002 [-0.001, 0.005] |
| context_affinity_without_copy | 0.023 [0.019, 0.027] | 0.023 [0.019, 0.028] | 0.000 [-0.001, 0.002] | 0.023 [0.019, 0.028] | 0.001 [-0.001, 0.003] | 0.024 [0.020, 0.029] | 0.002 [-0.001, 0.005] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.019 [0.008, 0.035] | 0.016 [0.005, 0.029] | -0.003 [-0.008, 0.000] | 0.019 [0.005, 0.035] | 0.000 [-0.011, 0.011] | 0.019 [0.005, 0.035] | 0.000 [-0.011, 0.011] |
| structural_pool_ecb | 4.536 [4.507, 4.565] | 4.537 [4.509, 4.567] | 0.001 [-0.009, 0.012] | 4.536 [4.505, 4.564] | 0.000 [-0.015, 0.015] | 4.535 [4.504, 4.565] | -0.001 [-0.017, 0.016] |
| structural_window_escape | 2.863 [2.813, 2.914] | 2.861 [2.811, 2.911] | -0.002 [-0.021, 0.015] | 2.845 [2.795, 2.896] | -0.017 [-0.044, 0.009] | 2.864 [2.815, 2.916] | 0.001 [-0.025, 0.031] |

C0: distinct-2 = 0.703 (basis 14028), distinct-3 = 0.857 (basis 12528) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.0/22.9 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.52); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C7a: distinct-2 = 0.702 (basis 13945), distinct-3 = 0.857 (basis 12445) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 15.2/24.4 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.51); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C7b: distinct-2 = 0.700 (basis 14005), distinct-3 = 0.854 (basis 12505) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 15.3/24.5 ms; cache_hit_rate: 28%; mean normalized entropy: 0.247 (branching 3.47); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C7c: distinct-2 = 0.701 (basis 13937), distinct-3 = 0.856 (basis 12437) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 15.1/24.1 ms; cache_hit_rate: 27%; mean normalized entropy: 0.244 (branching 3.45); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.031] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.017 [0.013, 0.022] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.040] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.031] |
| C7a | generic | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.030] |
| C7a | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.017 [0.012, 0.022] |
| C7a | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.028 [0.014, 0.044] |
| C7a | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.030] |
| C7b | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.030] |
| C7b | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.014 [0.010, 0.018] |
| C7b | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.032 [0.017, 0.049] |
| C7b | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.032] |
| C7c | generic | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.008] | 0.026 [0.020, 0.033] |
| C7c | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.014 [0.011, 0.019] |
| C7c | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.034 [0.018, 0.052] |
| C7c | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.031] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.590 [0.579, 0.602] | 0.989 [0.984, 0.994] | 0.559 [0.532, 0.582] | 0.022 [0.017, 0.028] | 0.007 [0.002, 0.014] | 14.5 / 15.0 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.100 [0.094, 0.106] | 0.447 [0.423, 0.472] | 0.273 [0.242, 0.307] | 0.028 [0.019, 0.038] | 0.000 [0.000, 0.000] | 12.9 / 15.8 | 0 |
| C0 | extension | 1500 | 0.310 [0.299, 0.321] | 0.843 [0.823, 0.863] | 0.386 [0.358, 0.413] | 0.022 [0.015, 0.030] | 0.000 [0.000, 0.000] | 14.8 / 13.2 | 0 |
| C7a | vanilla | 1500 | 0.587 [0.577, 0.599] | 0.989 [0.983, 0.994] | 0.554 [0.528, 0.578] | 0.022 [0.016, 0.028] | 0.009 [0.002, 0.016] | 15.6 / 16.7 | 0 |
| C7a | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7a | mutated | 1500 | 0.100 [0.094, 0.106] | 0.445 [0.421, 0.471] | 0.266 [0.235, 0.299] | 0.026 [0.018, 0.037] | 0.000 [0.000, 0.000] | 13.8 / 17.1 | 0 |
| C7a | extension | 1500 | 0.313 [0.302, 0.323] | 0.851 [0.833, 0.869] | 0.392 [0.368, 0.418] | 0.023 [0.016, 0.032] | 0.000 [0.000, 0.000] | 15.9 / 14.1 | 0 |
| C7b | vanilla | 1500 | 0.584 [0.573, 0.595] | 0.987 [0.981, 0.992] | 0.558 [0.532, 0.582] | 0.023 [0.017, 0.029] | 0.008 [0.002, 0.015] | 15.9 / 15.8 | 0 |
| C7b | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7b | mutated | 1500 | 0.099 [0.093, 0.105] | 0.442 [0.417, 0.467] | 0.262 [0.228, 0.294] | 0.027 [0.018, 0.038] | 0.000 [0.000, 0.000] | 14.0 / 17.4 | 0 |
| C7b | extension | 1500 | 0.317 [0.306, 0.327] | 0.857 [0.839, 0.875] | 0.389 [0.362, 0.416] | 0.024 [0.016, 0.032] | 0.000 [0.000, 0.000] | 16.2 / 14.4 | 0 |
| C7c | vanilla | 1500 | 0.588 [0.577, 0.599] | 0.992 [0.988, 0.996] | 0.548 [0.522, 0.574] | 0.024 [0.018, 0.032] | 0.010 [0.004, 0.017] | 15.6 / 16.3 | 0 |
| C7c | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7c | mutated | 1500 | 0.099 [0.093, 0.105] | 0.439 [0.415, 0.465] | 0.255 [0.222, 0.289] | 0.028 [0.018, 0.039] | 0.000 [0.000, 0.000] | 13.7 / 17.1 | 0 |
| C7c | extension | 1500 | 0.313 [0.303, 0.323] | 0.857 [0.839, 0.874] | 0.401 [0.371, 0.428] | 0.024 [0.016, 0.032] | 0.000 [0.000, 0.000] | 15.9 / 14.0 | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel[C7a]**: insufficient data — coverage: seeded starts 0.7% of 1500 (seeds drawn 345; floor 5%) — coverage below the floor: the channel reached too few replies to resolve its effect; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel[C7b]**: insufficient data — coverage: seeded starts 1.7% of 1500 (seeds drawn 345; floor 5%) — coverage below the floor: the channel reached too few replies to resolve its effect; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel[C7c]**: insufficient data — coverage: seeded starts 2.1% of 1500 (seeds drawn 345; floor 5%) — coverage below the floor: the channel reached too few replies to resolve its effect; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.863 [2.813, 2.914] (min 2.0); pool ECB 4.536 [4.507, 4.565] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C7a]**: insufficient data — window escape 2.861 [2.811, 2.911] (min 2.0); pool ECB 4.537 [4.509, 4.567] (floor 4.0); pool ECB share 0.907 [0.902, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:27%, 3:39%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C7b]**: insufficient data — window escape 2.845 [2.795, 2.896] (min 2.0); pool ECB 4.536 [4.505, 4.564] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:4%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C7c]**: insufficient data — window escape 2.864 [2.815, 2.916] (min 2.0); pool ECB 4.535 [4.504, 4.565] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.003] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1520 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 22.9 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7a]**: pass — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7b]**: pass — 7/27 memes reproduced (26%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7c]**: pass — 9/27 memes reproduced (33%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
