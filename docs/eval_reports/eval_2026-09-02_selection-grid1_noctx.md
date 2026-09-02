# Eval report 2026-09-02 snapshot=selection-grid1 prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `520c7a3`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C9m50**: 1500 generations
- **C9m80**: 1500 generations
- **C9w10**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C9m50 | Δ C9m50 vs C0 | C9m80 | Δ C9m80 vs C0 | C9w10 | Δ C9w10 vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.460 [10.248, 10.693] | 10.772 [10.541, 11.025] | 0.312 [0.143, 0.499] * | 10.836 [10.586, 11.114] | 0.376 [0.154, 0.605] * | 10.460 [10.248, 10.693] | 0.000 [0.000, 0.000] |
| unique_token_ratio | 0.987 [0.986, 0.989] | 0.985 [0.984, 0.987] | -0.002 [-0.003, -0.001] * | 0.984 [0.983, 0.986] | -0.003 [-0.004, -0.002] * | 0.987 [0.986, 0.989] | 0.000 [0.000, 0.000] |
| exact_context_copy_rate | 0.003 [0.001, 0.005] | 0.008 [0.003, 0.013] | 0.005 [0.001, 0.010] * | 0.011 [0.007, 0.017] | 0.009 [0.003, 0.014] * | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.001, 0.005] | 0.003 [0.001, 0.005] * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.025 [0.021, 0.029] | 0.025 [0.021, 0.030] | 0.001 [-0.002, 0.003] | 0.027 [0.023, 0.031] | 0.002 [-0.000, 0.005] | 0.025 [0.021, 0.029] | 0.000 [0.000, 0.000] |
| context_affinity_without_copy | 0.025 [0.021, 0.029] | 0.025 [0.021, 0.030] | 0.001 [-0.001, 0.003] | 0.027 [0.023, 0.032] | 0.003 [-0.000, 0.005] | 0.025 [0.021, 0.029] | 0.000 [0.000, 0.000] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.053 [0.035, 0.077] | 0.051 [0.029, 0.075] | -0.003 [-0.016, 0.011] | 0.053 [0.032, 0.077] | 0.000 [-0.016, 0.016] | 0.053 [0.035, 0.077] | 0.000 [0.000, 0.000] |
| structural_pool_ecb | 4.527 [4.497, 4.555] | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.868 [2.815, 2.917] | 3.503 [3.453, 3.547] | 0.635 [0.595, 0.675] * | 4.095 [4.053, 4.139] | 1.227 [1.181, 1.279] * | 2.868 [2.815, 2.917] | 0.000 [0.000, 0.000] |

C0: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 43.6/71.8 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C9m50: distinct-2 = 0.703 (basis 14658), distinct-3 = 0.857 (basis 13158) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 47.6/76.6 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C9m80: distinct-2 = 0.709 (basis 14754), distinct-3 = 0.859 (basis 13254) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 46.5/73.1 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C9w10: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 45.5/72.8 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.024, 0.037] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.042] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.020 [0.016, 0.024] |
| C9m50 | generic | 375 | 1.000 [1.000, 1.000] | 0.011 [0.003, 0.021] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.029] |
| C9m50 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.033 [0.026, 0.040] |
| C9m50 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.013 [0.003, 0.027] | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.040] |
| C9m50 | topical | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.019 [0.015, 0.023] |
| C9m80 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.003 [0.000, 0.008] | 0.026 [0.021, 0.032] |
| C9m80 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.016 [0.005, 0.029] | 0.003 [0.000, 0.008] | 0.031 [0.025, 0.038] |
| C9m80 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.016 [0.005, 0.032] | 0.003 [0.000, 0.011] | 0.030 [0.017, 0.048] |
| C9m80 | topical | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.003 [0.000, 0.008] | 0.021 [0.017, 0.026] |
| C9w10 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C9w10 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.024, 0.037] |
| C9w10 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.042] |
| C9w10 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.020 [0.016, 0.024] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.551 [0.526, 0.577] | 0.021 [0.016, 0.027] | 0.005 [0.001, 0.010] | 45.2 / 47.9 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.279 [0.246, 0.314] | 0.022 [0.016, 0.028] | 0.000 [0.000, 0.000] | 40.2 / 49.5 | 0 |
| C0 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.389 [0.364, 0.416] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 46.0 / 41.2 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9m50 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.539 [0.514, 0.563] | 0.022 [0.016, 0.027] | 0.015 [0.008, 0.024] | 49.1 / 50.8 | 0 |
| C9m50 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9m50 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.267 [0.235, 0.301] | 0.020 [0.014, 0.028] | 0.000 [0.000, 0.000] | 43.1 / 54.0 | 0 |
| C9m50 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.409 [0.383, 0.435] | 0.033 [0.025, 0.041] | 0.000 [0.000, 0.000] | 50.0 / 44.2 | 0 |
| C9m50 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9m80 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.545 [0.519, 0.568] | 0.023 [0.018, 0.028] | 0.021 [0.012, 0.031] | 47.9 / 50.5 | 0 |
| C9m80 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9m80 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.252 [0.218, 0.286] | 0.019 [0.012, 0.027] | 0.000 [0.000, 0.000] | 42.2 / 52.7 | 0 |
| C9m80 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.410 [0.385, 0.435] | 0.037 [0.028, 0.046] | 0.000 [0.000, 0.000] | 48.8 / 43.2 | 0 |
| C9m80 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9w10 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.551 [0.526, 0.577] | 0.021 [0.016, 0.027] | 0.005 [0.001, 0.010] | 47.1 / 49.9 | 0 |
| C9w10 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9w10 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.279 [0.246, 0.314] | 0.022 [0.016, 0.028] | 0.000 [0.000, 0.000] | 41.4 / 51.9 | 0 |
| C9w10 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.389 [0.364, 0.416] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 47.9 / 42.6 | 0 |
| C9w10 | hot | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window[C9m50]**: insufficient data — single_trajectory_share Δ -0.059 [-0.071, -0.046] *; window_escape Δ 0.635 [0.595, 0.675] *; affinity_without_copy Δ 0.001 [-0.001, 0.003]; copy Δ 0.005 [0.001, 0.010] *; repetition Δ 0.000 [0.000, 0.000]; pool ECB 4.527 [4.497, 4.555] (floor 4.0); p95 76.6 ms (budget 150) — copy rose significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window[C9m80]**: insufficient data — single_trajectory_share Δ -0.076 [-0.089, -0.062] *; window_escape Δ 1.227 [1.181, 1.279] *; affinity_without_copy Δ 0.003 [-0.000, 0.005]; copy Δ 0.009 [0.003, 0.014] *; repetition Δ 0.003 [0.001, 0.005] *; pool ECB 4.527 [4.497, 4.555] (floor 4.0); p95 73.1 ms (budget 150) — copy rose significantly; repetition rose significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window[C9w10]**: insufficient data — single_trajectory_share Δ 0.000 [0.000, 0.000] — coverage below the floor (-0.05): the share of single-trajectory inputs did not drop enough; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C9m50]**: insufficient data — window escape 3.503 [3.453, 3.547] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:2%, 2:11%, 3:36%, 4:36%, 5:15%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C9m80]**: insufficient data — window escape 4.095 [4.053, 4.139] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:0%, 2:3%, 3:19%, 4:43%, 5:35%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C9w10]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1552 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 71.8 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9m50]**: pass — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9m80]**: pass — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9w10]**: pass — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
