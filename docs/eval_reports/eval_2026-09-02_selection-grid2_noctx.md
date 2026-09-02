# Eval report 2026-09-02 snapshot=selection-grid2 prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `520c7a3`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C9d20**: 1500 generations
- **C9d40**: 1500 generations
- **C9w13**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C9d20 | Δ C9d20 vs C0 | C9d40 | Δ C9d40 vs C0 | C9w13 | Δ C9w13 vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.460 [10.248, 10.693] | 10.659 [10.433, 10.911] | 0.199 [0.041, 0.359] * | 10.839 [10.585, 11.093] | 0.379 [0.192, 0.606] * | 10.460 [10.248, 10.693] | 0.000 [0.000, 0.000] |
| unique_token_ratio | 0.987 [0.986, 0.989] | 0.985 [0.984, 0.987] | -0.002 [-0.003, -0.001] * | 0.983 [0.982, 0.985] | -0.004 [-0.005, -0.003] * | 0.987 [0.986, 0.989] | 0.000 [0.000, 0.000] |
| exact_context_copy_rate | 0.003 [0.001, 0.005] | 0.004 [0.001, 0.007] | 0.001 [-0.002, 0.005] | 0.006 [0.003, 0.010] | 0.003 [-0.001, 0.008] | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.025 [0.021, 0.029] | 0.024 [0.020, 0.028] | -0.001 [-0.004, 0.002] | 0.029 [0.024, 0.033] | 0.004 [-0.000, 0.008] | 0.025 [0.021, 0.029] | 0.000 [0.000, 0.000] |
| context_affinity_without_copy | 0.025 [0.021, 0.029] | 0.024 [0.020, 0.028] | -0.001 [-0.004, 0.002] | 0.029 [0.024, 0.034] | 0.004 [-0.001, 0.009] | 0.025 [0.021, 0.029] | 0.000 [0.000, 0.000] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.053 [0.035, 0.077] | 0.053 [0.032, 0.075] | 0.000 [-0.019, 0.019] | 0.061 [0.037, 0.085] | 0.008 [-0.016, 0.032] | 0.053 [0.035, 0.077] | 0.000 [0.000, 0.000] |
| structural_pool_ecb | 4.527 [4.497, 4.555] | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.868 [2.815, 2.917] | 3.281 [3.233, 3.326] | 0.413 [0.377, 0.453] * | 2.833 [2.781, 2.885] | -0.035 [-0.093, 0.026] | 2.868 [2.815, 2.917] | 0.000 [0.000, 0.000] |

C0: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 43.8/71.2 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C9d20: distinct-2 = 0.706 (basis 14489), distinct-3 = 0.860 (basis 12989) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 47.6/76.0 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C9d40: distinct-2 = 0.711 (basis 14758), distinct-3 = 0.862 (basis 13258) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 46.4/74.4 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C9w13: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 45.2/72.0 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.024, 0.037] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.042] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.020 [0.016, 0.024] |
| C9d20 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.022 [0.018, 0.027] |
| C9d20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.025, 0.036] |
| C9d20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.024 [0.013, 0.039] |
| C9d20 | topical | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.019 [0.016, 0.023] |
| C9d40 | generic | 375 | 1.000 [1.000, 1.000] | 0.011 [0.003, 0.021] | 0.000 [0.000, 0.000] | 0.026 [0.021, 0.031] |
| C9d40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.011] | 0.000 [0.000, 0.000] | 0.029 [0.024, 0.035] |
| C9d40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.011 [0.003, 0.021] | 0.000 [0.000, 0.000] | 0.040 [0.024, 0.062] |
| C9d40 | topical | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.019 [0.016, 0.024] |
| C9w13 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C9w13 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.024, 0.037] |
| C9w13 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.042] |
| C9w13 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.020 [0.016, 0.024] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.551 [0.526, 0.577] | 0.021 [0.016, 0.027] | 0.005 [0.001, 0.010] | 45.6 / 47.9 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.279 [0.246, 0.314] | 0.022 [0.016, 0.028] | 0.000 [0.000, 0.000] | 40.7 / 49.7 | 0 |
| C0 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.389 [0.364, 0.416] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 46.3 / 41.8 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9d20 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.541 [0.516, 0.564] | 0.021 [0.016, 0.026] | 0.007 [0.002, 0.014] | 49.2 / 51.7 | 0 |
| C9d20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9d20 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.270 [0.239, 0.301] | 0.024 [0.015, 0.034] | 0.000 [0.000, 0.000] | 43.2 / 54.2 | 0 |
| C9d20 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.405 [0.380, 0.433] | 0.030 [0.023, 0.037] | 0.000 [0.000, 0.000] | 50.1 / 44.5 | 0 |
| C9d20 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9d40 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.558 [0.532, 0.583] | 0.024 [0.019, 0.030] | 0.011 [0.004, 0.019] | 47.7 / 51.0 | 0 |
| C9d40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9d40 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.243 [0.210, 0.276] | 0.029 [0.016, 0.046] | 0.000 [0.000, 0.000] | 42.1 / 52.4 | 0 |
| C9d40 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.399 [0.374, 0.426] | 0.036 [0.027, 0.047] | 0.000 [0.000, 0.000] | 48.6 / 43.1 | 0 |
| C9d40 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9w13 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.551 [0.526, 0.577] | 0.021 [0.016, 0.027] | 0.005 [0.001, 0.010] | 46.9 / 50.5 | 0 |
| C9w13 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9w13 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.279 [0.246, 0.314] | 0.022 [0.016, 0.028] | 0.000 [0.000, 0.000] | 41.4 / 51.6 | 0 |
| C9w13 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.389 [0.364, 0.416] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 47.8 / 42.5 | 0 |
| C9w13 | hot | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window[C9d20]**: insufficient data — single_trajectory_share Δ -0.061 [-0.073, -0.048] *; window_escape Δ 0.413 [0.377, 0.453] *; affinity_without_copy Δ -0.001 [-0.004, 0.002]; copy Δ 0.001 [-0.002, 0.005]; repetition Δ 0.000 [0.000, 0.000]; pool ECB 4.527 [4.497, 4.555] (floor 4.0); p95 76.0 ms (budget 150); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window[C9d40]**: insufficient data — single_trajectory_share Δ 0.020 [-0.001, 0.042] — coverage below the floor (-0.05): the share of single-trajectory inputs did not drop enough; gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window[C9w13]**: insufficient data — single_trajectory_share Δ 0.000 [0.000, 0.000] — coverage below the floor (-0.05): the share of single-trajectory inputs did not drop enough; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C9d20]**: insufficient data — window escape 3.281 [3.233, 3.326] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:2%, 2:18%, 3:40%, 4:32%, 5:9%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C9d40]**: insufficient data — window escape 2.833 [2.781, 2.885] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:10%, 2:29%, 3:35%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C9w13]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1552 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 71.2 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9d20]**: pass — 11/18 memes reproduced (61%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9d40]**: pass — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9w13]**: pass — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
