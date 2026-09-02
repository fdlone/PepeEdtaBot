# Eval report 2026-09-02 snapshot=l1-route-v2 prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `1d36e26`. Generations per configuration: 500.
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
| mean_response_length | 10.460 [10.248, 10.693] | 10.365 [10.139, 10.619] | -0.095 [-0.368, 0.199] | 10.623 [10.396, 10.844] | 0.163 [-0.121, 0.455] |
| unique_token_ratio | 0.987 [0.986, 0.989] | 0.988 [0.986, 0.989] | 0.000 [-0.002, 0.002] | 0.988 [0.986, 0.989] | 0.000 [-0.002, 0.003] |
| exact_context_copy_rate | 0.003 [0.001, 0.005] | 0.009 [0.005, 0.014] | 0.006 [0.001, 0.011] * | 0.004 [0.001, 0.007] | 0.001 [-0.003, 0.005] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.025 [0.021, 0.029] | 0.033 [0.027, 0.039] | 0.008 [0.002, 0.015] * | 0.025 [0.021, 0.030] | 0.001 [-0.005, 0.006] |
| context_affinity_without_copy | 0.025 [0.021, 0.029] | 0.031 [0.026, 0.037] | 0.006 [-0.000, 0.013] | 0.024 [0.020, 0.028] | -0.001 [-0.007, 0.004] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.053 [0.035, 0.077] | 0.192 [0.155, 0.229] | 0.139 [0.093, 0.184] * | 0.352 [0.307, 0.405] | 0.299 [0.243, 0.352] * |
| structural_pool_ecb | 4.527 [4.497, 4.555] | 4.546 [4.517, 4.573] | 0.019 [-0.023, 0.059] | 4.526 [4.496, 4.556] | -0.001 [-0.044, 0.037] |
| structural_window_escape | 2.868 [2.815, 2.917] | 2.957 [2.907, 3.006] | 0.089 [0.021, 0.161] * | 2.959 [2.907, 3.009] | 0.091 [0.024, 0.164] * |

C0: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.6/25.9 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C7r20: distinct-2 = 0.683 (basis 14048), distinct-3 = 0.830 (basis 12548) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 13.5/21.8 ms; cache_hit_rate: 29%; mean normalized entropy: 0.245 (branching 3.37); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 1845 draws, empty 0%; storage_delta: n/a.

C7r40: distinct-2 = 0.641 (basis 14434), distinct-3 = 0.783 (basis 12934) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 13.3/21.9 ms; cache_hit_rate: 34%; mean normalized entropy: 0.252 (branching 3.40); mean applied temperature: 2.67; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 1845 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.024, 0.037] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.042] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.020 [0.016, 0.024] |
| C7r20 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.028 [0.021, 0.037] |
| C7r20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.016 [0.005, 0.029] | 0.000 [0.000, 0.000] | 0.034 [0.027, 0.043] |
| C7r20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.046 [0.027, 0.069] |
| C7r20 | topical | 375 | 1.000 [1.000, 1.000] | 0.011 [0.000, 0.021] | 0.000 [0.000, 0.000] | 0.025 [0.019, 0.032] |
| C7r40 | generic | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.021 [0.017, 0.025] |
| C7r40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.032 [0.026, 0.039] |
| C7r40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.021 [0.009, 0.036] |
| C7r40 | topical | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.026 [0.019, 0.036] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.551 [0.526, 0.577] | 0.021 [0.016, 0.027] | 0.005 [0.001, 0.010] | 15.5 / 16.5 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.279 [0.246, 0.314] | 0.022 [0.016, 0.028] | 0.000 [0.000, 0.000] | 13.8 / 17.0 | 0 |
| C0 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.389 [0.364, 0.416] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 15.9 / 13.8 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | vanilla | 1500 | 0.469 [0.460, 0.479] | 0.971 [0.963, 0.979] | 0.447 [0.420, 0.474] | 0.031 [0.022, 0.040] | 0.014 [0.005, 0.023] | 14.0 / 14.3 | 0 |
| C7r20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | mutated | 1500 | 0.097 [0.091, 0.102] | 0.437 [0.413, 0.461] | 0.276 [0.244, 0.308] | 0.029 [0.016, 0.048] | 0.000 [0.000, 0.000] | 12.4 / 15.3 | 0 |
| C7r20 | extension | 1500 | 0.234 [0.225, 0.243] | 0.761 [0.739, 0.781] | 0.326 [0.299, 0.355] | 0.039 [0.026, 0.054] | 0.005 [0.000, 0.013] | 14.5 / 12.6 | 0 |
| C7r20 | hot | 1500 | 0.200 [0.200, 0.200] | 1.000 [1.000, 1.000] | 0.197 [0.176, 0.217] | 0.023 [0.017, 0.030] | 0.007 [0.000, 0.017] | 14.0 / — | 0 |
| C7r40 | vanilla | 1500 | 0.329 [0.321, 0.337] | 0.916 [0.903, 0.929] | 0.322 [0.298, 0.348] | 0.022 [0.015, 0.030] | 0.009 [0.002, 0.018] | 14.0 / 14.2 | 0 |
| C7r40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r40 | mutated | 1500 | 0.099 [0.093, 0.105] | 0.440 [0.414, 0.464] | 0.280 [0.245, 0.317] | 0.013 [0.009, 0.018] | 0.000 [0.000, 0.000] | 12.3 / 15.3 | 0 |
| C7r40 | extension | 1500 | 0.173 [0.165, 0.180] | 0.648 [0.625, 0.671] | 0.274 [0.247, 0.298] | 0.032 [0.021, 0.045] | 0.000 [0.000, 0.000] | 14.5 / 13.1 | 0 |
| C7r40 | hot | 1500 | 0.400 [0.400, 0.400] | 1.000 [1.000, 1.000] | 0.404 [0.381, 0.429] | 0.025 [0.020, 0.030] | 0.003 [0.000, 0.008] | 14.0 / — | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel[C7r20]**: insufficient data — coverage: seeded starts 14.6% of 1500 (seeds drawn 345; floor 5%); historical_meme_rate Δ 0.139 [0.093, 0.184] *; copy Δ 0.006 [0.001, 0.011] *; repetition Δ 0.000 [0.000, 0.000]; p95 21.8 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — copy rose significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel[C7r40]**: insufficient data — coverage: seeded starts 30.5% of 1500 (seeds drawn 345; floor 5%); historical_meme_rate Δ 0.299 [0.243, 0.352] *; copy Δ 0.001 [-0.003, 0.005]; repetition Δ 0.000 [0.000, 0.000]; p95 21.9 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C7r20]**: insufficient data — window escape 2.957 [2.907, 3.006] (min 2.0); pool ECB 4.546 [4.517, 4.573] (floor 4.0); pool ECB share 0.909 [0.903, 0.915] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:7%, 2:27%, 3:37%, 4:24%, 5:6%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C7r40]**: insufficient data — window escape 2.959 [2.907, 3.009] (min 2.0); pool ECB 4.526 [4.496, 4.556] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:25%, 3:37%, 4:25%, 5:6%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1552 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 25.9 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C7r20]**: pass — 11/18 memes reproduced (61%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C7r40]**: pass — 11/18 memes reproduced (61%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
