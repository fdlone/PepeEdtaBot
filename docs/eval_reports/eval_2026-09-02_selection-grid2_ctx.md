# Eval report 2026-09-02 snapshot=selection-grid2 prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `b98b418`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
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
| candidate_accept_rate | 0.870 [0.862, 0.877] | 0.870 [0.862, 0.877] | 0.000 [0.000, 0.000] | 0.870 [0.862, 0.877] | 0.000 [0.000, 0.000] | 0.870 [0.862, 0.877] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.917 [10.691, 11.185] | 11.073 [10.828, 11.340] | 0.157 [0.007, 0.317] * | 11.096 [10.841, 11.369] | 0.179 [-0.019, 0.399] | 10.831 [10.604, 11.085] | -0.085 [-0.171, -0.005] * |
| unique_token_ratio | 0.984 [0.981, 0.986] | 0.983 [0.981, 0.985] | -0.001 [-0.002, 0.001] | 0.982 [0.980, 0.984] | -0.001 [-0.003, 0.000] | 0.984 [0.982, 0.986] | 0.001 [0.000, 0.002] * |
| exact_context_copy_rate | 0.226 [0.205, 0.247] | 0.205 [0.184, 0.226] | -0.021 [-0.030, -0.012] * | 0.185 [0.165, 0.205] | -0.041 [-0.055, -0.030] * | 0.207 [0.186, 0.227] | -0.019 [-0.027, -0.013] * |
| repetition_rate | 0.002 [0.000, 0.005] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.321 [0.305, 0.336] | 0.305 [0.289, 0.320] | -0.016 [-0.020, -0.012] * | 0.285 [0.268, 0.300] | -0.035 [-0.042, -0.029] * | 0.310 [0.294, 0.325] | -0.011 [-0.014, -0.008] * |
| context_affinity_without_copy | 0.253 [0.235, 0.273] | 0.240 [0.223, 0.256] | -0.011 [-0.015, -0.006] * | 0.223 [0.206, 0.240] | -0.025 [-0.032, -0.018] * | 0.244 [0.226, 0.262] | -0.006 [-0.009, -0.003] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.285 [0.243, 0.333] | 0.251 [0.208, 0.299] | -0.035 [-0.059, -0.013] * | 0.240 [0.195, 0.285] | -0.045 [-0.077, -0.013] * | 0.259 [0.216, 0.307] | -0.027 [-0.045, -0.011] * |
| structural_pool_ecb | 4.485 [4.452, 4.515] | 4.485 [4.452, 4.515] | 0.000 [0.000, 0.000] | 4.485 [4.452, 4.515] | 0.000 [0.000, 0.000] | 4.485 [4.452, 4.515] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.038 [1.985, 2.091] | 2.451 [2.394, 2.507] | 0.413 [0.380, 0.450] * | 2.410 [2.357, 2.463] | 0.372 [0.314, 0.432] * | 2.137 [2.081, 2.193] | 0.099 [0.081, 0.119] * |

C0: distinct-2 = 0.661 (basis 14875), distinct-3 = 0.818 (basis 13376) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 70.5/113.8 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C9d20: distinct-2 = 0.668 (basis 15110), distinct-3 = 0.824 (basis 13611) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 73.6/114.9 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C9d40: distinct-2 = 0.672 (basis 15144), distinct-3 = 0.830 (basis 13645) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 70.0/113.4 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C9w13: distinct-2 = 0.667 (basis 14747), distinct-3 = 0.823 (basis 13248) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 61.2/99.1 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.232 [0.189, 0.275] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.382] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.360] | 0.008 [0.000, 0.019] | 0.364 [0.337, 0.392] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.266 [0.225, 0.306] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.355 [0.307, 0.403] | 0.000 [0.000, 0.000] | 0.297 [0.269, 0.324] |
| C9d20 | generic | 375 | 1.000 [1.000, 1.000] | 0.211 [0.168, 0.253] | 0.000 [0.000, 0.000] | 0.334 [0.305, 0.366] |
| C9d20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.293 [0.248, 0.339] | 0.005 [0.000, 0.013] | 0.340 [0.313, 0.369] |
| C9d20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.267 [0.226, 0.308] |
| C9d20 | topical | 375 | 1.000 [1.000, 1.000] | 0.315 [0.272, 0.363] | 0.000 [0.000, 0.000] | 0.273 [0.244, 0.300] |
| C9d40 | generic | 375 | 1.000 [1.000, 1.000] | 0.187 [0.147, 0.227] | 0.000 [0.000, 0.000] | 0.313 [0.284, 0.345] |
| C9d40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.256 [0.211, 0.301] | 0.005 [0.000, 0.013] | 0.310 [0.282, 0.339] |
| C9d40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.259 [0.218, 0.299] |
| C9d40 | topical | 375 | 1.000 [1.000, 1.000] | 0.291 [0.248, 0.336] | 0.000 [0.000, 0.000] | 0.256 [0.228, 0.283] |
| C9w13 | generic | 375 | 1.000 [1.000, 1.000] | 0.208 [0.165, 0.251] | 0.000 [0.000, 0.000] | 0.339 [0.308, 0.370] |
| C9w13 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.288 [0.243, 0.333] | 0.005 [0.000, 0.013] | 0.348 [0.321, 0.377] |
| C9w13 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.264 [0.222, 0.304] |
| C9w13 | topical | 375 | 1.000 [1.000, 1.000] | 0.328 [0.280, 0.376] | 0.000 [0.000, 0.000] | 0.283 [0.255, 0.311] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.711 [0.687, 0.734] | 0.300 [0.277, 0.323] | 0.280 [0.256, 0.307] | 73.7 / 131.4 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.227 [0.195, 0.261] | 0.190 [0.140, 0.244] | 0.219 [0.152, 0.285] | 67.7 / 79.4 | 0 |
| C0 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.263 [0.239, 0.290] | 0.151 [0.120, 0.184] | 0.034 [0.017, 0.055] | 78.7 / 61.1 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9d20 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.697 [0.673, 0.720] | 0.289 [0.266, 0.314] | 0.257 [0.230, 0.282] | 76.0 / 132.9 | 0 |
| C9d20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9d20 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.242 [0.210, 0.276] | 0.173 [0.125, 0.225] | 0.205 [0.143, 0.267] | 69.4 / 82.3 | 0 |
| C9d20 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.272 [0.250, 0.299] | 0.137 [0.109, 0.168] | 0.030 [0.013, 0.050] | 81.3 / 62.9 | 0 |
| C9d20 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9d40 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.705 [0.681, 0.729] | 0.265 [0.244, 0.287] | 0.233 [0.209, 0.257] | 72.6 / 129.4 | 0 |
| C9d40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9d40 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.216 [0.186, 0.248] | 0.184 [0.130, 0.242] | 0.194 [0.132, 0.264] | 66.3 / 78.6 | 0 |
| C9d40 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.276 [0.252, 0.304] | 0.124 [0.098, 0.155] | 0.016 [0.003, 0.032] | 77.8 / 59.6 | 0 |
| C9d40 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9w13 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.699 [0.674, 0.721] | 0.293 [0.270, 0.316] | 0.260 [0.233, 0.287] | 63.5 / 116.1 | 0 |
| C9w13 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9w13 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.233 [0.201, 0.269] | 0.184 [0.135, 0.236] | 0.213 [0.155, 0.277] | 58.2 / 68.6 | 0 |
| C9w13 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.276 [0.251, 0.303] | 0.141 [0.111, 0.171] | 0.023 [0.010, 0.042] | 68.2 / 52.0 | 0 |
| C9w13 | hot | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window[C9d20]**: insufficient data — single_trajectory_share Δ -0.128 [-0.147, -0.111] *; window_escape Δ 0.413 [0.380, 0.450] *; affinity_without_copy Δ -0.011 [-0.015, -0.006] *; copy Δ -0.021 [-0.030, -0.012] *; repetition Δ -0.001 [-0.002, 0.000]; pool ECB 4.485 [4.452, 4.515] (floor 4.0); p95 114.9 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — affinity without copies dropped significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window[C9d40]**: insufficient data — single_trajectory_share Δ -0.161 [-0.186, -0.135] *; window_escape Δ 0.372 [0.314, 0.432] *; affinity_without_copy Δ -0.025 [-0.032, -0.018] *; copy Δ -0.041 [-0.055, -0.030] *; repetition Δ -0.001 [-0.002, 0.000]; pool ECB 4.485 [4.452, 4.515] (floor 4.0); p95 113.4 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — affinity without copies dropped significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window[C9w13]**: insufficient data — single_trajectory_share Δ -0.043 [-0.054, -0.033] * — coverage below the floor (-0.05): the share of single-trajectory inputs did not drop enough; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.038 [1.985, 2.091] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C9d20]**: insufficient data — window escape 2.451 [2.394, 2.507] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:27%, 2:24%, 3:28%, 4:16%, 5:4%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C9d40]**: insufficient data — window escape 2.410 [2.357, 2.463] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:24%, 2:31%, 3:28%, 4:14%, 5:3%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C9w13]**: insufficient data — window escape 2.137 [2.081, 2.193] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:36%, 2:28%, 3:24%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 885 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 113.8 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9d20]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9d40]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9w13]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
