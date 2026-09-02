# Eval report 2026-09-02 snapshot=selection-grid1 prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `b98b418`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
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
| candidate_accept_rate | 0.870 [0.862, 0.877] | 0.870 [0.862, 0.877] | 0.000 [0.000, 0.000] | 0.870 [0.862, 0.877] | 0.000 [0.000, 0.000] | 0.870 [0.862, 0.877] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.917 [10.691, 11.185] | 11.103 [10.850, 11.369] | 0.186 [0.033, 0.352] * | 11.079 [10.813, 11.369] | 0.163 [-0.069, 0.387] | 10.712 [10.489, 10.977] | -0.205 [-0.327, -0.101] * |
| unique_token_ratio | 0.984 [0.981, 0.986] | 0.983 [0.981, 0.985] | -0.000 [-0.002, 0.001] | 0.984 [0.982, 0.986] | 0.000 [-0.001, 0.002] | 0.985 [0.983, 0.987] | 0.001 [0.000, 0.002] * |
| exact_context_copy_rate | 0.226 [0.205, 0.247] | 0.210 [0.189, 0.231] | -0.016 [-0.024, -0.008] * | 0.185 [0.165, 0.205] | -0.041 [-0.053, -0.029] * | 0.193 [0.173, 0.213] | -0.033 [-0.042, -0.025] * |
| repetition_rate | 0.002 [0.000, 0.005] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.321 [0.305, 0.336] | 0.307 [0.291, 0.322] | -0.014 [-0.018, -0.009] * | 0.282 [0.265, 0.298] | -0.039 [-0.045, -0.032] * | 0.293 [0.278, 0.309] | -0.027 [-0.033, -0.022] * |
| context_affinity_without_copy | 0.253 [0.235, 0.273] | 0.242 [0.223, 0.260] | -0.010 [-0.014, -0.006] * | 0.222 [0.206, 0.238] | -0.026 [-0.033, -0.019] * | 0.229 [0.213, 0.246] | -0.019 [-0.025, -0.013] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.285 [0.243, 0.333] | 0.259 [0.216, 0.307] | -0.027 [-0.051, -0.005] * | 0.245 [0.203, 0.293] | -0.040 [-0.069, -0.011] * | 0.235 [0.195, 0.280] | -0.051 [-0.075, -0.029] * |
| structural_pool_ecb | 4.485 [4.452, 4.515] | 4.485 [4.452, 4.515] | 0.000 [0.000, 0.000] | 4.485 [4.452, 4.515] | 0.000 [0.000, 0.000] | 4.485 [4.452, 4.515] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.038 [1.985, 2.091] | 2.585 [2.525, 2.646] | 0.547 [0.512, 0.588] * | 3.246 [3.183, 3.312] | 1.208 [1.153, 1.264] * | 2.263 [2.206, 2.317] | 0.225 [0.197, 0.254] * |

C0: distinct-2 = 0.661 (basis 14875), distinct-3 = 0.818 (basis 13376) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 69.8/114.4 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C9m50: distinct-2 = 0.669 (basis 15154), distinct-3 = 0.825 (basis 13656) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 74.4/116.7 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C9m80: distinct-2 = 0.675 (basis 15119), distinct-3 = 0.831 (basis 13621) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 71.0/111.1 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C9w10: distinct-2 = 0.676 (basis 14568), distinct-3 = 0.833 (basis 13069) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 61.2/99.2 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.232 [0.189, 0.275] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.382] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.360] | 0.008 [0.000, 0.019] | 0.364 [0.337, 0.392] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.266 [0.225, 0.306] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.355 [0.307, 0.403] | 0.000 [0.000, 0.000] | 0.297 [0.269, 0.324] |
| C9m50 | generic | 375 | 1.000 [1.000, 1.000] | 0.211 [0.168, 0.253] | 0.000 [0.000, 0.000] | 0.333 [0.304, 0.366] |
| C9m50 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.291 [0.243, 0.336] | 0.005 [0.000, 0.013] | 0.342 [0.314, 0.371] |
| C9m50 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.269 [0.229, 0.307] |
| C9m50 | topical | 375 | 1.000 [1.000, 1.000] | 0.336 [0.291, 0.384] | 0.000 [0.000, 0.000] | 0.279 [0.251, 0.306] |
| C9m80 | generic | 375 | 1.000 [1.000, 1.000] | 0.187 [0.147, 0.229] | 0.000 [0.000, 0.000] | 0.302 [0.274, 0.333] |
| C9m80 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.248 [0.205, 0.291] | 0.005 [0.000, 0.013] | 0.308 [0.281, 0.338] |
| C9m80 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.265 [0.224, 0.303] |
| C9m80 | topical | 375 | 1.000 [1.000, 1.000] | 0.301 [0.256, 0.349] | 0.000 [0.000, 0.000] | 0.251 [0.223, 0.279] |
| C9w10 | generic | 375 | 1.000 [1.000, 1.000] | 0.200 [0.160, 0.243] | 0.000 [0.000, 0.000] | 0.324 [0.295, 0.357] |
| C9w10 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.267 [0.221, 0.312] | 0.005 [0.000, 0.013] | 0.325 [0.297, 0.355] |
| C9w10 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.254 [0.213, 0.295] |
| C9w10 | topical | 375 | 1.000 [1.000, 1.000] | 0.304 [0.259, 0.352] | 0.000 [0.000, 0.000] | 0.266 [0.237, 0.294] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.711 [0.687, 0.734] | 0.300 [0.277, 0.323] | 0.280 [0.256, 0.307] | 72.9 / 123.7 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.227 [0.195, 0.261] | 0.190 [0.140, 0.244] | 0.219 [0.152, 0.285] | 66.8 / 78.7 | 0 |
| C0 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.263 [0.239, 0.290] | 0.151 [0.120, 0.184] | 0.034 [0.017, 0.055] | 77.9 / 60.5 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9m50 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.695 [0.670, 0.717] | 0.292 [0.269, 0.315] | 0.265 [0.238, 0.293] | 76.5 / 134.1 | 0 |
| C9m50 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9m50 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.236 [0.203, 0.272] | 0.182 [0.134, 0.232] | 0.204 [0.140, 0.268] | 69.6 / 83.0 | 0 |
| C9m50 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.278 [0.254, 0.305] | 0.138 [0.110, 0.169] | 0.029 [0.013, 0.048] | 81.9 / 63.2 | 0 |
| C9m50 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9m80 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.699 [0.675, 0.723] | 0.264 [0.243, 0.285] | 0.234 [0.209, 0.258] | 73.1 / 129.8 | 0 |
| C9m80 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9m80 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.225 [0.194, 0.258] | 0.170 [0.122, 0.223] | 0.193 [0.133, 0.260] | 66.6 / 79.2 | 0 |
| C9m80 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.279 [0.254, 0.308] | 0.133 [0.103, 0.163] | 0.019 [0.006, 0.035] | 78.4 / 59.9 | 0 |
| C9m80 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C9w10 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.694 [0.670, 0.716] | 0.274 [0.252, 0.296] | 0.246 [0.221, 0.272] | 63.6 / 112.0 | 0 |
| C9w10 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C9w10 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.237 [0.204, 0.273] | 0.182 [0.136, 0.234] | 0.203 [0.139, 0.266] | 58.1 / 68.9 | 0 |
| C9w10 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.279 [0.255, 0.306] | 0.134 [0.105, 0.163] | 0.013 [0.003, 0.026] | 68.0 / 52.6 | 0 |
| C9w10 | hot | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window[C9m50]**: insufficient data — single_trajectory_share Δ -0.134 [-0.153, -0.116] *; window_escape Δ 0.547 [0.512, 0.588] *; affinity_without_copy Δ -0.010 [-0.014, -0.006] *; copy Δ -0.016 [-0.024, -0.008] *; repetition Δ -0.001 [-0.002, 0.000]; pool ECB 4.485 [4.452, 4.515] (floor 4.0); p95 116.7 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — affinity without copies dropped significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window[C9m80]**: insufficient data — single_trajectory_share Δ -0.262 [-0.285, -0.240] *; window_escape Δ 1.208 [1.153, 1.264] *; affinity_without_copy Δ -0.026 [-0.033, -0.019] *; copy Δ -0.041 [-0.053, -0.029] *; repetition Δ -0.001 [-0.002, 0.000]; pool ECB 4.485 [4.452, 4.515] (floor 4.0); p95 111.1 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — affinity without copies dropped significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window[C9w10]**: insufficient data — single_trajectory_share Δ -0.099 [-0.114, -0.084] *; window_escape Δ 0.225 [0.197, 0.254] *; affinity_without_copy Δ -0.019 [-0.025, -0.013] *; copy Δ -0.033 [-0.042, -0.025] *; repetition Δ -0.001 [-0.002, 0.000]; pool ECB 4.485 [4.452, 4.515] (floor 4.0); p95 99.2 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — affinity without copies dropped significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.038 [1.985, 2.091] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C9m50]**: insufficient data — window escape 2.585 [2.525, 2.646] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:27%, 2:21%, 3:25%, 4:20%, 5:6%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C9m80]**: insufficient data — window escape 3.246 [3.183, 3.312] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:14%, 2:16%, 3:21%, 4:31%, 5:19%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C9w10]**: insufficient data — window escape 2.263 [2.206, 2.317] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:30%, 2:29%, 3:26%, 4:12%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 885 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 114.4 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9m50]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9m80]**: pass — 16/18 memes reproduced (89%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C9w10]**: pass — 16/18 memes reproduced (89%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
