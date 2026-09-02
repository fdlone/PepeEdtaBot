# Eval report 2026-09-02 snapshot=pool-grid1-v2 prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `1d36e26`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C8b30**: 1500 generations
- **C8b40**: 1500 generations
- **C8s40**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C8b30 | Δ C8b30 vs C0 | C8b40 | Δ C8b40 vs C0 | C8s40 | Δ C8s40 vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.870 [0.862, 0.877] | 0.845 [0.838, 0.852] | -0.025 [-0.029, -0.020] * | 0.824 [0.816, 0.832] | -0.046 [-0.051, -0.041] * | 0.894 [0.888, 0.900] | 0.025 [0.021, 0.029] * |
| mean_response_length | 10.917 [10.691, 11.185] | 10.907 [10.667, 11.166] | -0.010 [-0.210, 0.189] | 11.150 [10.885, 11.434] | 0.233 [-0.025, 0.497] | 11.264 [10.999, 11.557] | 0.347 [0.191, 0.515] * |
| unique_token_ratio | 0.984 [0.981, 0.986] | 0.983 [0.981, 0.985] | -0.000 [-0.002, 0.001] | 0.983 [0.981, 0.985] | -0.000 [-0.002, 0.001] | 0.984 [0.982, 0.986] | 0.000 [-0.001, 0.001] |
| exact_context_copy_rate | 0.226 [0.205, 0.247] | 0.254 [0.232, 0.276] | 0.028 [0.014, 0.041] * | 0.279 [0.256, 0.301] | 0.053 [0.036, 0.069] * | 0.233 [0.211, 0.255] | 0.007 [-0.004, 0.018] |
| repetition_rate | 0.002 [0.000, 0.005] | 0.002 [0.000, 0.005] | 0.000 [-0.001, 0.002] | 0.002 [0.000, 0.005] | 0.000 [-0.001, 0.002] | 0.001 [0.000, 0.003] | -0.001 [-0.002, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.321 [0.305, 0.336] | 0.347 [0.331, 0.362] | 0.026 [0.016, 0.037] * | 0.368 [0.353, 0.383] | 0.047 [0.035, 0.061] * | 0.355 [0.340, 0.371] | 0.035 [0.026, 0.044] * |
| context_affinity_without_copy | 0.253 [0.235, 0.273] | 0.276 [0.258, 0.295] | 0.018 [0.006, 0.030] * | 0.295 [0.275, 0.315] | 0.029 [0.014, 0.044] * | 0.288 [0.269, 0.307] | 0.032 [0.023, 0.041] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.285 [0.243, 0.333] | 0.285 [0.243, 0.336] | 0.000 [-0.027, 0.027] | 0.299 [0.256, 0.349] | 0.013 [-0.021, 0.048] | 0.283 [0.240, 0.328] | -0.003 [-0.029, 0.024] |
| structural_pool_ecb | 4.485 [4.452, 4.515] | 4.466 [4.431, 4.501] | -0.019 [-0.043, 0.005] | 4.449 [4.415, 4.484] | -0.035 [-0.068, -0.004] * | 4.469 [4.437, 4.500] | -0.016 [-0.037, 0.005] |
| structural_window_escape | 2.038 [1.985, 2.091] | 2.005 [1.956, 2.061] | -0.033 [-0.077, 0.010] | 1.989 [1.936, 2.044] | -0.049 [-0.098, -0.001] * | 1.996 [1.947, 2.053] | -0.042 [-0.077, -0.011] * |

C0: distinct-2 = 0.661 (basis 14875), distinct-3 = 0.818 (basis 13376) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 28.0/48.5 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8b30: distinct-2 = 0.657 (basis 14860), distinct-3 = 0.816 (basis 13361) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 25.3/40.9 ms; cache_hit_rate: 44%; mean normalized entropy: 0.244 (branching 3.47); mean applied temperature: 2.79; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8b40: distinct-2 = 0.642 (basis 15225), distinct-3 = 0.803 (basis 13726) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 55.7/94.1 ms; cache_hit_rate: 47%; mean normalized entropy: 0.245 (branching 3.52); mean applied temperature: 2.81; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8s40: distinct-2 = 0.653 (basis 15396), distinct-3 = 0.812 (basis 13897) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 59.3/103.5 ms; cache_hit_rate: 40%; mean normalized entropy: 0.242 (branching 3.53); mean applied temperature: 2.75; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.232 [0.189, 0.275] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.382] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.360] | 0.008 [0.000, 0.019] | 0.364 [0.337, 0.392] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.266 [0.225, 0.306] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.355 [0.307, 0.403] | 0.000 [0.000, 0.000] | 0.297 [0.269, 0.324] |
| C8b30 | generic | 375 | 1.000 [1.000, 1.000] | 0.259 [0.213, 0.307] | 0.003 [0.000, 0.008] | 0.389 [0.361, 0.419] |
| C8b30 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.349 [0.304, 0.395] | 0.005 [0.000, 0.013] | 0.400 [0.368, 0.429] |
| C8b30 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.264 [0.223, 0.302] |
| C8b30 | topical | 375 | 1.000 [1.000, 1.000] | 0.405 [0.363, 0.456] | 0.000 [0.000, 0.000] | 0.326 [0.299, 0.353] |
| C8b40 | generic | 375 | 1.000 [1.000, 1.000] | 0.275 [0.232, 0.320] | 0.003 [0.000, 0.008] | 0.413 [0.382, 0.442] |
| C8b40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.405 [0.357, 0.453] | 0.005 [0.000, 0.013] | 0.437 [0.406, 0.467] |
| C8b40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.274 [0.233, 0.315] |
| C8b40 | topical | 375 | 1.000 [1.000, 1.000] | 0.432 [0.381, 0.483] | 0.000 [0.000, 0.000] | 0.339 [0.311, 0.366] |
| C8s40 | generic | 375 | 1.000 [1.000, 1.000] | 0.237 [0.195, 0.280] | 0.000 [0.000, 0.000] | 0.403 [0.373, 0.433] |
| C8s40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.325 [0.277, 0.371] | 0.005 [0.000, 0.013] | 0.399 [0.372, 0.428] |
| C8s40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.280 [0.236, 0.321] |
| C8s40 | topical | 375 | 1.000 [1.000, 1.000] | 0.365 [0.317, 0.413] | 0.000 [0.000, 0.000] | 0.332 [0.302, 0.359] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.711 [0.687, 0.734] | 0.300 [0.277, 0.323] | 0.280 [0.256, 0.307] | 29.5 / 56.5 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.227 [0.195, 0.261] | 0.190 [0.140, 0.244] | 0.219 [0.152, 0.285] | 27.0 / 31.9 | 0 |
| C0 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.263 [0.239, 0.290] | 0.151 [0.120, 0.184] | 0.034 [0.017, 0.055] | 31.5 / 24.6 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C8b30 | vanilla | 1500 | 0.666 [0.655, 0.677] | 0.994 [0.989, 0.998] | 0.728 [0.707, 0.751] | 0.328 [0.306, 0.352] | 0.310 [0.285, 0.337] | 26.2 / 57.6 | 0 |
| C8b30 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8b30 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.442 [0.418, 0.467] | 0.235 [0.205, 0.268] | 0.229 [0.176, 0.284] | 0.224 [0.167, 0.295] | 23.8 / 28.5 | 0 |
| C8b30 | extension | 1500 | 0.238 [0.227, 0.248] | 0.715 [0.692, 0.737] | 0.241 [0.216, 0.265] | 0.141 [0.106, 0.175] | 0.035 [0.016, 0.058] | 28.3 / 21.7 | 0 |
| C8b30 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C8b40 | vanilla | 1500 | 0.674 [0.663, 0.685] | 0.995 [0.991, 0.999] | 0.744 [0.723, 0.765] | 0.335 [0.312, 0.359] | 0.329 [0.302, 0.357] | 58.3 / 99.7 | 0 |
| C8b40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8b40 | mutated | 1500 | 0.098 [0.093, 0.105] | 0.445 [0.421, 0.469] | 0.222 [0.190, 0.256] | 0.273 [0.214, 0.329] | 0.270 [0.203, 0.345] | 52.7 / 63.1 | 0 |
| C8b40 | extension | 1500 | 0.228 [0.218, 0.238] | 0.699 [0.677, 0.722] | 0.230 [0.206, 0.255] | 0.169 [0.132, 0.209] | 0.050 [0.025, 0.075] | 62.5 / 49.2 | 0 |
| C8b40 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C8s40 | vanilla | 1500 | 0.664 [0.653, 0.675] | 0.993 [0.989, 0.997] | 0.738 [0.717, 0.758] | 0.342 [0.318, 0.366] | 0.283 [0.255, 0.309] | 63.1 / 116.8 | 0 |
| C8s40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8s40 | mutated | 1500 | 0.099 [0.093, 0.105] | 0.453 [0.429, 0.479] | 0.224 [0.193, 0.253] | 0.188 [0.138, 0.239] | 0.211 [0.151, 0.276] | 58.4 / 67.7 | 0 |
| C8s40 | extension | 1500 | 0.237 [0.228, 0.248] | 0.715 [0.691, 0.738] | 0.232 [0.208, 0.258] | 0.160 [0.124, 0.199] | 0.024 [0.008, 0.044] | 67.6 / 53.1 | 0 |
| C8s40 | hot | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8b30]**: insufficient data — coverage: context starts 58.9% vs 52.4% (shift +0.065; floor ±5%); affinity_without_copy Δ 0.018 [0.006, 0.030] *; window_escape Δ -0.033 [-0.077, 0.010]; pool ECB 4.466 [4.431, 4.501] (floor 4.0); copy Δ 0.028 [0.014, 0.041] *; repetition Δ 0.000 [-0.001, 0.002]; p95 40.9 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — copy rose significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8b40]**: insufficient data — coverage: context starts 63.1% vs 52.4% (shift +0.107; floor ±5%); affinity_without_copy Δ 0.029 [0.014, 0.044] *; window_escape Δ -0.049 [-0.098, -0.001] *; pool ECB 4.449 [4.415, 4.484] (floor 4.0); copy Δ 0.053 [0.036, 0.069] *; repetition Δ 0.000 [-0.001, 0.002]; p95 94.1 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — window escape dropped significantly; copy rose significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8s40]**: insufficient data — coverage: context starts 58.5% vs 52.4% (shift +0.061; floor ±5%); affinity_without_copy Δ 0.032 [0.023, 0.041] *; window_escape Δ -0.042 [-0.077, -0.011] *; pool ECB 4.469 [4.437, 4.500] (floor 4.0); copy Δ 0.007 [-0.004, 0.018]; repetition Δ -0.001 [-0.002, 0.000]; p95 103.5 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — window escape dropped significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.038 [1.985, 2.091] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8b30]**: insufficient data — window escape 2.005 [1.956, 2.061] (min 2.0); pool ECB 4.466 [4.431, 4.501] (floor 4.0); pool ECB share 0.893 [0.886, 0.900] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:42%, 2:28%, 3:20%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8b40]**: insufficient data — window escape 1.989 [1.936, 2.044] (min 2.0); pool ECB 4.449 [4.415, 4.484] (floor 4.0); pool ECB share 0.890 [0.883, 0.897] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:42%, 2:27%, 3:21%, 4:7%, 5:2% — window escape below the registered minimum; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8s40]**: insufficient data — window escape 1.996 [1.947, 2.053] (min 2.0); pool ECB 4.469 [4.437, 4.500] (floor 4.0); pool ECB share 0.894 [0.887, 0.900] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:31%, 3:20%, 4:7%, 5:2% — window escape below the registered minimum; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 885 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 48.5 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C8b30]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C8b40]**: pass — 16/18 memes reproduced (89%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C8s40]**: pass — 16/18 memes reproduced (89%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
