# Eval report 2026-09-11 snapshot=phrase-route prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `43bd3af`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C11p2**: 1500 generations
- **C11p3**: 1500 generations
- **C11p5**: 1500 generations
- **C11v**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C11p2 | Δ C11p2 vs C0 | C11p3 | Δ C11p3 vs C0 | C11p5 | Δ C11p5 vs C0 | C11v | Δ C11v vs C0 |
|---|---|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 1.000 [0.999, 1.000] | 0.999 [0.999, 1.000] | -0.000 [-0.001, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [0.999, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.296 [10.064, 10.515] | 10.354 [10.111, 10.608] | 0.058 [-0.209, 0.301] | 10.169 [9.934, 10.393] | -0.127 [-0.377, 0.099] | 10.159 [9.928, 10.389] | -0.137 [-0.367, 0.079] | 10.201 [9.970, 10.428] | -0.095 [-0.164, -0.033] * |
| unique_token_ratio | 0.989 [0.988, 0.991] | 0.988 [0.986, 0.990] | -0.001 [-0.003, 0.001] | 0.989 [0.988, 0.991] | -0.000 [-0.002, 0.002] | 0.990 [0.988, 0.991] | 0.000 [-0.001, 0.002] | 0.990 [0.988, 0.991] | 0.001 [0.000, 0.001] * |
| exact_context_copy_rate | 0.004 [0.001, 0.007] | 0.001 [0.000, 0.002] | -0.003 [-0.007, -0.001] * | 0.001 [0.000, 0.002] | -0.003 [-0.007, -0.001] * | 0.002 [0.000, 0.005] | -0.002 [-0.006, 0.001] | 0.002 [0.000, 0.005] | -0.002 [-0.005, 0.000] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.033 [0.028, 0.040] | 0.096 [0.086, 0.107] | 0.062 [0.053, 0.072] * | 0.086 [0.077, 0.096] | 0.053 [0.044, 0.062] * | 0.072 [0.063, 0.080] | 0.038 [0.031, 0.046] * | 0.031 [0.026, 0.037] | -0.002 [-0.005, 0.000] |
| context_affinity_without_copy | 0.034 [0.028, 0.040] | 0.096 [0.086, 0.107] | 0.062 [0.053, 0.071] * | 0.086 [0.078, 0.096] | 0.053 [0.044, 0.062] * | 0.072 [0.063, 0.080] | 0.038 [0.030, 0.045] * | 0.031 [0.026, 0.037] | -0.002 [-0.005, 0.001] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.069 [0.045, 0.096] | 0.125 [0.093, 0.157] | 0.056 [0.016, 0.096] * | 0.125 [0.093, 0.160] | 0.056 [0.013, 0.096] * | 0.136 [0.101, 0.173] | 0.067 [0.024, 0.104] * | 0.069 [0.043, 0.093] | 0.000 [-0.008, 0.008] |
| structural_pool_ecb | 4.558 [4.527, 4.589] | 4.653 [4.627, 4.678] | 0.095 [0.061, 0.126] * | 4.639 [4.611, 4.665] | 0.081 [0.047, 0.111] * | 4.632 [4.603, 4.659] | 0.074 [0.045, 0.103] * | 4.558 [4.527, 4.589] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.837 [2.785, 2.885] | 2.918 [2.868, 2.971] | 0.081 [0.016, 0.146] * | 2.943 [2.890, 2.994] | 0.107 [0.047, 0.167] * | 2.973 [2.921, 3.020] | 0.137 [0.077, 0.191] * | 2.935 [2.882, 2.983] | 0.098 [0.076, 0.122] * |

C0: distinct-2 = 0.717 (basis 13944), distinct-3 = 0.866 (basis 12444) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.1/23.0 ms; cache_hit_rate: 27%; mean normalized entropy: 0.248 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C11p2: distinct-2 = 0.672 (basis 14031), distinct-3 = 0.824 (basis 12531) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 16.3/25.2 ms; cache_hit_rate: 27%; mean normalized entropy: 0.247 (branching 3.52); mean applied temperature: 2.62; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C11p3: distinct-2 = 0.680 (basis 13754), distinct-3 = 0.833 (basis 12254) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 15.8/24.1 ms; cache_hit_rate: 27%; mean normalized entropy: 0.248 (branching 3.50); mean applied temperature: 2.63; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C11p5: distinct-2 = 0.692 (basis 13739), distinct-3 = 0.846 (basis 12239) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 15.7/24.1 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.45); mean applied temperature: 2.64; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C11v: distinct-2 = 0.721 (basis 13801), distinct-3 = 0.868 (basis 12301) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.3/23.2 ms; cache_hit_rate: 27%; mean normalized entropy: 0.248 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.020, 0.032] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.035 [0.029, 0.043] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.053 [0.031, 0.076] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.022 [0.018, 0.026] |
| C11p2 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.063 [0.054, 0.073] |
| C11p2 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.105 [0.088, 0.122] |
| C11p2 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.170 [0.136, 0.204] |
| C11p2 | topical | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.052 [0.044, 0.060] |
| C11p3 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.066 [0.057, 0.076] |
| C11p3 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.099 [0.082, 0.116] |
| C11p3 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.147 [0.117, 0.179] |
| C11p3 | topical | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.040 [0.034, 0.046] |
| C11p5 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.052 [0.043, 0.061] |
| C11p5 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.095 [0.079, 0.112] |
| C11p5 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.105 [0.078, 0.134] |
| C11p5 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.038 [0.032, 0.044] |
| C11v | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.029] |
| C11v | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.033 [0.027, 0.039] |
| C11v | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.049 [0.028, 0.071] |
| C11v | topical | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.022 [0.018, 0.026] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.603 [0.592, 0.613] | 0.992 [0.987, 0.996] | 0.583 [0.556, 0.608] | 0.027 [0.021, 0.033] | 0.007 [0.002, 0.013] | 14.8 / 14.0 | F4_stale 1 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.096 [0.089, 0.102] | 0.426 [0.400, 0.453] | 0.268 [0.232, 0.302] | 0.019 [0.012, 0.026] | 0.000 [0.000, 0.000] | 13.2 / 16.0 | 0 |
| C0 | extension | 1500 | 0.301 [0.292, 0.311] | 0.847 [0.829, 0.865] | 0.363 [0.337, 0.389] | 0.052 [0.037, 0.068] | 0.000 [0.000, 0.000] | 15.2 / 12.9 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | vanilla | 1500 | 0.421 [0.410, 0.431] | 0.949 [0.937, 0.960] | 0.451 [0.422, 0.476] | 0.027 [0.020, 0.036] | 0.002 [0.000, 0.005] | 16.8 / 17.0 | 0 |
| C11p2 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | mutated | 1500 | 0.071 [0.066, 0.076] | 0.338 [0.315, 0.361] | 0.205 [0.172, 0.239] | 0.014 [0.006, 0.026] | 0.000 [0.000, 0.000] | 15.0 / 17.8 | 0 |
| C11p2 | extension | 1500 | 0.219 [0.210, 0.229] | 0.717 [0.692, 0.741] | 0.305 [0.279, 0.331] | 0.045 [0.028, 0.065] | 0.000 [0.000, 0.000] | 17.4 / 15.4 | 0 |
| C11p2 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | phrase | 1500 | 0.289 [0.280, 0.299] | 0.758 [0.735, 0.783] | 0.375 [0.348, 0.403] | 0.252 [0.230, 0.277] | 0.000 [0.000, 0.000] | 16.6 / 17.6 | F4_stale 4 |
| C11p3 | vanilla | 1500 | 0.447 [0.436, 0.457] | 0.955 [0.945, 0.966] | 0.462 [0.437, 0.488] | 0.025 [0.018, 0.033] | 0.002 [0.000, 0.005] | 16.2 / 16.5 | 0 |
| C11p3 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11p3 | mutated | 1500 | 0.074 [0.069, 0.079] | 0.351 [0.326, 0.374] | 0.217 [0.181, 0.255] | 0.016 [0.007, 0.027] | 0.000 [0.000, 0.000] | 14.6 / 17.1 | 0 |
| C11p3 | extension | 1500 | 0.225 [0.215, 0.234] | 0.727 [0.703, 0.750] | 0.301 [0.276, 0.328] | 0.049 [0.033, 0.067] | 0.000 [0.000, 0.000] | 16.8 / 14.9 | 0 |
| C11p3 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11p3 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11p3 | phrase | 1500 | 0.254 [0.245, 0.265] | 0.665 [0.640, 0.692] | 0.397 [0.368, 0.426] | 0.235 [0.212, 0.259] | 0.000 [0.000, 0.000] | 15.8 / 17.1 | 0 |
| C11p5 | vanilla | 1500 | 0.466 [0.455, 0.477] | 0.960 [0.950, 0.971] | 0.472 [0.447, 0.498] | 0.023 [0.016, 0.031] | 0.003 [0.000, 0.007] | 16.2 / 16.0 | 0 |
| C11p5 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11p5 | mutated | 1500 | 0.078 [0.072, 0.083] | 0.363 [0.339, 0.389] | 0.218 [0.182, 0.251] | 0.020 [0.010, 0.032] | 0.000 [0.000, 0.000] | 14.5 / 17.2 | 0 |
| C11p5 | extension | 1500 | 0.242 [0.233, 0.252] | 0.751 [0.728, 0.773] | 0.333 [0.306, 0.364] | 0.049 [0.033, 0.066] | 0.000 [0.000, 0.000] | 16.7 / 14.5 | 0 |
| C11p5 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11p5 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11p5 | phrase | 1500 | 0.214 [0.205, 0.224] | 0.564 [0.540, 0.590] | 0.387 [0.352, 0.418] | 0.213 [0.189, 0.238] | 0.003 [0.000, 0.009] | 15.7 / 16.8 | 0 |
| C11v | vanilla | 1500 | 0.603 [0.592, 0.613] | 0.992 [0.987, 0.996] | 0.604 [0.579, 0.628] | 0.025 [0.019, 0.031] | 0.003 [0.000, 0.008] | 15.0 / 14.0 | F4_stale 1 |
| C11v | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11v | mutated | 1500 | 0.096 [0.089, 0.102] | 0.426 [0.400, 0.453] | 0.255 [0.221, 0.288] | 0.019 [0.012, 0.027] | 0.000 [0.000, 0.000] | 13.2 / 16.3 | 0 |
| C11v | extension | 1500 | 0.301 [0.292, 0.311] | 0.847 [0.829, 0.865] | 0.345 [0.321, 0.369] | 0.050 [0.036, 0.065] | 0.000 [0.000, 0.000] | 15.3 / 13.2 | 0 |
| C11v | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11v | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11v | phrase | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy[C11p2]**: fail — copy Δ -0.003 [-0.007, -0.001] *; distinct-2 Δ -0.045 [-0.044, -0.006] *; distinct-3 Δ -0.042 [-0.045, 0.004]; affinity_without_copy Δ 0.062 [0.053, 0.071] *; p95 25.2 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11p3]**: fail — copy Δ -0.003 [-0.007, -0.001] *; distinct-2 Δ -0.036 [-0.040, -0.002] *; distinct-3 Δ -0.033 [-0.041, 0.007]; affinity_without_copy Δ 0.053 [0.044, 0.062] *; p95 24.1 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11p5]**: fail — copy Δ -0.002 [-0.006, 0.001]; distinct-2 Δ -0.025 [-0.035, 0.005]; distinct-3 Δ -0.020 [-0.036, 0.014]; affinity_without_copy Δ 0.038 [0.030, 0.045] *; p95 24.1 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11v]**: fail — copy Δ -0.002 [-0.005, 0.000]; distinct-2 Δ 0.004 [-0.018, 0.021]; distinct-3 Δ 0.002 [-0.024, 0.024]; affinity_without_copy Δ -0.002 [-0.005, 0.001]; p95 23.2 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11p2]**: insufficient data — route phrase: present in 75.8% of pools (floor 10%); single_trajectory_share Δ -0.020 [-0.037, -0.005] *; affinity_without_copy Δ 0.062 [0.053, 0.071] *; copy Δ -0.003 [-0.007, -0.001] *; repetition Δ 0.000 [0.000, 0.000]; pool ECB 4.653 (floor 4.0); p95 25.2 ms (budget 150) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11p3]**: insufficient data — route phrase: present in 66.5% of pools (floor 10%); single_trajectory_share Δ -0.017 [-0.033, -0.001] *; affinity_without_copy Δ 0.053 [0.044, 0.062] *; copy Δ -0.003 [-0.007, -0.001] *; repetition Δ 0.000 [0.000, 0.000]; pool ECB 4.639 (floor 4.0); p95 24.1 ms (budget 150) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11p5]**: insufficient data — route phrase: present in 56.4% of pools (floor 10%); single_trajectory_share Δ -0.023 [-0.039, -0.009] *; affinity_without_copy Δ 0.038 [0.030, 0.045] *; copy Δ -0.002 [-0.006, 0.001]; repetition Δ 0.000 [0.000, 0.000]; pool ECB 4.632 (floor 4.0); p95 24.1 ms (budget 150) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11v]**: insufficient data — route under test is not exactly one new route in the arm's pools: none; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.837 [2.785, 2.885] (min 2.0); pool ECB 4.558 [4.527, 4.589] (floor 4.0); pool ECB share 0.912 [0.905, 0.918] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:9%, 2:31%, 3:35%, 4:20%, 5:6%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11p2]**: insufficient data — window escape 2.918 [2.868, 2.971] (min 2.0); pool ECB 4.653 [4.627, 4.678] (floor 4.0); pool ECB share 0.931 [0.925, 0.936] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:7%, 2:28%, 3:38%, 4:22%, 5:6%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11p3]**: insufficient data — window escape 2.943 [2.890, 2.994] (min 2.0); pool ECB 4.639 [4.611, 4.665] (floor 4.0); pool ECB share 0.928 [0.922, 0.933] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:7%, 2:26%, 3:37%, 4:23%, 5:6%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11p5]**: insufficient data — window escape 2.973 [2.921, 3.020] (min 2.0); pool ECB 4.632 [4.603, 4.659] (floor 4.0); pool ECB share 0.926 [0.921, 0.932] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:6%, 2:26%, 3:37%, 4:24%, 5:6%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11v]**: insufficient data — window escape 2.935 [2.882, 2.983] (min 2.0); pool ECB 4.558 [4.527, 4.589] (floor 4.0); pool ECB share 0.912 [0.905, 0.918] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:7%, 2:28%, 3:37%, 4:22%, 5:7%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1504 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 23.0 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 9/18 memes reproduced (50%); C0 9/18 (50%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11p2]**: pass — 10/18 memes reproduced (56%); C0 9/18 (50%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11p3]**: pass — 8/18 memes reproduced (44%); C0 9/18 (50%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11p5]**: pass — 8/18 memes reproduced (44%); C0 9/18 (50%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11v]**: pass — 10/18 memes reproduced (56%); C0 9/18 (50%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
