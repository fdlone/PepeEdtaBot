# Eval report 2026-09-11 snapshot=phrase-route prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `43bd3af`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
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
| candidate_accept_rate | 0.869 [0.862, 0.876] | 0.888 [0.882, 0.895] | 0.019 [0.012, 0.026] * | 0.888 [0.880, 0.894] | 0.019 [0.011, 0.026] * | 0.884 [0.876, 0.891] | 0.015 [0.009, 0.021] * | 0.869 [0.862, 0.876] | 0.000 [0.000, 0.000] |
| mean_response_length | 11.059 [10.782, 11.334] | 11.151 [10.873, 11.417] | 0.091 [-0.202, 0.365] | 11.073 [10.798, 11.340] | 0.013 [-0.251, 0.271] | 11.049 [10.783, 11.307] | -0.011 [-0.271, 0.243] | 10.931 [10.663, 11.192] | -0.128 [-0.218, -0.051] * |
| unique_token_ratio | 0.985 [0.983, 0.986] | 0.984 [0.983, 0.986] | -0.000 [-0.003, 0.002] | 0.985 [0.983, 0.987] | 0.000 [-0.002, 0.002] | 0.984 [0.982, 0.986] | -0.000 [-0.002, 0.002] | 0.985 [0.983, 0.987] | 0.001 [0.000, 0.001] * |
| exact_context_copy_rate | 0.211 [0.191, 0.233] | 0.180 [0.161, 0.199] | -0.031 [-0.053, -0.010] * | 0.174 [0.155, 0.195] | -0.037 [-0.057, -0.017] * | 0.190 [0.170, 0.211] | -0.021 [-0.039, -0.002] * | 0.210 [0.189, 0.231] | -0.001 [-0.005, 0.003] |
| repetition_rate | 0.002 [0.000, 0.005] | 0.001 [0.000, 0.002] | -0.001 [-0.004, 0.001] | 0.001 [0.000, 0.003] | -0.001 [-0.003, 0.001] | 0.002 [0.000, 0.005] | 0.000 [-0.003, 0.003] | 0.002 [0.000, 0.005] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.310 [0.294, 0.326] | 0.342 [0.326, 0.358] | 0.032 [0.016, 0.050] * | 0.328 [0.312, 0.344] | 0.018 [0.003, 0.034] * | 0.316 [0.302, 0.333] | 0.006 [-0.008, 0.021] | 0.310 [0.294, 0.326] | 0.000 [-0.002, 0.002] |
| context_affinity_without_copy | 0.250 [0.234, 0.269] | 0.294 [0.277, 0.311] | 0.057 [0.039, 0.075] * | 0.281 [0.264, 0.298] | 0.042 [0.027, 0.057] * | 0.263 [0.246, 0.280] | 0.019 [0.005, 0.032] * | 0.250 [0.233, 0.269] | 0.000 [-0.001, 0.003] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.267 [0.224, 0.309] | 0.272 [0.227, 0.317] | 0.005 [-0.051, 0.056] | 0.280 [0.235, 0.325] | 0.013 [-0.043, 0.061] | 0.291 [0.243, 0.336] | 0.024 [-0.024, 0.064] | 0.267 [0.224, 0.309] | 0.000 [-0.011, 0.011] |
| structural_pool_ecb | 4.479 [4.443, 4.510] | 4.642 [4.613, 4.669] | 0.163 [0.129, 0.199] * | 4.631 [4.601, 4.659] | 0.153 [0.121, 0.184] * | 4.626 [4.595, 4.653] | 0.147 [0.119, 0.177] * | 4.479 [4.443, 4.510] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.079 [2.025, 2.129] | 2.104 [2.047, 2.160] | 0.025 [-0.038, 0.086] | 2.111 [2.054, 2.163] | 0.033 [-0.026, 0.085] | 2.132 [2.076, 2.187] | 0.053 [0.001, 0.102] * | 2.116 [2.057, 2.168] | 0.037 [0.019, 0.057] * |

C0: distinct-2 = 0.656 (basis 15089), distinct-3 = 0.816 (basis 13591) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.5/36.5 ms; cache_hit_rate: 41%; mean normalized entropy: 0.248 (branching 3.52); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11p2: distinct-2 = 0.622 (basis 15226), distinct-3 = 0.778 (basis 13726) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.5/35.5 ms; cache_hit_rate: 40%; mean normalized entropy: 0.244 (branching 3.48); mean applied temperature: 2.69; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11p3: distinct-2 = 0.631 (basis 15109), distinct-3 = 0.789 (basis 13609) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.4/35.3 ms; cache_hit_rate: 41%; mean normalized entropy: 0.241 (branching 3.47); mean applied temperature: 2.71; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11p5: distinct-2 = 0.640 (basis 15073), distinct-3 = 0.797 (basis 13574) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 24.2/36.8 ms; cache_hit_rate: 40%; mean normalized entropy: 0.244 (branching 3.52); mean applied temperature: 2.72; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11v: distinct-2 = 0.656 (basis 14897), distinct-3 = 0.815 (basis 13398) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 24.2/38.2 ms; cache_hit_rate: 41%; mean normalized entropy: 0.248 (branching 3.52); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.208 [0.171, 0.248] | 0.003 [0.000, 0.008] | 0.344 [0.315, 0.374] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.309 [0.261, 0.355] | 0.003 [0.000, 0.008] | 0.374 [0.345, 0.404] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.261 [0.221, 0.304] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.323 [0.275, 0.368] | 0.003 [0.000, 0.008] | 0.256 [0.228, 0.283] |
| C11p2 | generic | 375 | 1.000 [1.000, 1.000] | 0.179 [0.139, 0.216] | 0.000 [0.000, 0.000] | 0.347 [0.320, 0.372] |
| C11p2 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.272 [0.227, 0.320] | 0.000 [0.000, 0.000] | 0.378 [0.348, 0.406] |
| C11p2 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.008] | 0.392 [0.349, 0.434] |
| C11p2 | topical | 375 | 1.000 [1.000, 1.000] | 0.267 [0.221, 0.309] | 0.000 [0.000, 0.000] | 0.256 [0.230, 0.281] |
| C11p3 | generic | 375 | 1.000 [1.000, 1.000] | 0.173 [0.133, 0.213] | 0.000 [0.000, 0.000] | 0.344 [0.317, 0.370] |
| C11p3 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.229 [0.187, 0.272] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.377] |
| C11p3 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.008] | 0.364 [0.321, 0.406] |
| C11p3 | topical | 375 | 1.000 [1.000, 1.000] | 0.291 [0.251, 0.339] | 0.003 [0.000, 0.008] | 0.257 [0.232, 0.283] |
| C11p5 | generic | 375 | 1.000 [1.000, 1.000] | 0.173 [0.133, 0.213] | 0.003 [0.000, 0.008] | 0.335 [0.308, 0.359] |
| C11p5 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.277 [0.232, 0.325] | 0.003 [0.000, 0.008] | 0.363 [0.333, 0.391] |
| C11p5 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.311 [0.265, 0.352] |
| C11p5 | topical | 375 | 1.000 [1.000, 1.000] | 0.307 [0.261, 0.355] | 0.003 [0.000, 0.008] | 0.256 [0.231, 0.283] |
| C11v | generic | 375 | 1.000 [1.000, 1.000] | 0.213 [0.176, 0.253] | 0.003 [0.000, 0.008] | 0.343 [0.313, 0.373] |
| C11v | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.309 [0.261, 0.355] | 0.003 [0.000, 0.008] | 0.376 [0.346, 0.405] |
| C11v | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.265 [0.226, 0.308] |
| C11v | topical | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.363] | 0.003 [0.000, 0.008] | 0.253 [0.225, 0.278] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.994 [0.990, 0.997] | 0.736 [0.714, 0.757] | 0.287 [0.266, 0.307] | 0.255 [0.230, 0.280] | 24.3 / 27.7 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.099 [0.093, 0.106] | 0.435 [0.412, 0.463] | 0.208 [0.179, 0.239] | 0.203 [0.143, 0.265] | 0.221 [0.154, 0.301] | 21.5 / 26.5 | 0 |
| C0 | extension | 1500 | 0.250 [0.240, 0.259] | 0.747 [0.724, 0.769] | 0.237 [0.213, 0.263] | 0.154 [0.118, 0.191] | 0.023 [0.008, 0.041] | 25.6 / 20.4 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | vanilla | 1500 | 0.472 [0.461, 0.482] | 0.971 [0.963, 0.979] | 0.550 [0.523, 0.574] | 0.305 [0.278, 0.332] | 0.305 [0.275, 0.337] | 24.3 / 26.7 | 0 |
| C11p2 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | mutated | 1500 | 0.061 [0.056, 0.067] | 0.289 [0.266, 0.313] | 0.200 [0.166, 0.240] | 0.227 [0.159, 0.299] | 0.184 [0.103, 0.276] | 20.8 / 25.8 | 0 |
| C11p2 | extension | 1500 | 0.178 [0.169, 0.187] | 0.615 [0.592, 0.639] | 0.192 [0.168, 0.218] | 0.143 [0.106, 0.186] | 0.023 [0.006, 0.045] | 25.6 / 22.3 | 0 |
| C11p2 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11p2 | phrase | 1500 | 0.289 [0.280, 0.299] | 0.758 [0.735, 0.783] | 0.383 [0.355, 0.410] | 0.349 [0.323, 0.375] | 0.014 [0.005, 0.028] | 23.9 / 25.9 | F4_stale 4 |
| C11p3 | vanilla | 1500 | 0.491 [0.480, 0.503] | 0.973 [0.965, 0.981] | 0.576 [0.550, 0.602] | 0.305 [0.279, 0.329] | 0.281 [0.251, 0.310] | 24.2 / 26.4 | 0 |
| C11p3 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11p3 | mutated | 1500 | 0.065 [0.059, 0.070] | 0.301 [0.278, 0.325] | 0.204 [0.168, 0.246] | 0.172 [0.113, 0.233] | 0.196 [0.120, 0.272] | 20.8 / 25.7 | 0 |
| C11p3 | extension | 1500 | 0.190 [0.180, 0.199] | 0.633 [0.609, 0.655] | 0.200 [0.174, 0.227] | 0.144 [0.107, 0.185] | 0.032 [0.011, 0.058] | 26.0 / 21.3 | 0 |
| C11p3 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11p3 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11p3 | phrase | 1500 | 0.254 [0.245, 0.265] | 0.665 [0.640, 0.692] | 0.378 [0.349, 0.407] | 0.328 [0.301, 0.354] | 0.003 [0.000, 0.008] | 23.3 / 26.2 | 0 |
| C11p5 | vanilla | 1500 | 0.519 [0.508, 0.530] | 0.980 [0.973, 0.986] | 0.615 [0.590, 0.639] | 0.293 [0.269, 0.319] | 0.280 [0.252, 0.310] | 24.9 / 28.7 | 0 |
| C11p5 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11p5 | mutated | 1500 | 0.068 [0.062, 0.073] | 0.311 [0.288, 0.335] | 0.218 [0.178, 0.257] | 0.164 [0.105, 0.231] | 0.216 [0.137, 0.304] | 21.8 / 26.4 | 0 |
| C11p5 | extension | 1500 | 0.199 [0.189, 0.208] | 0.647 [0.622, 0.669] | 0.204 [0.180, 0.228] | 0.147 [0.109, 0.187] | 0.040 [0.015, 0.071] | 26.9 / 21.4 | 0 |
| C11p5 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11p5 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11p5 | phrase | 1500 | 0.214 [0.205, 0.224] | 0.564 [0.540, 0.590] | 0.350 [0.319, 0.383] | 0.296 [0.267, 0.326] | 0.007 [0.000, 0.017] | 23.9 / 26.4 | 0 |
| C11v | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.994 [0.990, 0.997] | 0.748 [0.726, 0.769] | 0.282 [0.260, 0.305] | 0.250 [0.224, 0.276] | 25.1 / 30.1 | 0 |
| C11v | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11v | mutated | 1500 | 0.099 [0.093, 0.106] | 0.435 [0.412, 0.463] | 0.196 [0.167, 0.227] | 0.213 [0.154, 0.277] | 0.234 [0.164, 0.312] | 22.2 / 27.4 | 0 |
| C11v | extension | 1500 | 0.250 [0.240, 0.259] | 0.747 [0.724, 0.769] | 0.228 [0.204, 0.253] | 0.157 [0.125, 0.193] | 0.023 [0.008, 0.043] | 26.6 / 20.7 | 0 |
| C11v | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11v | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11v | phrase | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy[C11p2]**: fail — copy Δ -0.031 [-0.053, -0.010] *; distinct-2 Δ -0.034 [-0.040, -0.003] *; distinct-3 Δ -0.037 [-0.043, 0.004]; affinity_without_copy Δ 0.057 [0.039, 0.075] *; p95 35.5 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11p3]**: fail — copy Δ -0.037 [-0.057, -0.017] *; distinct-2 Δ -0.025 [-0.034, 0.003]; distinct-3 Δ -0.027 [-0.037, 0.011]; affinity_without_copy Δ 0.042 [0.027, 0.057] *; p95 35.3 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11p5]**: fail — copy Δ -0.021 [-0.039, -0.002] *; distinct-2 Δ -0.016 [-0.027, 0.010]; distinct-3 Δ -0.019 [-0.033, 0.016]; affinity_without_copy Δ 0.019 [0.005, 0.032] *; p95 36.8 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11v]**: fail — copy Δ -0.001 [-0.005, 0.003]; distinct-2 Δ 0.001 [-0.019, 0.019]; distinct-3 Δ -0.001 [-0.025, 0.024]; affinity_without_copy Δ 0.000 [-0.001, 0.003]; p95 38.2 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11p2]**: insufficient data — route phrase: present in 75.8% of pools (floor 10%); single_trajectory_share Δ -0.013 [-0.040, 0.015]; affinity_without_copy Δ 0.057 [0.039, 0.075] *; copy Δ -0.031 [-0.053, -0.010] *; repetition Δ -0.001 [-0.004, 0.001]; pool ECB 4.642 (floor 4.0); p95 35.5 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11p3]**: insufficient data — route phrase: present in 66.5% of pools (floor 10%); single_trajectory_share Δ -0.015 [-0.041, 0.013]; affinity_without_copy Δ 0.042 [0.027, 0.057] *; copy Δ -0.037 [-0.057, -0.017] *; repetition Δ -0.001 [-0.003, 0.001]; pool ECB 4.631 (floor 4.0); p95 35.3 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11p5]**: insufficient data — route phrase: present in 56.4% of pools (floor 10%); single_trajectory_share Δ -0.025 [-0.049, 0.001]; affinity_without_copy Δ 0.019 [0.005, 0.032] *; copy Δ -0.021 [-0.039, -0.002] *; repetition Δ 0.000 [-0.003, 0.003]; pool ECB 4.626 (floor 4.0); p95 36.8 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11v]**: insufficient data — route under test is not exactly one new route in the arm's pools: none; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.079 [2.025, 2.129] (min 2.0); pool ECB 4.479 [4.443, 4.510] (floor 4.0); pool ECB share 0.896 [0.889, 0.902] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:39%, 2:28%, 3:22%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11p2]**: insufficient data — window escape 2.104 [2.047, 2.160] (min 2.0); pool ECB 4.642 [4.613, 4.669] (floor 4.0); pool ECB share 0.928 [0.923, 0.934] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:37%, 2:29%, 3:21%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11p3]**: insufficient data — window escape 2.111 [2.054, 2.163] (min 2.0); pool ECB 4.631 [4.601, 4.659] (floor 4.0); pool ECB share 0.926 [0.920, 0.932] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:37%, 2:29%, 3:22%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11p5]**: insufficient data — window escape 2.132 [2.076, 2.187] (min 2.0); pool ECB 4.626 [4.595, 4.653] (floor 4.0); pool ECB share 0.925 [0.919, 0.931] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:36%, 2:29%, 3:23%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11v]**: insufficient data — window escape 2.116 [2.057, 2.168] (min 2.0); pool ECB 4.479 [4.443, 4.510] (floor 4.0); pool ECB share 0.896 [0.889, 0.902] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:27%, 3:23%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.002] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 857 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 36.5 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 15/18 memes reproduced (83%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11p2]**: pass — 16/18 memes reproduced (89%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11p3]**: pass — 16/18 memes reproduced (89%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11p5]**: pass — 16/18 memes reproduced (89%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11v]**: pass — 15/18 memes reproduced (83%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
