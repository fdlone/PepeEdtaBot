# Eval report 2026-09-11 snapshot=route-selection prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `4123e21`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C11d**: 1500 generations
- **C11s**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C11d | Δ C11d vs C0 | C11s | Δ C11s vs C0 |
|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.869 [0.862, 0.876] | 0.869 [0.862, 0.876] | 0.000 [0.000, 0.000] | 0.888 [0.882, 0.895] | 0.019 [0.012, 0.026] * |
| mean_response_length | 11.063 [10.786, 11.331] | 10.987 [10.724, 11.239] | -0.077 [-0.232, 0.076] | 11.183 [10.917, 11.439] | 0.119 [-0.172, 0.423] |
| unique_token_ratio | 0.985 [0.983, 0.986] | 0.984 [0.982, 0.986] | -0.000 [-0.002, 0.001] | 0.983 [0.981, 0.985] | -0.002 [-0.004, 0.001] |
| exact_context_copy_rate | 0.211 [0.191, 0.233] | 0.187 [0.168, 0.207] | -0.023 [-0.033, -0.013] * | 0.162 [0.143, 0.181] | -0.049 [-0.071, -0.028] * |
| repetition_rate | 0.002 [0.000, 0.005] | 0.002 [0.000, 0.005] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | -0.001 [-0.004, 0.001] |
| cycle_detection_rate | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.310 [0.294, 0.326] | 0.293 [0.276, 0.309] | -0.017 [-0.022, -0.013] * | 0.322 [0.305, 0.339] | 0.012 [-0.005, 0.032] |
| context_affinity_without_copy | 0.250 [0.234, 0.269] | 0.235 [0.219, 0.254] | -0.010 [-0.014, -0.006] * | 0.274 [0.256, 0.290] | 0.040 [0.021, 0.057] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.267 [0.224, 0.309] | 0.253 [0.208, 0.299] | -0.013 [-0.035, 0.008] | 0.264 [0.221, 0.309] | -0.003 [-0.059, 0.048] |
| structural_pool_ecb | 4.479 [4.443, 4.510] | 4.479 [4.443, 4.510] | 0.000 [0.000, 0.000] | 4.642 [4.613, 4.669] | 0.163 [0.129, 0.199] * |
| structural_window_escape | 2.084 [2.029, 2.134] | 2.535 [2.473, 2.592] | 0.451 [0.413, 0.491] * | 2.535 [2.475, 2.597] | 0.451 [0.381, 0.519] * |

C0: distinct-2 = 0.656 (basis 15095), distinct-3 = 0.816 (basis 13597) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.3/36.3 ms; cache_hit_rate: 41%; mean normalized entropy: 0.248 (branching 3.52); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11d: distinct-2 = 0.662 (basis 14980), distinct-3 = 0.820 (basis 13481) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.9/37.3 ms; cache_hit_rate: 41%; mean normalized entropy: 0.248 (branching 3.52); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11s: distinct-2 = 0.633 (basis 15274), distinct-3 = 0.787 (basis 13774) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.5/35.3 ms; cache_hit_rate: 40%; mean normalized entropy: 0.244 (branching 3.48); mean applied temperature: 2.69; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.208 [0.171, 0.248] | 0.003 [0.000, 0.008] | 0.344 [0.315, 0.374] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.309 [0.261, 0.355] | 0.003 [0.000, 0.008] | 0.374 [0.345, 0.404] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.261 [0.221, 0.304] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.323 [0.275, 0.368] | 0.003 [0.000, 0.008] | 0.256 [0.228, 0.283] |
| C11d | generic | 375 | 1.000 [1.000, 1.000] | 0.192 [0.155, 0.229] | 0.003 [0.000, 0.008] | 0.326 [0.297, 0.356] |
| C11d | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.280 [0.235, 0.325] | 0.003 [0.000, 0.008] | 0.349 [0.320, 0.380] |
| C11d | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.257 [0.217, 0.301] |
| C11d | topical | 375 | 1.000 [1.000, 1.000] | 0.277 [0.235, 0.323] | 0.003 [0.000, 0.008] | 0.235 [0.208, 0.262] |
| C11s | generic | 375 | 1.000 [1.000, 1.000] | 0.168 [0.131, 0.205] | 0.000 [0.000, 0.000] | 0.323 [0.297, 0.349] |
| C11s | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.227 [0.189, 0.267] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.378] |
| C11s | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.008] | 0.384 [0.340, 0.426] |
| C11s | topical | 375 | 1.000 [1.000, 1.000] | 0.251 [0.205, 0.293] | 0.000 [0.000, 0.000] | 0.238 [0.213, 0.263] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.994 [0.990, 0.997] | 0.736 [0.713, 0.757] | 0.287 [0.266, 0.307] | 0.255 [0.230, 0.281] | 24.3 / 28.1 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.099 [0.093, 0.106] | 0.435 [0.412, 0.463] | 0.207 [0.178, 0.237] | 0.203 [0.143, 0.265] | 0.222 [0.148, 0.296] | 21.5 / 26.4 | 0 |
| C0 | extension | 1500 | 0.250 [0.240, 0.259] | 0.747 [0.724, 0.769] | 0.239 [0.215, 0.265] | 0.154 [0.118, 0.191] | 0.022 [0.007, 0.045] | 25.7 / 20.2 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.994 [0.990, 0.997] | 0.714 [0.691, 0.736] | 0.273 [0.252, 0.296] | 0.230 [0.208, 0.255] | 24.8 / 29.3 | 0 |
| C11d | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | mutated | 1500 | 0.099 [0.093, 0.106] | 0.435 [0.412, 0.463] | 0.213 [0.182, 0.242] | 0.186 [0.131, 0.239] | 0.216 [0.151, 0.288] | 21.9 / 27.1 | 0 |
| C11d | extension | 1500 | 0.250 [0.240, 0.259] | 0.747 [0.724, 0.769] | 0.265 [0.239, 0.291] | 0.148 [0.118, 0.178] | 0.020 [0.007, 0.037] | 26.3 / 20.6 | 0 |
| C11d | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | vanilla | 1500 | 0.472 [0.461, 0.482] | 0.971 [0.963, 0.979] | 0.534 [0.509, 0.559] | 0.281 [0.254, 0.309] | 0.279 [0.246, 0.310] | 24.2 / 27.0 | 0 |
| C11s | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | mutated | 1500 | 0.061 [0.056, 0.067] | 0.289 [0.266, 0.313] | 0.198 [0.164, 0.233] | 0.201 [0.137, 0.272] | 0.163 [0.081, 0.244] | 20.7 / 25.7 | 0 |
| C11s | extension | 1500 | 0.178 [0.169, 0.187] | 0.615 [0.592, 0.639] | 0.220 [0.194, 0.248] | 0.123 [0.090, 0.160] | 0.025 [0.005, 0.049] | 25.6 / 22.1 | 0 |
| C11s | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | phrase | 1500 | 0.289 [0.280, 0.299] | 0.758 [0.735, 0.783] | 0.381 [0.353, 0.407] | 0.343 [0.318, 0.372] | 0.016 [0.005, 0.030] | 23.7 / 25.9 | F4_stale 4 |

## Gates

- **phase2_entropy[C11d]**: fail — copy Δ -0.023 [-0.033, -0.013] *; distinct-2 Δ 0.006 [-0.016, 0.021]; distinct-3 Δ 0.004 [-0.022, 0.027]; affinity_without_copy Δ -0.010 [-0.014, -0.006] *; p95 37.3 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; affinity without copies dropped significantly
- **phase2_entropy[C11s]**: fail — copy Δ -0.049 [-0.071, -0.028] *; distinct-2 Δ -0.023 [-0.032, 0.006]; distinct-3 Δ -0.028 [-0.038, 0.011]; affinity_without_copy Δ 0.040 [0.021, 0.057] *; p95 35.3 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11d]**: insufficient data — route under test is not exactly one new route in the arm's pools: none; gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11s]**: insufficient data — route phrase: present in 75.8% of pools (floor 10%); single_trajectory_share Δ -0.159 [-0.189, -0.130] *; affinity_without_copy Δ 0.040 [0.021, 0.057] *; copy Δ -0.049 [-0.071, -0.028] *; repetition Δ -0.001 [-0.004, 0.001]; pool ECB 4.642 (floor 4.0); p95 35.3 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted); gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.084 [2.029, 2.134] (min 2.0); pool ECB 4.479 [4.443, 4.510] (floor 4.0); pool ECB share 0.896 [0.889, 0.902] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:39%, 2:28%, 3:22%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11d]**: insufficient data — window escape 2.535 [2.473, 2.592] (min 2.0); pool ECB 4.479 [4.443, 4.510] (floor 4.0); pool ECB share 0.896 [0.889, 0.902] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:25%, 2:24%, 3:28%, 4:17%, 5:5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11s]**: insufficient data — window escape 2.535 [2.475, 2.597] (min 2.0); pool ECB 4.642 [4.613, 4.669] (floor 4.0); pool ECB share 0.928 [0.923, 0.934] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:23%, 2:28%, 3:28%, 4:17%, 5:5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.002] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 860 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 36.3 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 15/18 memes reproduced (83%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11d]**: pass — 15/18 memes reproduced (83%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11s]**: pass — 17/18 memes reproduced (94%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
