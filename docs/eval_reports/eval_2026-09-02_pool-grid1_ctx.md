# Eval report 2026-09-02 snapshot=pool-grid1 prompts=308b7deaea0f seeds=42,1337,2026 mode=ctx

Revision: `25da193`. Generations per configuration: 500.
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
| candidate_accept_rate | 0.881 [0.874, 0.888] | 0.858 [0.850, 0.865] | -0.023 [-0.027, -0.019] * | 0.842 [0.834, 0.850] | -0.038 [-0.043, -0.033] * | 0.906 [0.900, 0.912] | 0.025 [0.021, 0.029] * |
| mean_response_length | 11.152 [10.888, 11.417] | 11.147 [10.898, 11.401] | -0.005 [-0.194, 0.181] | 11.145 [10.891, 11.407] | -0.007 [-0.243, 0.221] | 11.203 [10.945, 11.489] | 0.051 [-0.093, 0.181] |
| unique_token_ratio | 0.987 [0.985, 0.988] | 0.987 [0.985, 0.989] | 0.000 [-0.001, 0.002] | 0.986 [0.985, 0.988] | -0.000 [-0.002, 0.002] | 0.986 [0.984, 0.988] | -0.001 [-0.002, 0.001] |
| exact_context_copy_rate | 0.219 [0.198, 0.239] | 0.261 [0.239, 0.282] | 0.041 [0.026, 0.057] * | 0.273 [0.251, 0.297] | 0.054 [0.036, 0.070] * | 0.220 [0.201, 0.241] | 0.001 [-0.011, 0.011] |
| repetition_rate | 0.003 [0.001, 0.005] | 0.001 [0.000, 0.003] | -0.001 [-0.003, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.003, 0.000] | 0.001 [0.000, 0.002] | -0.002 [-0.005, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.279 [0.264, 0.295] | 0.308 [0.293, 0.323] | 0.029 [0.019, 0.039] * | 0.322 [0.307, 0.337] | 0.043 [0.032, 0.054] * | 0.308 [0.293, 0.323] | 0.028 [0.020, 0.036] * |
| context_affinity_without_copy | 0.211 [0.196, 0.228] | 0.232 [0.216, 0.249] | 0.013 [0.003, 0.023] * | 0.247 [0.230, 0.264] | 0.026 [0.013, 0.039] * | 0.243 [0.227, 0.259] | 0.031 [0.023, 0.039] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.309 [0.264, 0.355] | 0.328 [0.283, 0.376] | 0.019 [-0.003, 0.043] | 0.347 [0.301, 0.397] | 0.037 [0.005, 0.069] * | 0.317 [0.269, 0.360] | 0.008 [-0.013, 0.032] |
| structural_pool_ecb | 4.430 [4.395, 4.464] | 4.411 [4.373, 4.446] | -0.019 [-0.047, 0.009] | 4.369 [4.331, 4.407] | -0.061 [-0.093, -0.025] * | 4.437 [4.405, 4.469] | 0.007 [-0.013, 0.029] |
| structural_window_escape | 2.081 [2.027, 2.135] | 2.035 [1.984, 2.089] | -0.046 [-0.087, -0.003] * | 2.055 [2.005, 2.113] | -0.026 [-0.074, 0.027] | 2.063 [2.008, 2.116] | -0.019 [-0.051, 0.014] |

C0: distinct-2 = 0.652 (basis 15228), distinct-3 = 0.804 (basis 13730) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 22.4/33.8 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8b30: distinct-2 = 0.643 (basis 15220), distinct-3 = 0.799 (basis 13721) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 22.4/34.6 ms; cache_hit_rate: 44%; mean normalized entropy: 0.247 (branching 3.55); mean applied temperature: 2.78; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8b40: distinct-2 = 0.641 (basis 15217), distinct-3 = 0.797 (basis 13719) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 21.7/33.3 ms; cache_hit_rate: 46%; mean normalized entropy: 0.246 (branching 3.56); mean applied temperature: 2.79; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8s40: distinct-2 = 0.652 (basis 15304), distinct-3 = 0.806 (basis 13805) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.1/34.7 ms; cache_hit_rate: 40%; mean normalized entropy: 0.247 (branching 3.58); mean applied temperature: 2.74; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.195 [0.157, 0.235] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.285 [0.243, 0.331] | 0.003 [0.000, 0.008] | 0.396 [0.369, 0.424] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.395 [0.344, 0.448] | 0.003 [0.000, 0.008] | 0.258 [0.232, 0.284] |
| C8b30 | generic | 375 | 1.000 [1.000, 1.000] | 0.248 [0.208, 0.296] | 0.003 [0.000, 0.008] | 0.349 [0.322, 0.378] |
| C8b30 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.301 [0.253, 0.347] | 0.000 [0.000, 0.000] | 0.428 [0.401, 0.454] |
| C8b30 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.138 [0.109, 0.172] |
| C8b30 | topical | 375 | 1.000 [1.000, 1.000] | 0.491 [0.443, 0.541] | 0.003 [0.000, 0.008] | 0.299 [0.274, 0.326] |
| C8b40 | generic | 375 | 1.000 [1.000, 1.000] | 0.251 [0.205, 0.296] | 0.003 [0.000, 0.008] | 0.360 [0.331, 0.390] |
| C8b40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.307 [0.259, 0.352] | 0.000 [0.000, 0.000] | 0.452 [0.426, 0.477] |
| C8b40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.143 [0.112, 0.178] |
| C8b40 | topical | 375 | 1.000 [1.000, 1.000] | 0.536 [0.488, 0.587] | 0.003 [0.000, 0.008] | 0.316 [0.289, 0.342] |
| C8s40 | generic | 375 | 1.000 [1.000, 1.000] | 0.205 [0.165, 0.243] | 0.000 [0.000, 0.000] | 0.364 [0.336, 0.393] |
| C8s40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.269 [0.224, 0.315] | 0.003 [0.000, 0.008] | 0.413 [0.385, 0.440] |
| C8s40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.150 [0.118, 0.183] |
| C8s40 | topical | 375 | 1.000 [1.000, 1.000] | 0.403 [0.352, 0.453] | 0.000 [0.000, 0.000] | 0.287 [0.262, 0.314] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.640 [0.629, 0.651] | 0.995 [0.991, 0.998] | 0.692 [0.670, 0.715] | 0.239 [0.219, 0.260] | 0.266 [0.240, 0.291] | 23.0 / 30.3 | F4_stale 2 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.104 [0.098, 0.109] | 0.462 [0.437, 0.487] | 0.244 [0.214, 0.276] | 0.201 [0.159, 0.249] | 0.183 [0.130, 0.237] | 20.2 / 25.4 | 0 |
| C0 | extension | 1500 | 0.256 [0.246, 0.266] | 0.751 [0.728, 0.772] | 0.264 [0.237, 0.291] | 0.139 [0.109, 0.170] | 0.077 [0.050, 0.107] | 24.2 / 19.3 | 0 |
| C8b30 | vanilla | 1500 | 0.656 [0.645, 0.667] | 0.994 [0.989, 0.997] | 0.710 [0.687, 0.733] | 0.261 [0.241, 0.280] | 0.320 [0.291, 0.347] | 23.0 / 30.7 | F4_stale 2 |
| C8b30 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8b30 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.456 [0.429, 0.483] | 0.231 [0.200, 0.262] | 0.236 [0.188, 0.286] | 0.209 [0.146, 0.272] | 20.1 / 25.5 | 0 |
| C8b30 | extension | 1500 | 0.243 [0.232, 0.253] | 0.725 [0.701, 0.747] | 0.260 [0.234, 0.287] | 0.147 [0.115, 0.180] | 0.067 [0.039, 0.099] | 24.6 / 19.0 | 0 |
| C8b40 | vanilla | 1500 | 0.662 [0.652, 0.673] | 0.994 [0.990, 0.997] | 0.711 [0.687, 0.734] | 0.280 [0.259, 0.301] | 0.325 [0.294, 0.353] | 22.2 / 29.5 | F4_stale 2 |
| C8b40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8b40 | mutated | 1500 | 0.105 [0.099, 0.111] | 0.469 [0.443, 0.493] | 0.230 [0.199, 0.260] | 0.231 [0.183, 0.278] | 0.247 [0.185, 0.315] | 19.6 / 24.7 | 0 |
| C8b40 | extension | 1500 | 0.233 [0.223, 0.244] | 0.713 [0.688, 0.735] | 0.260 [0.235, 0.285] | 0.160 [0.128, 0.197] | 0.090 [0.058, 0.126] | 23.8 / 18.5 | 0 |
| C8s40 | vanilla | 1500 | 0.652 [0.642, 0.663] | 0.996 [0.992, 0.999] | 0.714 [0.693, 0.738] | 0.273 [0.254, 0.292] | 0.263 [0.236, 0.290] | 23.7 / 29.6 | F4_stale 2 |
| C8s40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8s40 | mutated | 1500 | 0.103 [0.097, 0.109] | 0.467 [0.441, 0.492] | 0.244 [0.213, 0.277] | 0.236 [0.193, 0.282] | 0.158 [0.105, 0.216] | 20.9 / 26.2 | 0 |
| C8s40 | extension | 1500 | 0.245 [0.234, 0.254] | 0.734 [0.711, 0.755] | 0.238 [0.213, 0.262] | 0.148 [0.113, 0.184] | 0.084 [0.053, 0.118] | 25.3 / 19.4 | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8b30]**: insufficient data — coverage: context starts 56.3% vs 50.3% (shift +0.059; floor ±5%); affinity_without_copy Δ 0.013 [0.003, 0.023] *; window_escape Δ -0.046 [-0.087, -0.003] *; pool ECB 4.411 [4.373, 4.446] (floor 4.0); copy Δ 0.041 [0.026, 0.057] *; repetition Δ -0.001 [-0.003, 0.000]; p95 34.6 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — window escape dropped significantly; copy rose significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8b40]**: insufficient data — coverage: context starts 59.9% vs 50.3% (shift +0.095; floor ±5%); affinity_without_copy Δ 0.026 [0.013, 0.039] *; window_escape Δ -0.026 [-0.074, 0.027]; pool ECB 4.369 [4.331, 4.407] (floor 4.0); copy Δ 0.054 [0.036, 0.070] *; repetition Δ -0.001 [-0.003, 0.000]; p95 33.3 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — copy rose significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8s40]**: insufficient data — coverage: context starts 56.0% vs 50.3% (shift +0.057; floor ±5%); affinity_without_copy Δ 0.031 [0.023, 0.039] *; window_escape Δ -0.019 [-0.051, 0.014]; pool ECB 4.437 [4.405, 4.469] (floor 4.0); copy Δ 0.001 [-0.011, 0.011]; repetition Δ -0.002 [-0.005, 0.000]; p95 34.7 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted); gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.081 [2.027, 2.135] (min 2.0); pool ECB 4.430 [4.395, 4.464] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8b30]**: insufficient data — window escape 2.035 [1.984, 2.089] (min 2.0); pool ECB 4.411 [4.373, 4.446] (floor 4.0); pool ECB share 0.882 [0.875, 0.889] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:41%, 2:28%, 3:20%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8b40]**: insufficient data — window escape 2.055 [2.005, 2.113] (min 2.0); pool ECB 4.369 [4.331, 4.407] (floor 4.0); pool ECB share 0.874 [0.866, 0.881] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:39%, 2:29%, 3:21%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8s40]**: insufficient data — window escape 2.063 [2.008, 2.116] (min 2.0); pool ECB 4.437 [4.405, 4.469] (floor 4.0); pool ECB share 0.887 [0.881, 0.894] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:30%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 937 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 33.8 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8b30]**: pass — 16/27 memes reproduced (59%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8b40]**: pass — 16/27 memes reproduced (59%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8s40]**: pass — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
