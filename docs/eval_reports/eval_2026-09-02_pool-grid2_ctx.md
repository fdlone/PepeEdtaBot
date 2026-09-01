# Eval report 2026-09-02 snapshot=pool-grid2 prompts=308b7deaea0f seeds=42,1337,2026 mode=ctx

Revision: `25da193`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C8a10**: 1500 generations
- **C8m**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C8a10 | Δ C8a10 vs C0 | C8m | Δ C8m vs C0 |
|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.881 [0.874, 0.888] | 0.856 [0.847, 0.865] | -0.024 [-0.027, -0.021] * | 0.864 [0.856, 0.873] | -0.017 [-0.023, -0.010] * |
| mean_response_length | 11.152 [10.888, 11.417] | 11.176 [10.911, 11.455] | 0.024 [-0.117, 0.165] | 11.215 [10.970, 11.483] | 0.063 [-0.175, 0.307] |
| unique_token_ratio | 0.987 [0.985, 0.988] | 0.987 [0.985, 0.988] | -0.000 [-0.001, 0.001] | 0.987 [0.985, 0.989] | 0.000 [-0.002, 0.003] |
| exact_context_copy_rate | 0.219 [0.198, 0.239] | 0.244 [0.224, 0.265] | 0.025 [0.017, 0.033] * | 0.278 [0.256, 0.301] | 0.059 [0.041, 0.077] * |
| repetition_rate | 0.003 [0.001, 0.005] | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.005, 0.001] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.279 [0.264, 0.295] | 0.313 [0.297, 0.328] | 0.034 [0.027, 0.041] * | 0.365 [0.349, 0.380] | 0.086 [0.073, 0.099] * |
| context_affinity_without_copy | 0.211 [0.196, 0.228] | 0.243 [0.225, 0.259] | 0.028 [0.021, 0.036] * | 0.293 [0.274, 0.310] | 0.075 [0.061, 0.088] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.309 [0.264, 0.355] | 0.317 [0.269, 0.365] | 0.008 [-0.011, 0.027] | 0.341 [0.293, 0.389] | 0.032 [0.000, 0.064] |
| structural_pool_ecb | 4.430 [4.395, 4.464] | 4.375 [4.339, 4.410] | -0.055 [-0.072, -0.037] * | 4.363 [4.324, 4.400] | -0.067 [-0.100, -0.031] * |
| structural_window_escape | 2.081 [2.027, 2.135] | 2.027 [1.975, 2.081] | -0.055 [-0.087, -0.023] * | 2.031 [1.978, 2.086] | -0.050 [-0.105, 0.006] |

C0: distinct-2 = 0.652 (basis 15228), distinct-3 = 0.804 (basis 13730) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 22.7/34.6 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8a10: distinct-2 = 0.642 (basis 15264), distinct-3 = 0.797 (basis 13766) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 21.9/33.6 ms; cache_hit_rate: 43%; mean normalized entropy: 0.246 (branching 3.53); mean applied temperature: 2.78; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C8m: distinct-2 = 0.636 (basis 15322), distinct-3 = 0.792 (basis 13822) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 21.1/32.8 ms; cache_hit_rate: 46%; mean normalized entropy: 0.245 (branching 3.53); mean applied temperature: 2.79; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.195 [0.157, 0.235] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.285 [0.243, 0.331] | 0.003 [0.000, 0.008] | 0.396 [0.369, 0.424] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.395 [0.344, 0.448] | 0.003 [0.000, 0.008] | 0.258 [0.232, 0.284] |
| C8a10 | generic | 375 | 1.000 [1.000, 1.000] | 0.232 [0.192, 0.277] | 0.005 [0.000, 0.013] | 0.363 [0.336, 0.393] |
| C8a10 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.293 [0.251, 0.339] | 0.003 [0.000, 0.008] | 0.428 [0.402, 0.454] |
| C8a10 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.145 [0.113, 0.177] |
| C8a10 | topical | 375 | 1.000 [1.000, 1.000] | 0.448 [0.397, 0.501] | 0.003 [0.000, 0.008] | 0.299 [0.273, 0.325] |
| C8m | generic | 375 | 1.000 [1.000, 1.000] | 0.280 [0.235, 0.325] | 0.000 [0.000, 0.000] | 0.444 [0.418, 0.472] |
| C8m | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.312 [0.267, 0.360] | 0.003 [0.000, 0.008] | 0.472 [0.447, 0.497] |
| C8m | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.158 [0.128, 0.191] |
| C8m | topical | 375 | 1.000 [1.000, 1.000] | 0.517 [0.469, 0.565] | 0.003 [0.000, 0.008] | 0.365 [0.340, 0.392] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.640 [0.629, 0.651] | 0.995 [0.991, 0.998] | 0.692 [0.670, 0.715] | 0.239 [0.219, 0.260] | 0.266 [0.240, 0.291] | 23.4 / 30.9 | F4_stale 2 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.104 [0.098, 0.109] | 0.462 [0.437, 0.487] | 0.244 [0.214, 0.276] | 0.201 [0.159, 0.249] | 0.183 [0.130, 0.237] | 20.5 / 25.9 | 0 |
| C0 | extension | 1500 | 0.256 [0.246, 0.266] | 0.751 [0.728, 0.772] | 0.264 [0.237, 0.291] | 0.139 [0.109, 0.170] | 0.077 [0.050, 0.107] | 24.7 / 19.6 | 0 |
| C8a10 | vanilla | 1500 | 0.651 [0.640, 0.662] | 0.994 [0.989, 0.997] | 0.714 [0.691, 0.738] | 0.281 [0.259, 0.300] | 0.288 [0.261, 0.316] | 22.6 / 29.6 | F4_stale 2 |
| C8a10 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8a10 | mutated | 1500 | 0.103 [0.097, 0.109] | 0.457 [0.432, 0.482] | 0.234 [0.200, 0.267] | 0.206 [0.157, 0.258] | 0.225 [0.163, 0.294] | 19.7 / 25.1 | 0 |
| C8a10 | extension | 1500 | 0.246 [0.236, 0.257] | 0.734 [0.711, 0.756] | 0.250 [0.222, 0.275] | 0.145 [0.112, 0.182] | 0.084 [0.055, 0.116] | 24.0 / 18.9 | 0 |
| C8m | vanilla | 1500 | 0.686 [0.676, 0.697] | 0.996 [0.993, 0.999] | 0.748 [0.729, 0.769] | 0.331 [0.311, 0.350] | 0.321 [0.294, 0.347] | 21.8 / 28.2 | 0 |
| C8m | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8m | mutated | 1500 | 0.100 [0.094, 0.106] | 0.451 [0.425, 0.475] | 0.237 [0.204, 0.271] | 0.268 [0.215, 0.323] | 0.244 [0.175, 0.312] | 19.2 / 24.0 | 0 |
| C8m | extension | 1500 | 0.214 [0.204, 0.223] | 0.671 [0.647, 0.694] | 0.220 [0.196, 0.246] | 0.158 [0.119, 0.199] | 0.086 [0.050, 0.122] | 23.5 / 18.3 | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8a10]**: insufficient data — coverage: context starts 55.9% vs 50.3% (shift +0.055; floor ±5%); affinity_without_copy Δ 0.028 [0.021, 0.036] *; window_escape Δ -0.055 [-0.087, -0.023] *; pool ECB 4.375 [4.339, 4.410] (floor 4.0); copy Δ 0.025 [0.017, 0.033] *; repetition Δ 0.000 [0.000, 0.000]; p95 33.6 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — window escape dropped significantly; copy rose significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition[C8m]**: insufficient data — coverage: context starts 66.7% vs 50.3% (shift +0.164; floor ±5%); affinity_without_copy Δ 0.075 [0.061, 0.088] *; window_escape Δ -0.050 [-0.105, 0.006]; pool ECB 4.363 [4.324, 4.400] (floor 4.0); copy Δ 0.059 [0.041, 0.077] *; repetition Δ -0.001 [-0.005, 0.001]; p95 32.8 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — copy rose significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.081 [2.027, 2.135] (min 2.0); pool ECB 4.430 [4.395, 4.464] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8a10]**: insufficient data — window escape 2.027 [1.975, 2.081] (min 2.0); pool ECB 4.375 [4.339, 4.410] (floor 4.0); pool ECB share 0.885 [0.878, 0.891] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:39%, 2:31%, 3:21%, 4:7%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C8m]**: insufficient data — window escape 2.031 [1.978, 2.086] (min 2.0); pool ECB 4.363 [4.324, 4.400] (floor 4.0); pool ECB share 0.878 [0.871, 0.885] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:31%, 3:21%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 937 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 34.6 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8a10]**: pass — 15/27 memes reproduced (56%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8m]**: pass — 15/27 memes reproduced (56%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
