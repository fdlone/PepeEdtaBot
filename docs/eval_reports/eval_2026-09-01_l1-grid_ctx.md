# Eval report 2026-09-01 snapshot=l1-grid prompts=308b7deaea0f seeds=42,1337,2026 mode=ctx

Revision: `cd6e1c2`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C7a**: 1500 generations
- **C7b**: 1500 generations
- **C7c**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C7a | Δ C7a vs C0 | C7b | Δ C7b vs C0 | C7c | Δ C7c vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.881 [0.874, 0.888] | 0.881 [0.874, 0.888] | 0.000 [0.000, 0.000] | 0.881 [0.874, 0.888] | 0.000 [-0.000, 0.000] | 0.881 [0.873, 0.888] | -0.000 [-0.001, 0.001] |
| mean_response_length | 11.152 [10.888, 11.417] | 11.159 [10.897, 11.425] | 0.007 [-0.005, 0.024] | 11.169 [10.904, 11.437] | 0.017 [-0.016, 0.049] | 11.176 [10.915, 11.447] | 0.024 [-0.014, 0.066] |
| unique_token_ratio | 0.987 [0.985, 0.988] | 0.987 [0.985, 0.988] | 0.000 [-0.000, 0.000] | 0.987 [0.985, 0.988] | -0.000 [-0.000, 0.000] | 0.987 [0.985, 0.988] | -0.000 [-0.000, 0.000] |
| exact_context_copy_rate | 0.219 [0.198, 0.239] | 0.218 [0.197, 0.238] | -0.001 [-0.003, 0.000] | 0.221 [0.199, 0.241] | 0.001 [-0.001, 0.004] | 0.222 [0.201, 0.243] | 0.003 [0.000, 0.006] |
| repetition_rate | 0.003 [0.001, 0.005] | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.279 [0.264, 0.295] | 0.280 [0.264, 0.295] | 0.000 [-0.001, 0.001] | 0.280 [0.264, 0.295] | 0.000 [-0.001, 0.001] | 0.280 [0.264, 0.294] | 0.000 [-0.001, 0.002] |
| context_affinity_without_copy | 0.211 [0.196, 0.228] | 0.212 [0.197, 0.228] | 0.000 [0.000, 0.000] | 0.211 [0.197, 0.227] | -0.000 [-0.001, 0.000] | 0.211 [0.195, 0.227] | -0.001 [-0.002, 0.000] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.309 [0.264, 0.355] | 0.309 [0.264, 0.355] | 0.000 [0.000, 0.000] | 0.309 [0.264, 0.355] | 0.000 [0.000, 0.000] | 0.304 [0.259, 0.352] | -0.005 [-0.013, 0.000] |
| structural_pool_ecb | 4.430 [4.395, 4.464] | 4.430 [4.395, 4.464] | 0.000 [0.000, 0.000] | 4.431 [4.397, 4.465] | 0.001 [-0.002, 0.005] | 4.431 [4.397, 4.465] | 0.001 [-0.003, 0.005] |
| structural_window_escape | 2.081 [2.027, 2.135] | 2.080 [2.025, 2.133] | -0.001 [-0.005, 0.001] | 2.080 [2.024, 2.133] | -0.001 [-0.006, 0.004] | 2.085 [2.029, 2.139] | 0.004 [-0.003, 0.011] |

C0: distinct-2 = 0.652 (basis 15228), distinct-3 = 0.804 (basis 13730) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.5/35.8 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C7a: distinct-2 = 0.652 (basis 15239), distinct-3 = 0.804 (basis 13741) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 25.3/38.2 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C7b: distinct-2 = 0.651 (basis 15253), distinct-3 = 0.803 (basis 13755) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 25.2/38.0 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C7c: distinct-2 = 0.651 (basis 15264), distinct-3 = 0.803 (basis 13766) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.2/35.7 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.195 [0.157, 0.235] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.285 [0.243, 0.331] | 0.003 [0.000, 0.008] | 0.396 [0.369, 0.424] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.395 [0.344, 0.448] | 0.003 [0.000, 0.008] | 0.258 [0.232, 0.284] |
| C7a | generic | 375 | 1.000 [1.000, 1.000] | 0.195 [0.157, 0.235] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C7a | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.280 [0.240, 0.325] | 0.003 [0.000, 0.008] | 0.397 [0.370, 0.424] |
| C7a | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C7a | topical | 375 | 1.000 [1.000, 1.000] | 0.395 [0.344, 0.448] | 0.003 [0.000, 0.008] | 0.258 [0.232, 0.284] |
| C7b | generic | 375 | 1.000 [1.000, 1.000] | 0.197 [0.157, 0.237] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C7b | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.283 [0.243, 0.328] | 0.003 [0.000, 0.008] | 0.396 [0.369, 0.422] |
| C7b | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C7b | topical | 375 | 1.000 [1.000, 1.000] | 0.400 [0.347, 0.448] | 0.003 [0.000, 0.008] | 0.259 [0.233, 0.285] |
| C7c | generic | 375 | 1.000 [1.000, 1.000] | 0.200 [0.160, 0.240] | 0.005 [0.000, 0.013] | 0.310 [0.284, 0.339] |
| C7c | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.283 [0.240, 0.328] | 0.003 [0.000, 0.008] | 0.394 [0.366, 0.422] |
| C7c | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C7c | topical | 375 | 1.000 [1.000, 1.000] | 0.403 [0.352, 0.453] | 0.003 [0.000, 0.008] | 0.260 [0.234, 0.286] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.640 [0.629, 0.651] | 0.995 [0.991, 0.998] | 0.692 [0.670, 0.715] | 0.239 [0.219, 0.260] | 0.266 [0.240, 0.291] | 24.3 / 32.5 | F4_stale 2 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.104 [0.098, 0.109] | 0.462 [0.437, 0.487] | 0.244 [0.214, 0.276] | 0.201 [0.159, 0.249] | 0.183 [0.130, 0.237] | 21.4 / 26.9 | 0 |
| C0 | extension | 1500 | 0.256 [0.246, 0.266] | 0.751 [0.728, 0.772] | 0.264 [0.237, 0.291] | 0.139 [0.109, 0.170] | 0.077 [0.050, 0.107] | 25.6 / 20.5 | 0 |
| C7a | vanilla | 1500 | 0.640 [0.630, 0.651] | 0.995 [0.991, 0.998] | 0.690 [0.667, 0.712] | 0.239 [0.219, 0.259] | 0.264 [0.238, 0.290] | 25.9 / 34.8 | F4_stale 2 |
| C7a | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7a | mutated | 1500 | 0.104 [0.098, 0.109] | 0.462 [0.437, 0.487] | 0.247 [0.216, 0.278] | 0.200 [0.158, 0.249] | 0.187 [0.129, 0.246] | 22.7 / 28.7 | 0 |
| C7a | extension | 1500 | 0.256 [0.246, 0.266] | 0.751 [0.728, 0.771] | 0.266 [0.241, 0.291] | 0.140 [0.110, 0.172] | 0.077 [0.047, 0.107] | 27.4 / 21.6 | 0 |
| C7b | vanilla | 1500 | 0.641 [0.631, 0.652] | 0.995 [0.991, 0.998] | 0.689 [0.666, 0.712] | 0.239 [0.220, 0.259] | 0.268 [0.240, 0.296] | 25.9 / 33.7 | F4_stale 2 |
| C7b | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7b | mutated | 1500 | 0.103 [0.098, 0.109] | 0.462 [0.437, 0.487] | 0.250 [0.219, 0.283] | 0.198 [0.156, 0.245] | 0.185 [0.127, 0.249] | 22.7 / 28.7 | 0 |
| C7b | extension | 1500 | 0.255 [0.245, 0.266] | 0.749 [0.725, 0.769] | 0.266 [0.240, 0.292] | 0.140 [0.109, 0.173] | 0.080 [0.050, 0.110] | 27.3 / 21.7 | 0 |
| C7c | vanilla | 1500 | 0.640 [0.630, 0.651] | 0.995 [0.991, 0.998] | 0.691 [0.668, 0.714] | 0.238 [0.218, 0.257] | 0.270 [0.244, 0.297] | 23.8 / 31.6 | F4_stale 2 |
| C7c | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7c | mutated | 1500 | 0.103 [0.098, 0.109] | 0.462 [0.437, 0.487] | 0.247 [0.216, 0.278] | 0.200 [0.152, 0.245] | 0.170 [0.117, 0.228] | 20.8 / 26.4 | 0 |
| C7c | extension | 1500 | 0.256 [0.246, 0.267] | 0.751 [0.729, 0.773] | 0.264 [0.240, 0.289] | 0.140 [0.112, 0.172] | 0.087 [0.057, 0.121] | 25.2 / 19.8 | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel[C7a]**: insufficient data — ctx: seed draw not modelled (the pipeline never seeds addressed replies); this mode measures the mutation-protection half only; affinity_without_copy Δ 0.000 [0.000, 0.000]; copy Δ -0.001 [-0.003, 0.000]; repetition Δ 0.000 [0.000, 0.000]; p95 38.2 ms (budget 150); gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel[C7b]**: insufficient data — ctx: seed draw not modelled (the pipeline never seeds addressed replies); this mode measures the mutation-protection half only; affinity_without_copy Δ -0.000 [-0.001, 0.000]; copy Δ 0.001 [-0.001, 0.004]; repetition Δ 0.000 [0.000, 0.000]; p95 38.0 ms (budget 150); gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel[C7c]**: insufficient data — ctx: seed draw not modelled (the pipeline never seeds addressed replies); this mode measures the mutation-protection half only; affinity_without_copy Δ -0.001 [-0.002, 0.000]; copy Δ 0.003 [0.000, 0.006]; repetition Δ 0.000 [0.000, 0.000]; p95 35.7 ms (budget 150); gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.081 [2.027, 2.135] (min 2.0); pool ECB 4.430 [4.395, 4.464] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C7a]**: insufficient data — window escape 2.080 [2.025, 2.133] (min 2.0); pool ECB 4.430 [4.395, 4.464] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:29%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C7b]**: insufficient data — window escape 2.080 [2.024, 2.133] (min 2.0); pool ECB 4.431 [4.397, 4.465] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C7c]**: insufficient data — window escape 2.085 [2.029, 2.139] (min 2.0); pool ECB 4.431 [4.397, 4.465] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:23%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 937 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 35.8 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7a]**: pass — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7b]**: pass — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7c]**: pass — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
