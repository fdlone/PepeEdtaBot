# Eval report 2026-09-11 snapshot=promotion-check prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `3d04dce`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **CP**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | CP | Δ CP vs C0 |
|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.869 [0.862, 0.876] | 0.888 [0.881, 0.895] | 0.019 [0.012, 0.026] * |
| mean_response_length | 11.063 [10.786, 11.331] | 11.177 [10.917, 11.436] | 0.113 [-0.186, 0.409] |
| unique_token_ratio | 0.985 [0.983, 0.986] | 0.984 [0.982, 0.985] | -0.001 [-0.004, 0.001] |
| exact_context_copy_rate | 0.211 [0.191, 0.233] | 0.162 [0.143, 0.181] | -0.049 [-0.071, -0.028] * |
| repetition_rate | 0.002 [0.000, 0.005] | 0.001 [0.000, 0.002] | -0.001 [-0.004, 0.001] |
| cycle_detection_rate | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.310 [0.294, 0.326] | 0.322 [0.305, 0.340] | 0.012 [-0.005, 0.032] |
| context_affinity_without_copy | 0.250 [0.234, 0.269] | 0.274 [0.257, 0.292] | 0.040 [0.021, 0.058] * |
| seeded_present_rate | insufficient data | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — |
| historical_meme_rate | 0.267 [0.224, 0.309] | 0.264 [0.221, 0.309] | -0.003 [-0.056, 0.051] |
| structural_pool_ecb | 4.479 [4.443, 4.510] | 4.645 [4.616, 4.672] | 0.166 [0.131, 0.201] * |
| structural_window_escape | 2.084 [2.029, 2.134] | 2.530 [2.468, 2.590] | 0.446 [0.377, 0.515] * |

C0: distinct-2 = 0.656 (basis 15095), distinct-3 = 0.816 (basis 13597) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.3/36.3 ms; cache_hit_rate: 41%; mean normalized entropy: 0.248 (branching 3.52); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

CP: distinct-2 = 0.633 (basis 15265), distinct-3 = 0.788 (basis 13765) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.9/36.1 ms; cache_hit_rate: 41%; mean normalized entropy: 0.244 (branching 3.49); mean applied temperature: 2.69; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 13 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.208 [0.171, 0.248] | 0.003 [0.000, 0.008] | 0.344 [0.315, 0.374] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.309 [0.261, 0.355] | 0.003 [0.000, 0.008] | 0.374 [0.345, 0.404] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.261 [0.221, 0.304] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.323 [0.275, 0.368] | 0.003 [0.000, 0.008] | 0.256 [0.228, 0.283] |
| CP | generic | 375 | 1.000 [1.000, 1.000] | 0.168 [0.131, 0.205] | 0.000 [0.000, 0.000] | 0.325 [0.299, 0.352] |
| CP | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.227 [0.189, 0.267] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.378] |
| CP | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.008] | 0.383 [0.340, 0.424] |
| CP | topical | 375 | 1.000 [1.000, 1.000] | 0.251 [0.205, 0.293] | 0.000 [0.000, 0.000] | 0.238 [0.213, 0.263] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.994 [0.990, 0.997] | 0.736 [0.713, 0.757] | 0.287 [0.266, 0.307] | 0.255 [0.230, 0.281] | 24.2 / 27.7 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.099 [0.093, 0.106] | 0.435 [0.412, 0.463] | 0.207 [0.178, 0.237] | 0.203 [0.143, 0.265] | 0.222 [0.148, 0.296] | 21.5 / 26.3 | 0 |
| C0 | extension | 1500 | 0.250 [0.240, 0.259] | 0.747 [0.724, 0.769] | 0.239 [0.215, 0.265] | 0.154 [0.118, 0.191] | 0.022 [0.007, 0.045] | 25.6 / 20.1 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| CP | vanilla | 1500 | 0.470 [0.459, 0.480] | 0.971 [0.961, 0.979] | 0.532 [0.505, 0.556] | 0.281 [0.254, 0.309] | 0.281 [0.249, 0.311] | 24.5 / 27.0 | 0 |
| CP | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| CP | mutated | 1500 | 0.061 [0.056, 0.066] | 0.288 [0.265, 0.313] | 0.201 [0.164, 0.241] | 0.208 [0.146, 0.275] | 0.149 [0.080, 0.230] | 21.1 / 26.0 | 0 |
| CP | extension | 1500 | 0.177 [0.168, 0.187] | 0.615 [0.591, 0.640] | 0.218 [0.192, 0.243] | 0.122 [0.089, 0.158] | 0.025 [0.005, 0.045] | 25.8 / 22.6 | 0 |
| CP | hot | 13 | 0.003 [0.002, 0.006] | 0.009 [0.005, 0.014] | 0.462 [0.231, 0.769] | insufficient data | 0.000 [0.000, 0.000] | 16.3 / 24.7 | 0 |
| CP | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| CP | phrase | 1500 | 0.289 [0.280, 0.299] | 0.758 [0.735, 0.783] | 0.379 [0.351, 0.406] | 0.344 [0.317, 0.372] | 0.016 [0.005, 0.030] | 24.1 / 26.1 | F4_stale 4 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate**: insufficient data — no route-gate arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.084 [2.029, 2.134] (min 2.0); pool ECB 4.479 [4.443, 4.510] (floor 4.0); pool ECB share 0.896 [0.889, 0.902] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:39%, 2:28%, 3:22%, 4:10%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[CP]**: insufficient data — window escape 2.530 [2.468, 2.590] (min 2.0); pool ECB 4.645 [4.616, 4.672] (floor 4.0); pool ECB share 0.929 [0.923, 0.934] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:23%, 2:28%, 3:28%, 4:17%, 5:5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.002] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 860 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 36.3 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 15/18 memes reproduced (83%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[CP]**: pass — 17/18 memes reproduced (94%); C0 15/18 (83%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
