# Eval report 2026-09-02 snapshot=pool-grid1 prompts=308b7deaea0f seeds=42,1337,2026 mode=noctx

Revision: `25da193`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
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
| candidate_accept_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.352 [10.138, 10.603] | 10.352 [10.138, 10.603] | 0.000 [0.000, 0.000] | 10.352 [10.138, 10.603] | 0.000 [0.000, 0.000] | 10.352 [10.138, 10.603] | 0.000 [0.000, 0.000] |
| unique_token_ratio | 0.988 [0.986, 0.989] | 0.988 [0.986, 0.989] | 0.000 [0.000, 0.000] | 0.988 [0.986, 0.989] | 0.000 [0.000, 0.000] | 0.988 [0.986, 0.989] | 0.000 [0.000, 0.000] |
| exact_context_copy_rate | 0.004 [0.001, 0.007] | 0.004 [0.001, 0.007] | 0.000 [0.000, 0.000] | 0.004 [0.001, 0.007] | 0.000 [0.000, 0.000] | 0.004 [0.001, 0.007] | 0.000 [0.000, 0.000] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.023 [0.019, 0.027] | 0.023 [0.019, 0.027] | 0.000 [0.000, 0.000] | 0.023 [0.019, 0.027] | 0.000 [0.000, 0.000] | 0.023 [0.019, 0.027] | 0.000 [0.000, 0.000] |
| context_affinity_without_copy | 0.023 [0.019, 0.027] | 0.023 [0.019, 0.027] | 0.000 [0.000, 0.000] | 0.023 [0.019, 0.027] | 0.000 [0.000, 0.000] | 0.023 [0.019, 0.027] | 0.000 [0.000, 0.000] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.019 [0.008, 0.035] | 0.019 [0.008, 0.035] | 0.000 [0.000, 0.000] | 0.019 [0.008, 0.035] | 0.000 [0.000, 0.000] | 0.019 [0.008, 0.035] | 0.000 [0.000, 0.000] |
| structural_pool_ecb | 4.536 [4.507, 4.565] | 4.536 [4.507, 4.565] | 0.000 [0.000, 0.000] | 4.536 [4.507, 4.565] | 0.000 [0.000, 0.000] | 4.536 [4.507, 4.565] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.863 [2.813, 2.914] | 2.863 [2.813, 2.914] | 0.000 [0.000, 0.000] | 2.863 [2.813, 2.914] | 0.000 [0.000, 0.000] | 2.863 [2.813, 2.914] | 0.000 [0.000, 0.000] |

C0: distinct-2 = 0.703 (basis 14028), distinct-3 = 0.857 (basis 12528) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 13.5/21.7 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.52); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C8b30: distinct-2 = 0.703 (basis 14028), distinct-3 = 0.857 (basis 12528) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 13.8/22.0 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.52); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C8b40: distinct-2 = 0.703 (basis 14028), distinct-3 = 0.857 (basis 12528) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.1/22.6 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.52); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C8s40: distinct-2 = 0.703 (basis 14028), distinct-3 = 0.857 (basis 12528) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 13.9/22.3 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.52); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.031] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.017 [0.013, 0.022] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.040] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.031] |
| C8b30 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.031] |
| C8b30 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.017 [0.013, 0.022] |
| C8b30 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.040] |
| C8b30 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.031] |
| C8b40 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.031] |
| C8b40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.017 [0.013, 0.022] |
| C8b40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.040] |
| C8b40 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.031] |
| C8s40 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.024 [0.018, 0.031] |
| C8s40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.017 [0.013, 0.022] |
| C8s40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.025 [0.013, 0.040] |
| C8s40 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.031] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.590 [0.579, 0.602] | 0.989 [0.984, 0.994] | 0.559 [0.532, 0.582] | 0.022 [0.017, 0.028] | 0.007 [0.002, 0.014] | 14.1 / 14.5 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.100 [0.094, 0.106] | 0.447 [0.423, 0.472] | 0.273 [0.242, 0.307] | 0.028 [0.019, 0.038] | 0.000 [0.000, 0.000] | 12.5 / 15.3 | 0 |
| C0 | extension | 1500 | 0.310 [0.299, 0.321] | 0.843 [0.823, 0.863] | 0.386 [0.358, 0.413] | 0.022 [0.015, 0.030] | 0.000 [0.000, 0.000] | 14.3 / 12.7 | 0 |
| C8b30 | vanilla | 1500 | 0.590 [0.579, 0.602] | 0.989 [0.984, 0.994] | 0.559 [0.532, 0.582] | 0.022 [0.017, 0.028] | 0.007 [0.002, 0.014] | 14.3 / 14.9 | 0 |
| C8b30 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8b30 | mutated | 1500 | 0.100 [0.094, 0.106] | 0.447 [0.423, 0.472] | 0.273 [0.242, 0.307] | 0.028 [0.019, 0.038] | 0.000 [0.000, 0.000] | 12.5 / 15.7 | 0 |
| C8b30 | extension | 1500 | 0.310 [0.299, 0.321] | 0.843 [0.823, 0.863] | 0.386 [0.358, 0.413] | 0.022 [0.015, 0.030] | 0.000 [0.000, 0.000] | 14.5 / 12.9 | 0 |
| C8b40 | vanilla | 1500 | 0.590 [0.579, 0.602] | 0.989 [0.984, 0.994] | 0.559 [0.532, 0.582] | 0.022 [0.017, 0.028] | 0.007 [0.002, 0.014] | 14.6 / 14.8 | 0 |
| C8b40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8b40 | mutated | 1500 | 0.100 [0.094, 0.106] | 0.447 [0.423, 0.472] | 0.273 [0.242, 0.307] | 0.028 [0.019, 0.038] | 0.000 [0.000, 0.000] | 12.9 / 16.0 | 0 |
| C8b40 | extension | 1500 | 0.310 [0.299, 0.321] | 0.843 [0.823, 0.863] | 0.386 [0.358, 0.413] | 0.022 [0.015, 0.030] | 0.000 [0.000, 0.000] | 14.9 / 13.3 | 0 |
| C8s40 | vanilla | 1500 | 0.590 [0.579, 0.602] | 0.989 [0.984, 0.994] | 0.559 [0.532, 0.582] | 0.022 [0.017, 0.028] | 0.007 [0.002, 0.014] | 14.4 / 14.8 | 0 |
| C8s40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C8s40 | mutated | 1500 | 0.100 [0.094, 0.106] | 0.447 [0.423, 0.472] | 0.273 [0.242, 0.307] | 0.028 [0.019, 0.038] | 0.000 [0.000, 0.000] | 12.7 / 15.8 | 0 |
| C8s40 | extension | 1500 | 0.310 [0.299, 0.321] | 0.843 [0.823, 0.863] | 0.386 [0.358, 0.413] | 0.022 [0.015, 0.030] | 0.000 [0.000, 0.000] | 14.6 / 13.0 | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition[C8b30]**: insufficient data — noctx: the context knobs are inert without context; this mode checks that nothing moved; copy Δ 0.000 [0.000, 0.000]; repetition Δ 0.000 [0.000, 0.000]; p95 22.0 ms (budget 150); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition[C8b40]**: insufficient data — noctx: the context knobs are inert without context; this mode checks that nothing moved; copy Δ 0.000 [0.000, 0.000]; repetition Δ 0.000 [0.000, 0.000]; p95 22.6 ms (budget 150); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition[C8s40]**: insufficient data — noctx: the context knobs are inert without context; this mode checks that nothing moved; copy Δ 0.000 [0.000, 0.000]; repetition Δ 0.000 [0.000, 0.000]; p95 22.3 ms (budget 150); gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.863 [2.813, 2.914] (min 2.0); pool ECB 4.536 [4.507, 4.565] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C8b30]**: insufficient data — window escape 2.863 [2.813, 2.914] (min 2.0); pool ECB 4.536 [4.507, 4.565] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C8b40]**: insufficient data — window escape 2.863 [2.813, 2.914] (min 2.0); pool ECB 4.536 [4.507, 4.565] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C8s40]**: insufficient data — window escape 2.863 [2.813, 2.914] (min 2.0); pool ECB 4.536 [4.507, 4.565] (floor 4.0); pool ECB share 0.907 [0.901, 0.913] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.003] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1520 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 21.7 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8b30]**: pass — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8b40]**: pass — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C8s40]**: pass — 6/27 memes reproduced (22%); C0 6/27 (22%), tolerance 10%; prompt set 308b7deaea0f

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
