# Eval report 2026-09-02 snapshot=l1-route-v2 prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `1d36e26`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C7r20**: 1500 generations
- **C7r40**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C7r20 | Δ C7r20 vs C0 | C7r40 | Δ C7r40 vs C0 |
|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.870 [0.862, 0.877] | 0.869 [0.862, 0.876] | -0.001 [-0.001, -0.000] * | 0.869 [0.862, 0.876] | -0.001 [-0.001, -0.000] * |
| mean_response_length | 10.917 [10.691, 11.185] | 10.927 [10.691, 11.195] | 0.010 [-0.026, 0.049] | 10.931 [10.699, 11.193] | 0.015 [-0.021, 0.053] |
| unique_token_ratio | 0.984 [0.981, 0.986] | 0.984 [0.982, 0.986] | 0.000 [-0.000, 0.001] | 0.984 [0.981, 0.986] | 0.000 [-0.000, 0.001] |
| exact_context_copy_rate | 0.226 [0.205, 0.247] | 0.227 [0.206, 0.249] | 0.001 [0.000, 0.003] | 0.227 [0.206, 0.249] | 0.001 [0.000, 0.003] |
| repetition_rate | 0.002 [0.000, 0.005] | 0.002 [0.000, 0.005] | 0.000 [0.000, 0.000] | 0.002 [0.000, 0.005] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.321 [0.305, 0.336] | 0.321 [0.306, 0.336] | 0.001 [-0.000, 0.002] | 0.321 [0.306, 0.336] | 0.001 [-0.000, 0.002] |
| context_affinity_without_copy | 0.253 [0.235, 0.273] | 0.253 [0.236, 0.271] | 0.000 [-0.001, 0.002] | 0.253 [0.236, 0.271] | 0.000 [-0.001, 0.002] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.285 [0.243, 0.333] | 0.285 [0.243, 0.333] | 0.000 [0.000, 0.000] | 0.285 [0.243, 0.333] | 0.000 [0.000, 0.000] |
| structural_pool_ecb | 4.485 [4.452, 4.515] | 4.487 [4.456, 4.517] | 0.002 [-0.003, 0.007] | 4.483 [4.453, 4.513] | -0.001 [-0.006, 0.003] |
| structural_window_escape | 2.038 [1.985, 2.091] | 2.039 [1.986, 2.091] | 0.001 [-0.009, 0.011] | 2.038 [1.983, 2.091] | 0.000 [-0.008, 0.008] |

C0: distinct-2 = 0.661 (basis 14875), distinct-3 = 0.818 (basis 13376) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 23.2/38.4 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C7r20: distinct-2 = 0.660 (basis 14890), distinct-3 = 0.817 (basis 13391) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 28.4/49.4 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.47); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 13 draws, empty 0%; storage_delta: n/a.

C7r40: distinct-2 = 0.659 (basis 14897), distinct-3 = 0.816 (basis 13398) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 26.3/41.7 ms; cache_hit_rate: 41%; mean normalized entropy: 0.243 (branching 3.47); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 13 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.232 [0.189, 0.275] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.382] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.360] | 0.008 [0.000, 0.019] | 0.364 [0.337, 0.392] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.266 [0.225, 0.306] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.355 [0.307, 0.403] | 0.000 [0.000, 0.000] | 0.297 [0.269, 0.324] |
| C7r20 | generic | 375 | 1.000 [1.000, 1.000] | 0.237 [0.195, 0.283] | 0.000 [0.000, 0.000] | 0.353 [0.324, 0.384] |
| C7r20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.360] | 0.008 [0.000, 0.019] | 0.363 [0.337, 0.392] |
| C7r20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.266 [0.225, 0.306] |
| C7r20 | topical | 375 | 1.000 [1.000, 1.000] | 0.355 [0.307, 0.403] | 0.000 [0.000, 0.000] | 0.297 [0.268, 0.324] |
| C7r40 | generic | 375 | 1.000 [1.000, 1.000] | 0.237 [0.195, 0.283] | 0.000 [0.000, 0.000] | 0.353 [0.324, 0.384] |
| C7r40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.360] | 0.008 [0.000, 0.019] | 0.363 [0.337, 0.392] |
| C7r40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.266 [0.225, 0.306] |
| C7r40 | topical | 375 | 1.000 [1.000, 1.000] | 0.355 [0.307, 0.403] | 0.000 [0.000, 0.000] | 0.297 [0.268, 0.324] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.711 [0.687, 0.734] | 0.300 [0.277, 0.323] | 0.280 [0.256, 0.307] | 24.3 / 44.9 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.227 [0.195, 0.261] | 0.190 [0.140, 0.244] | 0.219 [0.152, 0.285] | 22.4 / 26.1 | 0 |
| C0 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.263 [0.239, 0.290] | 0.151 [0.120, 0.184] | 0.034 [0.017, 0.055] | 26.1 / 19.8 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.990 [0.985, 0.995] | 0.712 [0.690, 0.735] | 0.301 [0.278, 0.322] | 0.283 [0.256, 0.308] | 30.1 / 51.0 | 0 |
| C7r20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | mutated | 1500 | 0.096 [0.091, 0.102] | 0.440 [0.417, 0.465] | 0.223 [0.189, 0.256] | 0.187 [0.136, 0.237] | 0.218 [0.156, 0.286] | 27.5 / 32.5 | 0 |
| C7r20 | extension | 1500 | 0.251 [0.241, 0.262] | 0.741 [0.720, 0.763] | 0.263 [0.237, 0.290] | 0.152 [0.121, 0.186] | 0.034 [0.017, 0.055] | 32.2 / 24.7 | 0 |
| C7r20 | hot | 13 | 0.002 [0.001, 0.003] | 0.009 [0.005, 0.014] | 0.231 [0.000, 0.462] | insufficient data | 0.000 [0.000, 0.000] | 19.0 / 30.4 | 0 |
| C7r40 | vanilla | 1500 | 0.650 [0.638, 0.661] | 0.990 [0.985, 0.995] | 0.712 [0.688, 0.735] | 0.301 [0.278, 0.322] | 0.283 [0.258, 0.310] | 27.1 / 47.8 | 0 |
| C7r40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r40 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.442 [0.419, 0.467] | 0.225 [0.195, 0.258] | 0.187 [0.136, 0.237] | 0.215 [0.154, 0.282] | 24.8 / 29.4 | 0 |
| C7r40 | extension | 1500 | 0.250 [0.239, 0.261] | 0.739 [0.718, 0.761] | 0.259 [0.234, 0.285] | 0.152 [0.121, 0.186] | 0.035 [0.014, 0.056] | 29.2 / 22.1 | 0 |
| C7r40 | hot | 13 | 0.003 [0.002, 0.006] | 0.009 [0.005, 0.014] | 0.538 [0.231, 0.769] | insufficient data | 0.000 [0.000, 0.000] | 17.9 / 27.4 | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel[C7r20]**: insufficient data — ctx: seed draw not modelled (the pipeline never seeds addressed replies); this mode measures the mutation-protection half only; affinity_without_copy Δ 0.000 [-0.001, 0.002]; copy Δ 0.001 [0.000, 0.003]; repetition Δ 0.000 [0.000, 0.000]; p95 49.4 ms (budget 150); gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel[C7r40]**: insufficient data — ctx: seed draw not modelled (the pipeline never seeds addressed replies); this mode measures the mutation-protection half only; affinity_without_copy Δ 0.000 [-0.001, 0.002]; copy Δ 0.001 [0.000, 0.003]; repetition Δ 0.000 [0.000, 0.000]; p95 41.7 ms (budget 150); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.038 [1.985, 2.091] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C7r20]**: insufficient data — window escape 2.039 [1.986, 2.091] (min 2.0); pool ECB 4.487 [4.456, 4.517] (floor 4.0); pool ECB share 0.897 [0.891, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C7r40]**: insufficient data — window escape 2.038 [1.983, 2.091] (min 2.0); pool ECB 4.483 [4.453, 4.513] (floor 4.0); pool ECB share 0.897 [0.891, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 885 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 38.4 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C7r20]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C7r40]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
