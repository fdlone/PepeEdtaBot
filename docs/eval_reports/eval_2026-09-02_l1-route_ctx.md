# Eval report 2026-09-02 snapshot=l1-route prompts=308b7deaea0f seeds=42,1337,2026 mode=ctx

Revision: `c16cc98`. Generations per configuration: 500.
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
| candidate_accept_rate | 0.881 [0.874, 0.888] | 0.881 [0.874, 0.888] | 0.000 [-0.000, 0.000] | 0.881 [0.874, 0.888] | 0.000 [-0.000, 0.000] |
| mean_response_length | 11.152 [10.888, 11.417] | 11.174 [10.909, 11.441] | 0.022 [-0.020, 0.062] | 11.179 [10.915, 11.451] | 0.027 [-0.017, 0.068] |
| unique_token_ratio | 0.987 [0.985, 0.988] | 0.987 [0.985, 0.988] | 0.000 [-0.000, 0.000] | 0.987 [0.985, 0.988] | -0.000 [-0.001, 0.000] |
| exact_context_copy_rate | 0.219 [0.198, 0.239] | 0.221 [0.199, 0.241] | 0.001 [-0.001, 0.004] | 0.221 [0.199, 0.241] | 0.001 [-0.001, 0.004] |
| repetition_rate | 0.003 [0.001, 0.005] | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.279 [0.264, 0.295] | 0.280 [0.264, 0.295] | 0.000 [-0.001, 0.001] | 0.280 [0.264, 0.295] | 0.000 [-0.001, 0.001] |
| context_affinity_without_copy | 0.211 [0.196, 0.228] | 0.211 [0.197, 0.227] | -0.000 [-0.001, 0.000] | 0.211 [0.197, 0.227] | -0.000 [-0.001, 0.000] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.309 [0.264, 0.355] | 0.309 [0.264, 0.355] | 0.000 [0.000, 0.000] | 0.309 [0.264, 0.355] | 0.000 [0.000, 0.000] |
| structural_pool_ecb | 4.430 [4.395, 4.464] | 4.435 [4.401, 4.469] | 0.005 [0.000, 0.011] | 4.432 [4.397, 4.465] | 0.002 [-0.003, 0.007] |
| structural_window_escape | 2.081 [2.027, 2.135] | 2.084 [2.029, 2.137] | 0.003 [-0.005, 0.011] | 2.083 [2.027, 2.137] | 0.002 [-0.004, 0.009] |

C0: distinct-2 = 0.652 (basis 15228), distinct-3 = 0.804 (basis 13730) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 26.5/40.5 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C7r20: distinct-2 = 0.651 (basis 15261), distinct-3 = 0.804 (basis 13763) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 25.5/38.9 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.56); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 13 draws, empty 0%; storage_delta: n/a.

C7r40: distinct-2 = 0.651 (basis 15268), distinct-3 = 0.803 (basis 13770) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 24.0/36.6 ms; cache_hit_rate: 40%; mean normalized entropy: 0.248 (branching 3.57); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 13 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.195 [0.157, 0.235] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.285 [0.243, 0.331] | 0.003 [0.000, 0.008] | 0.396 [0.369, 0.424] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.395 [0.344, 0.448] | 0.003 [0.000, 0.008] | 0.258 [0.232, 0.284] |
| C7r20 | generic | 375 | 1.000 [1.000, 1.000] | 0.197 [0.157, 0.237] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C7r20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.283 [0.243, 0.328] | 0.003 [0.000, 0.008] | 0.396 [0.369, 0.422] |
| C7r20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C7r20 | topical | 375 | 1.000 [1.000, 1.000] | 0.400 [0.347, 0.448] | 0.003 [0.000, 0.008] | 0.259 [0.233, 0.285] |
| C7r40 | generic | 375 | 1.000 [1.000, 1.000] | 0.197 [0.157, 0.237] | 0.005 [0.000, 0.013] | 0.309 [0.282, 0.339] |
| C7r40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.283 [0.243, 0.328] | 0.003 [0.000, 0.008] | 0.396 [0.369, 0.422] |
| C7r40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.140 [0.108, 0.172] |
| C7r40 | topical | 375 | 1.000 [1.000, 1.000] | 0.400 [0.347, 0.448] | 0.003 [0.000, 0.008] | 0.259 [0.233, 0.285] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.640 [0.629, 0.651] | 0.995 [0.991, 0.998] | 0.692 [0.670, 0.715] | 0.239 [0.219, 0.260] | 0.266 [0.240, 0.291] | 27.1 / 37.7 | F4_stale 2 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.104 [0.098, 0.109] | 0.462 [0.437, 0.487] | 0.244 [0.214, 0.276] | 0.201 [0.159, 0.249] | 0.183 [0.130, 0.237] | 23.8 / 30.1 | 0 |
| C0 | extension | 1500 | 0.256 [0.246, 0.266] | 0.751 [0.728, 0.772] | 0.264 [0.237, 0.291] | 0.139 [0.109, 0.170] | 0.077 [0.050, 0.107] | 28.6 / 22.9 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | vanilla | 1500 | 0.641 [0.630, 0.651] | 0.993 [0.989, 0.997] | 0.689 [0.664, 0.709] | 0.239 [0.220, 0.259] | 0.268 [0.244, 0.296] | 26.2 / 30.9 | F4_stale 2 |
| C7r20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r20 | mutated | 1500 | 0.103 [0.097, 0.108] | 0.458 [0.433, 0.483] | 0.249 [0.218, 0.282] | 0.198 [0.156, 0.245] | 0.187 [0.129, 0.246] | 22.9 / 29.0 | 0 |
| C7r20 | extension | 1500 | 0.255 [0.245, 0.266] | 0.748 [0.725, 0.769] | 0.267 [0.242, 0.293] | 0.140 [0.109, 0.173] | 0.080 [0.053, 0.110] | 27.6 / 22.0 | 0 |
| C7r20 | hot | 13 | 0.002 [0.001, 0.003] | 0.009 [0.005, 0.014] | 0.231 [0.000, 0.462] | insufficient data | 0.000 [0.000, 0.000] | 17.0 / 26.3 | 0 |
| C7r40 | vanilla | 1500 | 0.639 [0.629, 0.650] | 0.993 [0.989, 0.997] | 0.688 [0.662, 0.711] | 0.239 [0.220, 0.259] | 0.268 [0.241, 0.294] | 24.7 / 29.4 | F4_stale 2 |
| C7r40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C7r40 | mutated | 1500 | 0.103 [0.097, 0.109] | 0.460 [0.435, 0.486] | 0.251 [0.220, 0.284] | 0.198 [0.156, 0.245] | 0.185 [0.133, 0.243] | 21.6 / 27.4 | 0 |
| C7r40 | extension | 1500 | 0.254 [0.244, 0.264] | 0.746 [0.724, 0.767] | 0.264 [0.238, 0.290] | 0.140 [0.109, 0.173] | 0.081 [0.054, 0.115] | 26.1 / 20.6 | 0 |
| C7r40 | hot | 13 | 0.003 [0.002, 0.006] | 0.009 [0.005, 0.014] | 0.538 [0.231, 0.769] | insufficient data | 0.000 [0.000, 0.000] | 15.3 / 24.8 | 0 |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel[C7r20]**: insufficient data — ctx: seed draw not modelled (the pipeline never seeds addressed replies); this mode measures the mutation-protection half only; affinity_without_copy Δ -0.000 [-0.001, 0.000]; copy Δ 0.001 [-0.001, 0.004]; repetition Δ 0.000 [0.000, 0.000]; p95 38.9 ms (budget 150); gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel[C7r40]**: insufficient data — ctx: seed draw not modelled (the pipeline never seeds addressed replies); this mode measures the mutation-protection half only; affinity_without_copy Δ -0.000 [-0.001, 0.000]; copy Δ 0.001 [-0.001, 0.004]; repetition Δ 0.000 [0.000, 0.000]; p95 36.6 ms (budget 150); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.081 [2.027, 2.135] (min 2.0); pool ECB 4.430 [4.395, 4.464] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C7r20]**: insufficient data — window escape 2.084 [2.029, 2.137] (min 2.0); pool ECB 4.435 [4.401, 4.469] (floor 4.0); pool ECB share 0.887 [0.880, 0.894] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C7r40]**: insufficient data — window escape 2.083 [2.027, 2.137] (min 2.0); pool ECB 4.432 [4.397, 4.465] (floor 4.0); pool ECB share 0.886 [0.879, 0.893] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:38%, 2:28%, 3:22%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 937 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 40.5 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7r20]**: pass — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f
- **meme_regression[C7r40]**: pass — 13/27 memes reproduced (48%); C0 13/27 (48%), tolerance 10%; prompt set 308b7deaea0f

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
