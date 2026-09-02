# Eval report 2026-09-02 snapshot=route-gate prompts=bababb4b7693 seeds=42,1337,2026 mode=ctx

Revision: `49cb23c`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C11a20**: 1500 generations
- **C11a40**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C11a20 | Δ C11a20 vs C0 | C11a40 | Δ C11a40 vs C0 |
|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.870 [0.862, 0.877] | 0.877 [0.870, 0.884] | 0.008 [0.001, 0.015] * | 0.888 [0.881, 0.894] | 0.018 [0.012, 0.025] * |
| mean_response_length | 10.917 [10.691, 11.185] | 10.805 [10.553, 11.061] | -0.111 [-0.370, 0.174] | 11.019 [10.749, 11.293] | 0.103 [-0.159, 0.399] |
| unique_token_ratio | 0.984 [0.981, 0.986] | 0.985 [0.983, 0.987] | 0.002 [-0.001, 0.004] | 0.985 [0.983, 0.986] | 0.001 [-0.001, 0.003] |
| exact_context_copy_rate | 0.226 [0.205, 0.247] | 0.203 [0.183, 0.223] | -0.023 [-0.043, -0.001] * | 0.196 [0.176, 0.216] | -0.030 [-0.050, -0.010] * |
| repetition_rate | 0.002 [0.000, 0.005] | 0.001 [0.000, 0.002] | -0.001 [-0.004, 0.001] | 0.002 [0.000, 0.005] | 0.000 [-0.003, 0.003] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.321 [0.305, 0.336] | 0.320 [0.305, 0.335] | -0.001 [-0.019, 0.016] | 0.300 [0.284, 0.315] | -0.021 [-0.039, -0.003] * |
| context_affinity_without_copy | 0.253 [0.235, 0.273] | 0.260 [0.242, 0.278] | 0.008 [-0.009, 0.026] | 0.237 [0.218, 0.253] | -0.011 [-0.028, 0.007] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.285 [0.243, 0.333] | 0.277 [0.229, 0.325] | -0.008 [-0.067, 0.048] | 0.237 [0.197, 0.280] | -0.048 [-0.101, 0.008] |
| structural_pool_ecb | 4.485 [4.452, 4.515] | 4.588 [4.559, 4.617] | 0.103 [0.068, 0.139] * | 4.663 [4.637, 4.687] | 0.179 [0.144, 0.213] * |
| structural_window_escape | 2.038 [1.985, 2.091] | 2.007 [1.954, 2.063] | -0.031 [-0.086, 0.027] | 2.053 [2.000, 2.107] | 0.015 [-0.043, 0.073] |

C0: distinct-2 = 0.661 (basis 14875), distinct-3 = 0.818 (basis 13376) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 22.4/35.8 ms; cache_hit_rate: 41%; mean normalized entropy: 0.242 (branching 3.46); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11a20: distinct-2 = 0.652 (basis 14708), distinct-3 = 0.812 (basis 13208) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 24.0/37.5 ms; cache_hit_rate: 40%; mean normalized entropy: 0.245 (branching 3.47); mean applied temperature: 2.73; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

C11a40: distinct-2 = 0.660 (basis 15029), distinct-3 = 0.815 (basis 13529) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 22.1/34.3 ms; cache_hit_rate: 40%; mean normalized entropy: 0.245 (branching 3.53); mean applied temperature: 2.70; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.232 [0.189, 0.275] | 0.000 [0.000, 0.000] | 0.350 [0.321, 0.382] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.315 [0.267, 0.360] | 0.008 [0.000, 0.019] | 0.364 [0.337, 0.392] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.266 [0.225, 0.306] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.355 [0.307, 0.403] | 0.000 [0.000, 0.000] | 0.297 [0.269, 0.324] |
| C11a20 | generic | 375 | 1.000 [1.000, 1.000] | 0.221 [0.181, 0.264] | 0.003 [0.000, 0.008] | 0.361 [0.330, 0.388] |
| C11a20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.264 [0.224, 0.312] | 0.000 [0.000, 0.000] | 0.373 [0.344, 0.403] |
| C11a20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.263 [0.222, 0.303] |
| C11a20 | topical | 375 | 1.000 [1.000, 1.000] | 0.320 [0.275, 0.365] | 0.000 [0.000, 0.000] | 0.277 [0.251, 0.305] |
| C11a40 | generic | 375 | 1.000 [1.000, 1.000] | 0.197 [0.157, 0.235] | 0.003 [0.000, 0.008] | 0.330 [0.301, 0.360] |
| C11a40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.259 [0.216, 0.301] | 0.003 [0.000, 0.008] | 0.342 [0.315, 0.371] |
| C11a40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.258 [0.218, 0.299] |
| C11a40 | topical | 375 | 1.000 [1.000, 1.000] | 0.323 [0.275, 0.371] | 0.003 [0.000, 0.008] | 0.264 [0.239, 0.291] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.651 [0.639, 0.662] | 0.991 [0.985, 0.995] | 0.711 [0.687, 0.734] | 0.300 [0.277, 0.323] | 0.280 [0.256, 0.307] | 23.3 / 43.6 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.097 [0.091, 0.103] | 0.444 [0.421, 0.469] | 0.227 [0.195, 0.261] | 0.190 [0.140, 0.244] | 0.219 [0.152, 0.285] | 21.4 / 25.1 | 0 |
| C0 | extension | 1500 | 0.252 [0.241, 0.263] | 0.743 [0.721, 0.765] | 0.263 [0.239, 0.290] | 0.151 [0.120, 0.184] | 0.034 [0.017, 0.055] | 25.0 / 19.2 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11a20 | vanilla | 1500 | 0.564 [0.553, 0.573] | 0.991 [0.985, 0.995] | 0.692 [0.670, 0.715] | 0.310 [0.289, 0.332] | 0.261 [0.236, 0.289] | 25.1 / 33.1 | 0 |
| C11a20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11a20 | mutated | 1500 | 0.077 [0.072, 0.083] | 0.360 [0.335, 0.385] | 0.222 [0.185, 0.259] | 0.198 [0.143, 0.262] | 0.225 [0.150, 0.300] | 22.3 / 26.8 | 0 |
| C11a20 | extension | 1500 | 0.215 [0.205, 0.225] | 0.696 [0.673, 0.719] | 0.233 [0.209, 0.259] | 0.137 [0.108, 0.171] | 0.037 [0.016, 0.066] | 27.0 / 21.0 | 0 |
| C11a20 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11a20 | assoc | 1500 | 0.144 [0.140, 0.149] | 0.722 [0.699, 0.747] | 0.100 [0.082, 0.117] | 0.224 [0.171, 0.279] | 0.000 [0.000, 0.000] | 25.3 / 24.8 | F4_stale 17 |
| C11a40 | vanilla | 1500 | 0.484 [0.473, 0.494] | 0.969 [0.960, 0.978] | 0.653 [0.625, 0.676] | 0.293 [0.269, 0.316] | 0.287 [0.258, 0.316] | 22.9 / 27.4 | 0 |
| C11a40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11a40 | mutated | 1500 | 0.064 [0.059, 0.069] | 0.305 [0.283, 0.329] | 0.231 [0.194, 0.273] | 0.180 [0.122, 0.241] | 0.170 [0.104, 0.245] | 19.7 / 24.5 | 0 |
| C11a40 | extension | 1500 | 0.186 [0.177, 0.196] | 0.631 [0.605, 0.657] | 0.248 [0.223, 0.275] | 0.131 [0.102, 0.162] | 0.013 [0.000, 0.030] | 24.7 / 20.1 | 0 |
| C11a40 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11a40 | assoc | 1500 | 0.266 [0.257, 0.275] | 0.731 [0.708, 0.756] | 0.191 [0.169, 0.213] | 0.199 [0.164, 0.236] | 0.005 [0.000, 0.014] | 22.5 / 24.4 | F4_stale 29 |

## Gates

- **phase2_entropy[C11a20]**: fail — copy Δ -0.023 [-0.043, -0.001] *; distinct-2 Δ -0.009 [-0.023, 0.016]; distinct-3 Δ -0.006 [-0.026, 0.021]; affinity_without_copy Δ 0.008 [-0.009, 0.026]; p95 37.5 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11a40]**: fail — copy Δ -0.030 [-0.050, -0.010] *; distinct-2 Δ -0.000 [-0.018, 0.020]; distinct-3 Δ -0.003 [-0.025, 0.023]; affinity_without_copy Δ -0.011 [-0.028, 0.007]; p95 34.3 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured ctx only (noctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured ctx only (noctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11a20]**: insufficient data — route assoc: present in 72.2% of pools (floor 10%); single_trajectory_share Δ 0.021 [-0.009, 0.049]; affinity_without_copy Δ 0.008 [-0.009, 0.026]; copy Δ -0.023 [-0.043, -0.001] *; repetition Δ -0.001 [-0.004, 0.001]; pool ECB 4.588 (floor 4.0); p95 37.5 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **route_gate[C11a40]**: insufficient data — route assoc: present in 73.1% of pools (floor 10%); single_trajectory_share Δ -0.004 [-0.035, 0.025]; affinity_without_copy Δ -0.011 [-0.028, 0.007]; copy Δ -0.030 [-0.050, -0.010] *; repetition Δ 0.000 [-0.003, 0.003]; pool ECB 4.663 (floor 4.0); p95 34.3 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol not conducted) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.038 [1.985, 2.091] (min 2.0); pool ECB 4.485 [4.452, 4.515] (floor 4.0); pool ECB share 0.897 [0.890, 0.903] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:22%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11a20]**: insufficient data — window escape 2.007 [1.954, 2.063] (min 2.0); pool ECB 4.588 [4.559, 4.617] (floor 4.0); pool ECB share 0.918 [0.912, 0.923] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:42%, 2:26%, 3:21%, 4:8%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C11a40]**: insufficient data — window escape 2.053 [2.000, 2.107] (min 2.0); pool ECB 4.663 [4.637, 4.687] (floor 4.0); pool ECB share 0.933 [0.927, 0.937] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:40%, 2:28%, 3:21%, 4:9%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 885 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 35.8 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11a20]**: pass — 17/18 memes reproduced (94%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11a40]**: pass — 16/18 memes reproduced (89%); C0 17/18 (94%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
