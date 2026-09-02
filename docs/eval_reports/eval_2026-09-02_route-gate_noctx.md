# Eval report 2026-09-02 snapshot=route-gate prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `49cb23c`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
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
| candidate_accept_rate | 1.000 [1.000, 1.000] | 0.998 [0.997, 0.999] | -0.002 [-0.003, -0.001] * | 0.997 [0.995, 0.998] | -0.003 [-0.005, -0.002] * |
| mean_response_length | 10.460 [10.248, 10.693] | 10.492 [10.268, 10.741] | 0.032 [-0.221, 0.287] | 10.403 [10.164, 10.655] | -0.057 [-0.309, 0.199] |
| unique_token_ratio | 0.987 [0.986, 0.989] | 0.988 [0.986, 0.989] | 0.001 [-0.002, 0.003] | 0.989 [0.987, 0.990] | 0.001 [-0.001, 0.003] |
| exact_context_copy_rate | 0.003 [0.001, 0.005] | 0.004 [0.001, 0.007] | 0.001 [-0.001, 0.004] | 0.006 [0.003, 0.010] | 0.003 [0.000, 0.007] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.025 [0.021, 0.029] | 0.032 [0.027, 0.038] | 0.008 [0.002, 0.013] * | 0.037 [0.032, 0.043] | 0.013 [0.007, 0.018] * |
| context_affinity_without_copy | 0.025 [0.021, 0.029] | 0.032 [0.027, 0.037] | 0.007 [0.002, 0.012] * | 0.037 [0.031, 0.042] | 0.012 [0.006, 0.018] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.053 [0.035, 0.077] | 0.099 [0.072, 0.131] | 0.045 [0.008, 0.085] * | 0.117 [0.083, 0.152] | 0.064 [0.027, 0.101] * |
| structural_pool_ecb | 4.527 [4.497, 4.555] | 4.621 [4.595, 4.647] | 0.094 [0.062, 0.129] * | 4.663 [4.639, 4.689] | 0.136 [0.104, 0.171] * |
| structural_window_escape | 2.868 [2.815, 2.917] | 2.819 [2.771, 2.867] | -0.049 [-0.112, 0.013] | 2.753 [2.708, 2.798] | -0.115 [-0.174, -0.058] * |

C0: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 13.7/22.4 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C11a20: distinct-2 = 0.709 (basis 14238), distinct-3 = 0.860 (basis 12738) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 15.0/23.5 ms; cache_hit_rate: 26%; mean normalized entropy: 0.243 (branching 3.44); mean applied temperature: 2.65; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C11a40: distinct-2 = 0.703 (basis 14104), distinct-3 = 0.855 (basis 12604) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 15.4/24.5 ms; cache_hit_rate: 26%; mean normalized entropy: 0.248 (branching 3.48); mean applied temperature: 2.62; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.024, 0.037] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.042] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.020 [0.016, 0.024] |
| C11a20 | generic | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.026 [0.021, 0.032] |
| C11a20 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.037 [0.029, 0.045] |
| C11a20 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.043 [0.026, 0.060] |
| C11a20 | topical | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.024 [0.019, 0.030] |
| C11a40 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.003 [0.000, 0.008] | 0.033 [0.027, 0.041] |
| C11a40 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.054 [0.042, 0.068] |
| C11a40 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.033 [0.019, 0.050] |
| C11a40 | topical | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.028 [0.022, 0.033] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.551 [0.526, 0.577] | 0.021 [0.016, 0.027] | 0.005 [0.001, 0.010] | 14.3 / 14.9 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.279 [0.246, 0.314] | 0.022 [0.016, 0.028] | 0.000 [0.000, 0.000] | 12.7 / 15.6 | 0 |
| C0 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.389 [0.364, 0.416] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 14.5 / 12.9 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11a20 | vanilla | 1500 | 0.501 [0.491, 0.513] | 0.973 [0.965, 0.981] | 0.499 [0.474, 0.525] | 0.024 [0.018, 0.031] | 0.008 [0.003, 0.015] | 15.5 / 15.9 | 0 |
| C11a20 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11a20 | mutated | 1500 | 0.082 [0.077, 0.088] | 0.387 [0.363, 0.411] | 0.241 [0.207, 0.278] | 0.023 [0.014, 0.036] | 0.000 [0.000, 0.000] | 13.5 / 16.8 | 0 |
| C11a20 | extension | 1500 | 0.272 [0.262, 0.282] | 0.809 [0.790, 0.829] | 0.386 [0.360, 0.412] | 0.029 [0.021, 0.038] | 0.000 [0.000, 0.000] | 15.8 / 14.5 | 0 |
| C11a20 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11a20 | assoc | 1500 | 0.144 [0.140, 0.149] | 0.722 [0.699, 0.747] | 0.151 [0.132, 0.173] | 0.082 [0.059, 0.107] | 0.000 [0.000, 0.000] | 15.8 / 14.8 | F4_stale 17 |
| C11a40 | vanilla | 1500 | 0.429 [0.419, 0.440] | 0.956 [0.945, 0.966] | 0.451 [0.426, 0.477] | 0.024 [0.018, 0.032] | 0.011 [0.003, 0.019] | 15.9 / 17.0 | 0 |
| C11a40 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11a40 | mutated | 1500 | 0.071 [0.066, 0.076] | 0.340 [0.316, 0.364] | 0.275 [0.239, 0.316] | 0.020 [0.012, 0.029] | 0.000 [0.000, 0.000] | 13.9 / 16.9 | 0 |
| C11a40 | extension | 1500 | 0.234 [0.225, 0.243] | 0.756 [0.733, 0.777] | 0.354 [0.325, 0.383] | 0.032 [0.023, 0.042] | 0.000 [0.000, 0.000] | 16.2 / 14.9 | 0 |
| C11a40 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11a40 | assoc | 1500 | 0.266 [0.257, 0.275] | 0.731 [0.708, 0.756] | 0.284 [0.255, 0.311] | 0.074 [0.060, 0.092] | 0.006 [0.000, 0.016] | 16.0 / 15.6 | F4_stale 29 |

## Gates

- **phase2_entropy[C11a20]**: fail — copy Δ 0.001 [-0.001, 0.004]; distinct-2 Δ 0.004 [-0.017, 0.021]; distinct-3 Δ 0.002 [-0.024, 0.025]; affinity_without_copy Δ 0.007 [0.002, 0.012] *; p95 23.5 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11a40]**: fail — copy Δ 0.003 [0.000, 0.007]; distinct-2 Δ -0.002 [-0.020, 0.017]; distinct-3 Δ -0.004 [-0.027, 0.021]; affinity_without_copy Δ 0.012 [0.006, 0.018] *; p95 24.5 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11a20]**: insufficient data — route assoc: present in 72.2% of pools (floor 10%); single_trajectory_share Δ 0.005 [-0.012, 0.021]; affinity_without_copy Δ 0.007 [0.002, 0.012] *; copy Δ 0.001 [-0.001, 0.004]; repetition Δ 0.000 [0.000, 0.000]; pool ECB 4.621 (floor 4.0); p95 23.5 ms (budget 150) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11a40]**: insufficient data — route assoc: present in 73.1% of pools (floor 10%); single_trajectory_share Δ 0.007 [-0.010, 0.023]; affinity_without_copy Δ 0.012 [0.006, 0.018] *; copy Δ 0.003 [0.000, 0.007]; repetition Δ 0.001 [0.000, 0.002]; pool ECB 4.663 (floor 4.0); p95 24.5 ms (budget 150) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11a20]**: insufficient data — window escape 2.819 [2.771, 2.867] (min 2.0); pool ECB 4.621 [4.595, 4.647] (floor 4.0); pool ECB share 0.924 [0.919, 0.929] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:29%, 3:39%, 4:20%, 5:4%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11a40]**: insufficient data — window escape 2.753 [2.708, 2.798] (min 2.0); pool ECB 4.663 [4.639, 4.689] (floor 4.0); pool ECB share 0.933 [0.928, 0.938] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:33%, 3:36%, 4:19%, 5:3%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1552 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 22.4 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11a20]**: pass — 9/18 memes reproduced (50%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11a40]**: pass — 9/18 memes reproduced (50%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
