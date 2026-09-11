# Eval report 2026-09-11 snapshot=route-selection prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `4123e21`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
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
| candidate_accept_rate | 1.000 [0.999, 1.000] | 1.000 [0.999, 1.000] | 0.000 [0.000, 0.000] | 0.999 [0.999, 1.000] | -0.000 [-0.001, 0.000] |
| mean_response_length | 10.685 [10.460, 10.933] | 10.555 [10.312, 10.815] | -0.129 [-0.215, -0.051] * | 10.570 [10.315, 10.818] | -0.115 [-0.415, 0.166] |
| unique_token_ratio | 0.986 [0.984, 0.988] | 0.987 [0.985, 0.989] | 0.001 [0.000, 0.002] * | 0.987 [0.985, 0.989] | 0.001 [-0.002, 0.003] |
| exact_context_copy_rate | 0.003 [0.001, 0.005] | 0.002 [0.000, 0.005] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.003] | -0.001 [-0.004, 0.001] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.034 [0.028, 0.041] | 0.035 [0.029, 0.041] | 0.000 [-0.003, 0.003] | 0.090 [0.081, 0.100] | 0.055 [0.047, 0.065] * |
| context_affinity_without_copy | 0.035 [0.029, 0.041] | 0.035 [0.029, 0.041] | 0.000 [-0.002, 0.003] | 0.090 [0.081, 0.100] | 0.055 [0.046, 0.065] * |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.083 [0.056, 0.112] | 0.080 [0.053, 0.109] | -0.003 [-0.008, 0.000] | 0.125 [0.093, 0.160] | 0.043 [0.003, 0.085] * |
| structural_pool_ecb | 4.558 [4.527, 4.589] | 4.558 [4.527, 4.589] | 0.000 [0.000, 0.000] | 4.653 [4.627, 4.678] | 0.095 [0.061, 0.126] * |
| structural_window_escape | 3.298 [3.251, 3.346] | 3.353 [3.307, 3.400] | 0.055 [0.033, 0.075] * | 3.263 [3.217, 3.308] | -0.035 [-0.095, 0.022] |

C0: distinct-2 = 0.710 (basis 14527), distinct-3 = 0.861 (basis 13027) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.0/22.8 ms; cache_hit_rate: 27%; mean normalized entropy: 0.248 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C11d: distinct-2 = 0.715 (basis 14333), distinct-3 = 0.866 (basis 12833) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 14.4/23.4 ms; cache_hit_rate: 27%; mean normalized entropy: 0.248 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

C11s: distinct-2 = 0.675 (basis 14355), distinct-3 = 0.827 (basis 12855) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 16.1/24.3 ms; cache_hit_rate: 27%; mean normalized entropy: 0.247 (branching 3.52); mean applied temperature: 2.62; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 0%; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.027 [0.021, 0.033] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.033 [0.027, 0.041] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.060 [0.038, 0.086] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.021 [0.017, 0.025] |
| C11d | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.027 [0.021, 0.032] |
| C11d | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.034 [0.028, 0.041] |
| C11d | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.060 [0.038, 0.087] |
| C11d | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.021 [0.017, 0.026] |
| C11s | generic | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.063 [0.053, 0.073] |
| C11s | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.096 [0.080, 0.113] |
| C11s | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.153 [0.124, 0.188] |
| C11s | topical | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.053 [0.045, 0.062] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.603 [0.592, 0.613] | 0.992 [0.987, 0.996] | 0.549 [0.524, 0.575] | 0.030 [0.023, 0.039] | 0.005 [0.001, 0.010] | 14.7 / 14.1 | F4_stale 1 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.096 [0.089, 0.102] | 0.426 [0.400, 0.453] | 0.280 [0.246, 0.313] | 0.023 [0.013, 0.037] | 0.000 [0.000, 0.000] | 13.1 / 15.9 | 0 |
| C0 | extension | 1500 | 0.301 [0.292, 0.311] | 0.847 [0.829, 0.865] | 0.397 [0.370, 0.423] | 0.045 [0.033, 0.059] | 0.000 [0.000, 0.000] | 15.0 / 12.9 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | vanilla | 1500 | 0.603 [0.592, 0.613] | 0.992 [0.987, 0.996] | 0.569 [0.544, 0.594] | 0.028 [0.022, 0.035] | 0.004 [0.000, 0.008] | 15.0 / 14.1 | F4_stale 1 |
| C11d | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | mutated | 1500 | 0.096 [0.089, 0.102] | 0.426 [0.400, 0.453] | 0.261 [0.225, 0.294] | 0.031 [0.016, 0.051] | 0.000 [0.000, 0.000] | 13.3 / 16.3 | 0 |
| C11d | extension | 1500 | 0.301 [0.292, 0.311] | 0.847 [0.829, 0.865] | 0.382 [0.357, 0.408] | 0.049 [0.036, 0.065] | 0.000 [0.000, 0.000] | 15.4 / 13.0 | 0 |
| C11d | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11d | phrase | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | vanilla | 1500 | 0.421 [0.410, 0.431] | 0.949 [0.937, 0.960] | 0.448 [0.420, 0.475] | 0.030 [0.022, 0.039] | 0.003 [0.000, 0.008] | 16.6 / 16.9 | 0 |
| C11s | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | mutated | 1500 | 0.071 [0.066, 0.076] | 0.338 [0.315, 0.361] | 0.221 [0.185, 0.258] | 0.027 [0.010, 0.049] | 0.000 [0.000, 0.000] | 14.9 / 17.4 | 0 |
| C11s | extension | 1500 | 0.219 [0.210, 0.229] | 0.717 [0.692, 0.741] | 0.335 [0.306, 0.360] | 0.044 [0.029, 0.061] | 0.000 [0.000, 0.000] | 17.0 / 15.4 | 0 |
| C11s | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | assoc | 0 | not attempted | — | — | — | — | — | — | — |
| C11s | phrase | 1500 | 0.289 [0.280, 0.299] | 0.758 [0.735, 0.783] | 0.343 [0.317, 0.371] | 0.242 [0.221, 0.265] | 0.000 [0.000, 0.000] | 16.3 / 17.3 | F4_stale 4 |

## Gates

- **phase2_entropy[C11d]**: fail — copy Δ -0.001 [-0.002, 0.000]; distinct-2 Δ 0.005 [-0.018, 0.023]; distinct-3 Δ 0.004 [-0.025, 0.026]; affinity_without_copy Δ 0.000 [-0.002, 0.003]; p95 23.4 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C11s]**: fail — copy Δ -0.001 [-0.004, 0.001]; distinct-2 Δ -0.035 [-0.040, -0.001] *; distinct-3 Δ -0.035 [-0.043, 0.006]; affinity_without_copy Δ 0.055 [0.046, 0.065] *; p95 24.3 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled); gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **selection_window**: insufficient data — no selection-window arm in this run (window knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **assoc_pilot**: insufficient data — no assoc-pilot arm in this run (assoc_slot_ratio at its default); gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11d]**: insufficient data — route under test is not exactly one new route in the arm's pools: none; gate requires both context modes, this run measured noctx only (ctx not measured)
- **route_gate[C11s]**: insufficient data — route phrase: present in 75.8% of pools (floor 10%); single_trajectory_share Δ -0.003 [-0.010, 0.004]; affinity_without_copy Δ 0.055 [0.046, 0.065] *; copy Δ -0.001 [-0.004, 0.001]; repetition Δ 0.000 [0.000, 0.000]; pool ECB 4.653 (floor 4.0); p95 24.3 ms (budget 150) — single-trajectory share did not drop significantly by 5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 3.298 [3.251, 3.346] (min 2.0); pool ECB 4.558 [4.527, 4.589] (floor 4.0); pool ECB share 0.912 [0.905, 0.918] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:2%, 2:19%, 3:38%, 4:31%, 5:10%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11d]**: insufficient data — window escape 3.353 [3.307, 3.400] (min 2.0); pool ECB 4.558 [4.527, 4.589] (floor 4.0); pool ECB share 0.912 [0.905, 0.918] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:1%, 2:17%, 3:38%, 4:32%, 5:11%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C11s]**: insufficient data — window escape 3.263 [3.217, 3.308] (min 2.0); pool ECB 4.653 [4.627, 4.678] (floor 4.0); pool ECB share 0.931 [0.925, 0.936] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:1%, 2:19%, 3:40%, 4:32%, 5:8%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1328 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 22.8 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 8/18 memes reproduced (44%); C0 8/18 (44%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11d]**: pass — 8/18 memes reproduced (44%); C0 8/18 (44%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C11s]**: pass — 13/18 memes reproduced (72%); C0 8/18 (44%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
