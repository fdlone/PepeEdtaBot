# Eval report 2026-09-02 snapshot=baseline-2026-09-01 prompts=bababb4b7693 seeds=42,1337,2026 mode=noctx

Revision: `1d36e26`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C3**: 1500 generations
- **C4**: 1500 generations
- **CF**: 1500 generations — results shared with C0
- unavailable (feature not implemented yet): C1, C2, C5

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI **парный**], `*` = significant (interval excludes 0, doc 05 §4).

> Интервал дельты в ЭТОЙ таблице — **парный**: армы идут по одним и тем же промптам и сидам, поэтому ресэмплируются пары наблюдений, а не два арма независимо. С отчётами до 2026-08-26 ширина интервалов **несопоставима** — там дельта считалась независимым ресэмплингом и интервал был шире истинного тем сильнее, чем выше корреляция армов. Точечные оценки сопоставимы: они не изменились. **distinct-2/3 парность НЕ получили** — их дельта считается по целым ответам (`distinct_delta_ci`) и остаётся непарной; там интервал по-прежнему шире истинного, то есть вердикт консервативен, но сравнивать его ширину с таблицей выше нельзя.

| metric | C0 | C3 | Δ C3 vs C0 | C4 | Δ C4 vs C0 | CF | Δ CF vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.992 [0.990, 0.993] | -0.008 [-0.010, -0.006] * | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| mean_response_length | 10.460 [10.248, 10.693] | 11.171 [10.909, 11.445] | 0.711 [0.487, 0.937] * | 10.527 [10.268, 10.783] | 0.067 [-0.175, 0.335] | 10.460 [10.248, 10.693] | 0.000 [0.000, 0.000] |
| unique_token_ratio | 0.987 [0.986, 0.989] | 0.985 [0.984, 0.987] | -0.002 [-0.004, -0.000] * | 0.988 [0.987, 0.990] | 0.001 [-0.001, 0.003] | 0.987 [0.986, 0.989] | 0.000 [0.000, 0.000] |
| exact_context_copy_rate | 0.003 [0.001, 0.005] | 0.003 [0.001, 0.007] | 0.001 [-0.002, 0.004] | 0.021 [0.013, 0.027] | 0.018 [0.011, 0.025] * | 0.003 [0.001, 0.005] | 0.000 [0.000, 0.000] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.004 [0.001, 0.007] | 0.004 [0.001, 0.007] * | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.025 [0.021, 0.029] | 0.026 [0.022, 0.030] | 0.001 [-0.003, 0.005] | 0.080 [0.071, 0.089] | 0.055 [0.047, 0.064] * | 0.025 [0.021, 0.029] | 0.000 [0.000, 0.000] |
| context_affinity_without_copy | 0.025 [0.021, 0.029] | 0.026 [0.022, 0.030] | 0.001 [-0.003, 0.005] | 0.072 [0.065, 0.080] | 0.048 [0.040, 0.056] * | 0.025 [0.021, 0.029] | 0.000 [0.000, 0.000] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.053 [0.035, 0.077] | 0.115 [0.083, 0.144] | 0.061 [0.032, 0.093] * | 0.128 [0.096, 0.165] | 0.075 [0.035, 0.120] * | 0.053 [0.035, 0.077] | 0.000 [0.000, 0.000] |
| structural_pool_ecb | 4.527 [4.497, 4.555] | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] | 4.686 [4.661, 4.710] | 0.159 [0.125, 0.195] * | 4.527 [4.497, 4.555] | 0.000 [0.000, 0.000] |
| structural_window_escape | 2.868 [2.815, 2.917] | 2.080 [2.029, 2.131] | -0.788 [-0.848, -0.726] * | 2.671 [2.623, 2.720] | -0.197 [-0.262, -0.137] * | 2.868 [2.815, 2.917] | 0.000 [0.000, 0.000] |

C0: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 18.7/34.9 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C3: distinct-2 = 0.671 (basis 15256), distinct-3 = 0.827 (basis 13756) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 19.2/31.9 ms; cache_hit_rate: 27%; mean normalized entropy: 0.246 (branching 3.48); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

C4: distinct-2 = 0.704 (basis 14290), distinct-3 = 0.849 (basis 12790) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 18.4/32.2 ms; cache_hit_rate: 25%; mean normalized entropy: 0.242 (branching 3.42); mean applied temperature: 2.62; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); hot-ngram seeds: 345 draws, empty 100%; storage_delta: n/a.

CF: distinct-2 = 0.705 (basis 14190), distinct-3 = 0.859 (basis 12690) — type/token ratios, comparable only at equal basis; их дельта считается НЕпарным бутстрапом (`distinct_delta_ci`), в отличие от таблицы метрик — интервал шире истинного, вердикт консервативен; latency p50/p95 = 18.7/34.9 ms; cache_hit_rate: insufficient data; mean normalized entropy: insufficient data; mean applied temperature: insufficient data; temporal blend: insufficient data; order interpolation: insufficient data; shadow order-4 share: insufficient data; hot-ngram seeds: no draws; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.030 [0.024, 0.037] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.026 [0.013, 0.042] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.020 [0.016, 0.024] |
| C3 | generic | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.003 [0.000, 0.008] | 0.025 [0.020, 0.030] |
| C3 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.005 [0.000, 0.013] | 0.030 [0.024, 0.037] |
| C3 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.003 [0.000, 0.011] | 0.027 [0.014, 0.042] |
| C3 | topical | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.005 [0.000, 0.013] | 0.022 [0.018, 0.026] |
| C4 | generic | 375 | 1.000 [1.000, 1.000] | 0.019 [0.005, 0.035] | 0.000 [0.000, 0.000] | 0.079 [0.063, 0.094] |
| C4 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.021 [0.008, 0.037] | 0.000 [0.000, 0.000] | 0.089 [0.074, 0.104] |
| C4 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.086 [0.060, 0.113] |
| C4 | topical | 375 | 1.000 [1.000, 1.000] | 0.037 [0.019, 0.056] | 0.000 [0.000, 0.000] | 0.067 [0.055, 0.079] |

## Per-route breakdown (M3R-103)

Маршрут — механизм, построивший кандидата (`CandidateRoute`), как его атрибутировал генератор при создании. Два знаменателя раздельно: **доля пула** — кандидаты маршрута среди всех кандидатов генерации; **присутствие** — доля генераций, где маршрут положил хотя бы одного кандидата; **win given present** — доля побед среди них. Affinity без копий и copy — по ответам, которые выиграл маршрут. Латентность — средняя по генерациям с маршрутом в пуле / без него: верхняя оценка цены маршрута, не измерение его шага. Отклонения — до пула, по классам M3R-021, из телеметрии генератора. `not attempted` — механизм маршрута в этой конфигурации не запускался (не то же, что «запускался и ничего не произвёл»).

| config | route | attempts | pool share | presence | win given present | winners' affinity w/o copy | winners' copy | latency with / without, ms | rejected before pool (F-classes) |
|---|---|---|---|---|---|---|---|---|---|
| C0 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.551 [0.526, 0.577] | 0.021 [0.016, 0.027] | 0.005 [0.001, 0.010] | 20.4 / 19.6 | 0 |
| C0 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C0 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.279 [0.246, 0.314] | 0.022 [0.016, 0.028] | 0.000 [0.000, 0.000] | 18.4 / 22.1 | 0 |
| C0 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.389 [0.364, 0.416] | 0.032 [0.024, 0.041] | 0.000 [0.000, 0.000] | 20.9 / 17.9 | 0 |
| C0 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C3 | vanilla | 1500 | 0.587 [0.576, 0.599] | 0.989 [0.983, 0.994] | 0.552 [0.528, 0.575] | 0.024 [0.019, 0.030] | 0.006 [0.001, 0.012] | 20.1 / 21.7 | 0 |
| C3 | seeded | 0 | not attempted | — | — | — | — | — | — | — |
| C3 | mutated | 1500 | 0.101 [0.095, 0.107] | 0.455 [0.429, 0.481] | 0.242 [0.208, 0.273] | 0.022 [0.016, 0.030] | 0.000 [0.000, 0.000] | 17.9 / 22.0 | 0 |
| C3 | extension | 1500 | 0.312 [0.301, 0.322] | 0.845 [0.827, 0.865] | 0.408 [0.382, 0.434] | 0.029 [0.022, 0.037] | 0.000 [0.000, 0.000] | 20.5 / 18.2 | 0 |
| C3 | hot | 0 | not attempted | — | — | — | — | — | — | — |
| C4 | vanilla | 1500 | 0.425 [0.415, 0.437] | 0.945 [0.933, 0.957] | 0.463 [0.440, 0.491] | 0.024 [0.017, 0.033] | 0.009 [0.003, 0.017] | 19.6 / 21.1 | 0 |
| C4 | seeded | 1500 | 0.270 [0.261, 0.278] | 0.799 [0.777, 0.821] | 0.262 [0.235, 0.288] | 0.255 [0.233, 0.280] | 0.080 [0.051, 0.108] | 20.0 / 18.4 | F4_stale 71, F6_structural_repeat 5 |
| C4 | mutated | 1500 | 0.066 [0.061, 0.071] | 0.320 [0.295, 0.343] | 0.242 [0.206, 0.281] | 0.013 [0.008, 0.020] | 0.000 [0.000, 0.000] | 17.2 / 20.8 | 0 |
| C4 | extension | 1500 | 0.239 [0.230, 0.249] | 0.755 [0.732, 0.775] | 0.365 [0.338, 0.396] | 0.033 [0.022, 0.043] | 0.000 [0.000, 0.000] | 19.8 / 19.3 | 0 |
| C4 | hot | 0 | not attempted | — | — | — | — | — | — | — |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes[C3]**: insufficient data — copy Δ 0.001 [-0.002, 0.004]; affinity_without_copy Δ 0.001 [-0.003, 0.005]; p95 31.9 ms (budget 150) — missing: manual top-meme rating (doc 05 §5)
- **phase5_promotion[C4]**: insufficient data — prod corpus: n_docs 232 (floor 500), singleton share 80% (ceiling 60%); seeded present 80% (bar 30%), win|present 26% (bar 40%); affinity_without_copy Δ 0.048 [0.040, 0.056] *; p95 32.2 ms (budget 150) — missing: prod n_docs 232 below the floor 500, df singleton share 80% above the ceiling 60%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp**: insufficient data — no Phase 9 arm in this run; gate requires both context modes, this run measured noctx only (ctx not measured)
- **l1_hot_channel**: insufficient data — no L1 arm in this run (hot-n-gram thresholds at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **pool_composition**: insufficient data — no pool-composition arm in this run (context knobs at their defaults); gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C3]**: insufficient data — window escape 2.080 [2.029, 2.131] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:36%, 2:32%, 3:22%, 4:10%, 5:1%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[C4]**: insufficient data — window escape 2.671 [2.623, 2.720] (min 2.0); pool ECB 4.686 [4.661, 4.710] (floor 4.0); pool ECB share 0.937 [0.932, 0.942] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:12%, 2:32%, 3:35%, 4:17%, 5:3%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **structural_escape[CF]**: insufficient data — window escape 2.868 [2.815, 2.917] (min 2.0); pool ECB 4.527 [4.497, 4.555] (floor 4.0); pool ECB share 0.905 [0.899, 0.911] (доля различных траекторий в пуле; порога нет — справочно к полу выше); window distribution 1:8%, 2:28%, 3:38%, 4:21%, 5:5%; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1552 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 34.9 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression[C0]**: baseline — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[C3]**: fail — 5/18 memes reproduced (28%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693; never reproduced (indices): [0, 2, 3, 4, 5, 7, 9, 10, 11, 12, 14, 15, 17]
- **meme_regression[C4]**: pass — 9/18 memes reproduced (50%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693
- **meme_regression[CF]**: pass — 10/18 memes reproduced (56%); C0 10/18 (56%), tolerance 10%; prompt set bababb4b7693

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
