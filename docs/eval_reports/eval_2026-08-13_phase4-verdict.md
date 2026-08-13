# Eval report 2026-08-13 snapshot=phase4-verdict prompts=308b7deaea0f seeds=42,1337,2026

Revision: `382c826`. Generations per configuration: 500.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C3**: 1500 generations
- **C4**: 1500 generations
- **CF**: 1500 generations — results shared with C0
- unavailable (feature not implemented yet): C1, C2, C5

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI], `*` = significant (interval excludes 0, doc 05 §4).

| metric | C0 | C3 | Δ C3 vs C0 | C4 | Δ C4 vs C0 | CF | Δ CF vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.879 [0.873, 0.887] | 0.879 [0.873, 0.887] | 0.000 [-0.010, 0.010] | 0.901 [0.895, 0.907] | 0.021 [0.012, 0.030] * | 0.879 [0.873, 0.887] | 0.000 [-0.010, 0.010] |
| mean_response_length | 11.051 [10.821, 11.313] | 11.333 [11.069, 11.595] | 0.282 [-0.068, 0.661] | 11.215 [10.953, 11.489] | 0.163 [-0.177, 0.549] | 11.051 [10.821, 11.313] | 0.000 [-0.340, 0.393] |
| unique_token_ratio | 0.985 [0.983, 0.987] | 0.983 [0.981, 0.985] | -0.002 [-0.005, 0.001] | 0.984 [0.982, 0.986] | -0.001 [-0.004, 0.001] | 0.985 [0.983, 0.987] | 0.000 [-0.003, 0.002] |
| exact_context_copy_rate | 0.215 [0.195, 0.236] | 0.176 [0.159, 0.195] | -0.039 [-0.067, -0.009] * | 0.238 [0.217, 0.259] | 0.023 [-0.007, 0.054] | 0.215 [0.195, 0.236] | 0.000 [-0.029, 0.030] |
| repetition_rate | 0.003 [0.001, 0.005] | 0.008 [0.004, 0.013] | 0.005 [0.001, 0.011] * | 0.002 [0.000, 0.005] | -0.001 [-0.004, 0.003] | 0.003 [0.001, 0.005] | 0.000 [-0.003, 0.003] |
| cycle_detection_rate | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.002] | -0.001 [-0.003, 0.001] | 0.001 [0.000, 0.002] | -0.001 [-0.003, 0.001] | 0.001 [0.000, 0.003] | 0.000 [-0.003, 0.003] |
| cycle_harm_rate | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] |
| context_affinity | 0.277 [0.262, 0.291] | 0.249 [0.235, 0.264] | -0.028 [-0.049, -0.005] * | 0.351 [0.334, 0.367] | 0.074 [0.052, 0.096] * | 0.277 [0.262, 0.291] | 0.000 [-0.020, 0.021] |
| context_affinity_without_copy | 0.205 [0.190, 0.222] | 0.189 [0.174, 0.203] | -0.017 [-0.037, 0.004] | 0.278 [0.261, 0.297] | 0.073 [0.049, 0.097] * | 0.205 [0.190, 0.222] | 0.000 [-0.022, 0.021] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.317 [0.272, 0.368] | 0.309 [0.264, 0.357] | -0.008 [-0.072, 0.064] | 0.352 [0.304, 0.400] | 0.035 [-0.032, 0.101] | 0.317 [0.272, 0.368] | 0.000 [-0.064, 0.069] |

C0: distinct-2 = 0.646 (basis 15077), distinct-3 = 0.798 (basis 13577) — type/token ratios, comparable only at equal basis; latency p50/p95 = 25.0/40.7 ms; cache_hit_rate: 41%; mean normalized entropy: 0.234 (branching 3.19); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C3: distinct-2 = 0.637 (basis 15500), distinct-3 = 0.788 (basis 14000) — type/token ratios, comparable only at equal basis; latency p50/p95 = 26.4/41.6 ms; cache_hit_rate: 41%; mean normalized entropy: 0.234 (branching 3.19); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C4: distinct-2 = 0.628 (basis 15322), distinct-3 = 0.776 (basis 13822) — type/token ratios, comparable only at equal basis; latency p50/p95 = 53.7/119.1 ms; cache_hit_rate: 38%; mean normalized entropy: 0.234 (branching 3.19); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

CF: distinct-2 = 0.646 (basis 15077), distinct-3 = 0.798 (basis 13577) — type/token ratios, comparable only at equal basis; latency p50/p95 = 25.0/40.7 ms; cache_hit_rate: insufficient data; mean normalized entropy: insufficient data; mean applied temperature: insufficient data; temporal blend: insufficient data; shadow order-4 share: insufficient data; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.237 [0.195, 0.283] | 0.003 [0.000, 0.008] | 0.327 [0.295, 0.356] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.237 [0.200, 0.280] | 0.008 [0.000, 0.019] | 0.380 [0.353, 0.407] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.126 [0.095, 0.159] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.379 [0.331, 0.429] | 0.000 [0.000, 0.000] | 0.259 [0.236, 0.284] |
| C3 | generic | 375 | 1.000 [1.000, 1.000] | 0.200 [0.163, 0.243] | 0.005 [0.000, 0.013] | 0.285 [0.256, 0.315] |
| C3 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.187 [0.147, 0.224] | 0.019 [0.008, 0.032] | 0.351 [0.325, 0.380] |
| C3 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.003 [0.000, 0.008] | 0.122 [0.091, 0.155] |
| C3 | topical | 375 | 1.000 [1.000, 1.000] | 0.309 [0.267, 0.357] | 0.005 [0.000, 0.013] | 0.225 [0.202, 0.250] |
| C4 | generic | 375 | 1.000 [1.000, 1.000] | 0.264 [0.219, 0.309] | 0.003 [0.000, 0.008] | 0.376 [0.344, 0.408] |
| C4 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.283 [0.240, 0.328] | 0.003 [0.000, 0.008] | 0.444 [0.414, 0.473] |
| C4 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.003 [0.000, 0.008] | 0.284 [0.248, 0.329] |
| C4 | topical | 375 | 1.000 [1.000, 1.000] | 0.400 [0.352, 0.451] | 0.000 [0.000, 0.000] | 0.293 [0.267, 0.321] |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes[C3]**: fail — manual: meme 8/20 genuine (40%, bar 70%) vs frequency control 2/20 (10%), Δ +30%; decoys 1/5 (20%); 3 rater(s), agreement 0.17; copy Δ -0.039 [-0.067, -0.009] *; affinity_without_copy Δ -0.017 [-0.037, 0.004]; p95 41.6 ms (budget 150) — genuine share below the bar
- **phase5_promotion[C4]**: insufficient data — seeded present 82% (bar 30%), win|present 20% (bar 40%); affinity_without_copy Δ 0.073 [0.049, 0.097] *; p95 119.1 ms (budget 150) — missing: a protocol run over prod-accumulated df (df here is window-approximated, design D4)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.003] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 933 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 40.7 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression_pass (C0, informational at Phase 0)**: fail — memes never reproduced (indices): [4, 5, 12, 13, 14, 15, 21, 22, 23] (list of 27 memes, prompt set 308b7deaea0f)

## Manual eval summary

Raters: 3; inter-rater agreement: 0.17.

| source | rated | genuine | share |
|---|---|---|---|
| association ranking | 20 | 8 | 40% |
| frequency control | 20 | 2 | 10% |
| decoys | 5 | 1 | 20% |

Ranking version rated: `b81b634`.

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
