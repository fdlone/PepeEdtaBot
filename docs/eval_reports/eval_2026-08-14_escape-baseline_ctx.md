# Eval report 2026-08-14 snapshot=escape-baseline prompts=308b7deaea0f seeds=42,1337,2026 mode=ctx

Revision: `3fad11f`. Generations per configuration: 500.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C6b15**: 1500 generations
- **C6b30**: 1500 generations
- **C6b50**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI], `*` = significant (interval excludes 0, doc 05 §4).

| metric | C0 | C6b15 | Δ C6b15 vs C0 | C6b30 | Δ C6b30 vs C0 | C6b50 | Δ C6b50 vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.879 [0.873, 0.887] | 0.884 [0.876, 0.891] | 0.004 [-0.006, 0.014] | 0.887 [0.879, 0.893] | 0.007 [-0.003, 0.017] | 0.884 [0.877, 0.892] | 0.005 [-0.005, 0.015] |
| mean_response_length | 11.051 [10.821, 11.313] | 10.896 [10.659, 11.145] | -0.155 [-0.477, 0.212] | 10.918 [10.674, 11.179] | -0.133 [-0.470, 0.229] | 10.946 [10.699, 11.214] | -0.105 [-0.448, 0.275] |
| unique_token_ratio | 0.985 [0.983, 0.987] | 0.987 [0.985, 0.988] | 0.001 [-0.001, 0.004] | 0.986 [0.984, 0.988] | 0.001 [-0.001, 0.003] | 0.986 [0.984, 0.987] | 0.000 [-0.002, 0.003] |
| exact_context_copy_rate | 0.215 [0.195, 0.236] | 0.204 [0.185, 0.224] | -0.011 [-0.041, 0.017] | 0.202 [0.183, 0.221] | -0.013 [-0.040, 0.015] | 0.201 [0.182, 0.222] | -0.013 [-0.042, 0.015] |
| repetition_rate | 0.003 [0.001, 0.005] | 0.003 [0.001, 0.007] | 0.001 [-0.003, 0.004] | 0.005 [0.001, 0.009] | 0.002 [-0.002, 0.006] | 0.005 [0.001, 0.009] | 0.002 [-0.002, 0.006] |
| cycle_detection_rate | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.003, 0.000] |
| cycle_harm_rate | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] |
| context_affinity | 0.277 [0.262, 0.291] | 0.275 [0.260, 0.290] | -0.002 [-0.023, 0.018] | 0.273 [0.258, 0.289] | -0.004 [-0.025, 0.017] | 0.274 [0.260, 0.290] | -0.003 [-0.024, 0.017] |
| context_affinity_without_copy | 0.205 [0.190, 0.222] | 0.209 [0.195, 0.225] | 0.004 [-0.018, 0.024] | 0.208 [0.192, 0.223] | 0.003 [-0.019, 0.024] | 0.208 [0.192, 0.223] | 0.003 [-0.018, 0.024] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.317 [0.272, 0.368] | 0.291 [0.243, 0.339] | -0.027 [-0.093, 0.040] | 0.288 [0.243, 0.333] | -0.029 [-0.096, 0.037] | 0.293 [0.251, 0.339] | -0.024 [-0.091, 0.043] |
| structural_pool_ecb | 4.455 [4.423, 4.491] | 4.455 [4.422, 4.489] | 0.000 [-0.046, 0.048] | 4.463 [4.430, 4.495] | 0.007 [-0.039, 0.054] | 4.465 [4.432, 4.497] | 0.010 [-0.037, 0.055] |
| structural_window_escape | 2.169 [2.115, 2.227] | 2.198 [2.142, 2.257] | 0.029 [-0.051, 0.103] | 2.211 [2.157, 2.268] | 0.042 [-0.041, 0.119] | 2.210 [2.153, 2.267] | 0.041 [-0.045, 0.118] |

C0: distinct-2 = 0.646 (basis 15077), distinct-3 = 0.798 (basis 13577) — type/token ratios, comparable only at equal basis; latency p50/p95 = 19.8/30.0 ms; cache_hit_rate: 41%; mean normalized entropy: 0.234 (branching 3.19); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b15: distinct-2 = 0.656 (basis 14844), distinct-3 = 0.807 (basis 13344) — type/token ratios, comparable only at equal basis; latency p50/p95 = 20.4/31.2 ms; cache_hit_rate: 47%; mean normalized entropy: 0.249 (branching 3.81); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 7.0%, shift 0.0074; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b30: distinct-2 = 0.647 (basis 14877), distinct-3 = 0.800 (basis 13377) — type/token ratios, comparable only at equal basis; latency p50/p95 = 20.7/31.1 ms; cache_hit_rate: 47%; mean normalized entropy: 0.265 (branching 3.84); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 7.0%, shift 0.0148; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b50: distinct-2 = 0.646 (basis 14919), distinct-3 = 0.798 (basis 13419) — type/token ratios, comparable only at equal basis; latency p50/p95 = 20.8/31.1 ms; cache_hit_rate: 47%; mean normalized entropy: 0.277 (branching 3.79); mean applied temperature: 2.77; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 7.1%, shift 0.0246; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.237 [0.195, 0.283] | 0.003 [0.000, 0.008] | 0.327 [0.295, 0.356] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.237 [0.200, 0.280] | 0.008 [0.000, 0.019] | 0.380 [0.353, 0.407] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.126 [0.095, 0.159] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.379 [0.331, 0.429] | 0.000 [0.000, 0.000] | 0.259 [0.236, 0.284] |
| C6b15 | generic | 375 | 1.000 [1.000, 1.000] | 0.224 [0.184, 0.272] | 0.005 [0.000, 0.013] | 0.318 [0.290, 0.348] |
| C6b15 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.229 [0.189, 0.272] | 0.008 [0.000, 0.019] | 0.384 [0.357, 0.411] |
| C6b15 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.122 [0.091, 0.155] |
| C6b15 | topical | 375 | 1.000 [1.000, 1.000] | 0.360 [0.317, 0.405] | 0.000 [0.000, 0.000] | 0.259 [0.237, 0.282] |
| C6b30 | generic | 375 | 1.000 [1.000, 1.000] | 0.219 [0.179, 0.264] | 0.005 [0.000, 0.013] | 0.327 [0.298, 0.357] |
| C6b30 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.221 [0.181, 0.267] | 0.013 [0.003, 0.027] | 0.373 [0.347, 0.400] |
| C6b30 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.115 [0.084, 0.145] |
| C6b30 | topical | 375 | 1.000 [1.000, 1.000] | 0.365 [0.317, 0.413] | 0.000 [0.000, 0.000] | 0.262 [0.238, 0.285] |
| C6b50 | generic | 375 | 1.000 [1.000, 1.000] | 0.216 [0.176, 0.259] | 0.005 [0.000, 0.013] | 0.324 [0.295, 0.355] |
| C6b50 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.224 [0.184, 0.267] | 0.013 [0.003, 0.027] | 0.377 [0.351, 0.403] |
| C6b50 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.117 [0.085, 0.147] |
| C6b50 | topical | 375 | 1.000 [1.000, 1.000] | 0.363 [0.315, 0.413] | 0.000 [0.000, 0.000] | 0.260 [0.236, 0.285] |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled)
- **phase9_interp[C6b15]**: insufficient data — distinct-2 Δ 0.009 [-0.013, 0.025]; distinct-3 Δ 0.010 [-0.020, 0.029]; copy Δ -0.011 [-0.041, 0.017]; affinity_without_copy Δ 0.004 [-0.018, 0.024]; repetition Δ 0.001 [-0.003, 0.004]; cycles 0.000 [0.000, 0.000]; p95 31.2 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp[C6b30]**: insufficient data — distinct-2 Δ 0.001 [-0.019, 0.021]; distinct-3 Δ 0.003 [-0.023, 0.025]; copy Δ -0.013 [-0.040, 0.015]; affinity_without_copy Δ 0.003 [-0.019, 0.024]; repetition Δ 0.002 [-0.002, 0.006]; cycles 0.000 [0.000, 0.000]; p95 31.1 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp[C6b50]**: insufficient data — distinct-2 Δ 0.000 [-0.019, 0.019]; distinct-3 Δ 0.001 [-0.024, 0.023]; copy Δ -0.013 [-0.042, 0.015]; affinity_without_copy Δ 0.003 [-0.018, 0.024]; repetition Δ 0.002 [-0.002, 0.006]; cycles 0.000 [0.000, 0.000]; p95 31.1 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C0]**: insufficient data — window escape 2.169 [2.115, 2.227] (min 2.0); pool ECB 4.455 [4.423, 4.491] (floor 4.0); window distribution 1:36%, 2:27%, 3:22%, 4:12%, 5:2%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C6b15]**: insufficient data — window escape 2.198 [2.142, 2.257] (min 2.0); pool ECB 4.455 [4.422, 4.489] (floor 4.0); window distribution 1:36%, 2:26%, 3:22%, 4:13%, 5:3%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C6b30]**: insufficient data — window escape 2.211 [2.157, 2.268] (min 2.0); pool ECB 4.463 [4.430, 4.495] (floor 4.0); window distribution 1:36%, 2:27%, 3:22%, 4:13%, 5:3%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **structural_escape[C6b50]**: insufficient data — window escape 2.210 [2.153, 2.267] (min 2.0); pool ECB 4.465 [4.432, 4.497] (floor 4.0); window distribution 1:35%, 2:27%, 3:21%, 4:13%, 5:3%; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.001 [0.000, 0.003] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 933 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 30.0 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression_pass (C0, informational at Phase 0)**: fail — memes never reproduced (indices): [4, 5, 12, 13, 14, 15, 21, 22, 23] (list of 27 memes, prompt set 308b7deaea0f)

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
