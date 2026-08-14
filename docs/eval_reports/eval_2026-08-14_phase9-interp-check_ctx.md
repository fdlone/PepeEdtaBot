# Eval report 2026-08-14 snapshot=phase9-interp-check prompts=308b7deaea0f seeds=42 mode=ctx

Revision: `c475e0a`. Generations per configuration: 100.
Context mode: **ctx** — the prompt is supplied to the generator as context.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 100 generations
- **C6b15**: 100 generations
- **C6b30**: 100 generations
- **C6b50**: 100 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI], `*` = significant (interval excludes 0, doc 05 §4).

| metric | C0 | C6b15 | Δ C6b15 vs C0 | C6b30 | Δ C6b30 vs C0 | C6b50 | Δ C6b50 vs C0 |
|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.896 [0.871, 0.924] | 0.892 [0.865, 0.919] | -0.004 [-0.037, 0.033] | 0.891 [0.866, 0.916] | -0.005 [-0.038, 0.032] | 0.892 [0.865, 0.918] | -0.005 [-0.036, 0.032] |
| mean_response_length | 11.260 [10.260, 12.340] | 11.170 [10.200, 12.100] | -0.090 [-1.510, 1.290] | 10.840 [9.880, 11.770] | -0.420 [-1.820, 0.940] | 11.070 [10.130, 12.040] | -0.190 [-1.610, 1.240] |
| unique_token_ratio | 0.985 [0.978, 0.992] | 0.985 [0.977, 0.991] | -0.001 [-0.011, 0.010] | 0.986 [0.979, 0.993] | 0.001 [-0.008, 0.011] | 0.985 [0.978, 0.991] | -0.001 [-0.011, 0.009] |
| exact_context_copy_rate | 0.210 [0.130, 0.290] | 0.210 [0.130, 0.290] | 0.000 [-0.110, 0.110] | 0.180 [0.110, 0.260] | -0.030 [-0.140, 0.080] | 0.200 [0.120, 0.280] | -0.010 [-0.120, 0.100] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.264 [0.210, 0.321] | 0.291 [0.241, 0.347] | 0.027 [-0.056, 0.107] | 0.281 [0.230, 0.337] | 0.017 [-0.066, 0.098] | 0.290 [0.239, 0.348] | 0.026 [-0.056, 0.109] |
| context_affinity_without_copy | 0.200 [0.143, 0.259] | 0.236 [0.170, 0.302] | 0.036 [-0.049, 0.119] | 0.223 [0.164, 0.288] | 0.023 [-0.060, 0.111] | 0.230 [0.170, 0.292] | 0.029 [-0.058, 0.116] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.280 [0.120, 0.440] | 0.320 [0.160, 0.520] | 0.040 [-0.240, 0.320] | 0.320 [0.160, 0.520] | 0.040 [-0.240, 0.320] | 0.280 [0.120, 0.480] | 0.000 [-0.240, 0.280] |

C0: distinct-2 = 0.924 (basis 1026), distinct-3 = 0.973 (basis 926) — type/token ratios, comparable only at equal basis; latency p50/p95 = 21.4/36.4 ms; cache_hit_rate: 38%; mean normalized entropy: 0.240 (branching 3.28); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b15: distinct-2 = 0.928 (basis 1017), distinct-3 = 0.974 (basis 917) — type/token ratios, comparable only at equal basis; latency p50/p95 = 22.7/33.9 ms; cache_hit_rate: 45%; mean normalized entropy: 0.256 (branching 3.91); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 7.6%, shift 0.0078; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b30: distinct-2 = 0.927 (basis 984), distinct-3 = 0.980 (basis 884) — type/token ratios, comparable only at equal basis; latency p50/p95 = 21.6/35.4 ms; cache_hit_rate: 46%; mean normalized entropy: 0.276 (branching 3.97); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 7.6%, shift 0.0155; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b50: distinct-2 = 0.930 (basis 1007), distinct-3 = 0.978 (basis 907) — type/token ratios, comparable only at equal basis; latency p50/p95 = 21.1/34.9 ms; cache_hit_rate: 46%; mean normalized entropy: 0.292 (branching 3.91); mean applied temperature: 2.76; temporal blend: coverage 0.0%, shift 0.0000; order interpolation: coverage 7.6%, shift 0.0260; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 25 | 1.000 [1.000, 1.000] | 0.240 [0.080, 0.400] | 0.000 [0.000, 0.000] | 0.317 [0.218, 0.427] |
| C0 | meme-bait | 25 | 1.000 [1.000, 1.000] | 0.320 [0.160, 0.520] | 0.000 [0.000, 0.000] | 0.417 [0.327, 0.510] |
| C0 | short-degenerate | 25 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.149 [0.024, 0.298] |
| C0 | topical | 25 | 1.000 [1.000, 1.000] | 0.280 [0.120, 0.480] | 0.000 [0.000, 0.000] | 0.170 [0.104, 0.245] |
| C6b15 | generic | 25 | 1.000 [1.000, 1.000] | 0.160 [0.040, 0.320] | 0.000 [0.000, 0.000] | 0.334 [0.242, 0.434] |
| C6b15 | meme-bait | 25 | 1.000 [1.000, 1.000] | 0.360 [0.200, 0.520] | 0.000 [0.000, 0.000] | 0.474 [0.377, 0.575] |
| C6b15 | short-degenerate | 25 | 1.000 [1.000, 1.000] | 0.040 [0.000, 0.120] | 0.000 [0.000, 0.000] | 0.186 [0.055, 0.334] |
| C6b15 | topical | 25 | 1.000 [1.000, 1.000] | 0.280 [0.120, 0.480] | 0.000 [0.000, 0.000] | 0.166 [0.101, 0.234] |
| C6b30 | generic | 25 | 1.000 [1.000, 1.000] | 0.160 [0.040, 0.320] | 0.000 [0.000, 0.000] | 0.338 [0.246, 0.437] |
| C6b30 | meme-bait | 25 | 1.000 [1.000, 1.000] | 0.360 [0.200, 0.520] | 0.000 [0.000, 0.000] | 0.459 [0.353, 0.569] |
| C6b30 | short-degenerate | 25 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.186 [0.055, 0.334] |
| C6b30 | topical | 25 | 1.000 [1.000, 1.000] | 0.200 [0.040, 0.360] | 0.000 [0.000, 0.000] | 0.139 [0.077, 0.208] |
| C6b50 | generic | 25 | 1.000 [1.000, 1.000] | 0.160 [0.040, 0.320] | 0.000 [0.000, 0.000] | 0.336 [0.244, 0.435] |
| C6b50 | meme-bait | 25 | 1.000 [1.000, 1.000] | 0.320 [0.160, 0.520] | 0.000 [0.000, 0.000] | 0.460 [0.344, 0.578] |
| C6b50 | short-degenerate | 25 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.186 [0.055, 0.334] |
| C6b50 | topical | 25 | 1.000 [1.000, 1.000] | 0.320 [0.160, 0.520] | 0.000 [0.000, 0.000] | 0.174 [0.102, 0.254] |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled)
- **phase9_interp[C6b15]**: insufficient data — distinct-2 Δ 0.004 [-0.100, 0.097]; distinct-3 Δ 0.001 [-0.104, 0.110]; copy Δ 0.000 [-0.110, 0.110]; affinity_without_copy Δ 0.036 [-0.049, 0.119]; repetition Δ 0.000 [0.000, 0.000]; cycles 0.000 [0.000, 0.000]; p95 33.9 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; affinity without copies could be down to -0.049; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp[C6b30]**: insufficient data — distinct-2 Δ 0.003 [-0.099, 0.096]; distinct-3 Δ 0.007 [-0.099, 0.108]; copy Δ -0.030 [-0.140, 0.080]; affinity_without_copy Δ 0.023 [-0.060, 0.111]; repetition Δ 0.000 [0.000, 0.000]; cycles 0.000 [0.000, 0.000]; p95 35.4 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; affinity without copies could be down to -0.060; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase9_interp[C6b50]**: insufficient data — distinct-2 Δ 0.007 [-0.096, 0.099]; distinct-3 Δ 0.005 [-0.100, 0.113]; copy Δ -0.010 [-0.120, 0.100]; affinity_without_copy Δ 0.029 [-0.058, 0.116]; repetition Δ 0.000 [0.000, 0.000]; cycles 0.000 [0.000, 0.000]; p95 34.9 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; affinity without copies could be down to -0.058; gate requires both context modes, this run measured ctx only (noctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: insufficient data — shadow data: 46 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 36.4 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression_pass (C0, informational at Phase 0)**: fail — memes never reproduced (indices): [1, 2, 3, 4, 5, 6, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26] (list of 27 memes, prompt set 308b7deaea0f)

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
