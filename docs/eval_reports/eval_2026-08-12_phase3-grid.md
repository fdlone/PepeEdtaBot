# Eval report 2026-08-12 snapshot=phase3-grid prompts=308b7deaea0f seeds=42,1337,2026

Revision: `4c3a532`. Generations per configuration: 500.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).
- reconstructed temporal snapshot: 1000 replayed messages spanning 2026-06-03..2026-07-12 (39 days); fresh slice = tokens first seen in the last 14 days = 1310 of 3137 vocabulary tokens. Built from the retention window, NOT from live accumulation: its C0 is not the frozen baseline and deltas are only valid within this snapshot.

## Config matrix

- **C0**: 1500 generations
- **C2a03_log**: 1500 generations
- **C2a03_pow50**: 1500 generations
- **C2a03_pow75**: 1500 generations
- **C2a05_log**: 1500 generations
- **C2a05_pow50**: 1500 generations
- **C2a05_pow75**: 1500 generations
- **C2a07_log**: 1500 generations
- **C2a07_pow50**: 1500 generations
- **C2a07_pow75**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI], `*` = significant (interval excludes 0, doc 05 §4).

| metric | C0 | C2a03_log | Δ C2a03_log vs C0 | C2a03_pow50 | Δ C2a03_pow50 vs C0 | C2a03_pow75 | Δ C2a03_pow75 vs C0 | C2a05_log | Δ C2a05_log vs C0 | C2a05_pow50 | Δ C2a05_pow50 vs C0 | C2a05_pow75 | Δ C2a05_pow75 vs C0 | C2a07_log | Δ C2a07_log vs C0 | C2a07_pow50 | Δ C2a07_pow50 vs C0 | C2a07_pow75 | Δ C2a07_pow75 vs C0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.845 [0.837, 0.853] | 0.846 [0.838, 0.854] | 0.001 [-0.008, 0.012] | 0.846 [0.838, 0.854] | 0.001 [-0.008, 0.013] | 0.846 [0.838, 0.853] | 0.001 [-0.009, 0.012] | 0.847 [0.839, 0.855] | 0.002 [-0.008, 0.013] | 0.847 [0.839, 0.855] | 0.002 [-0.008, 0.013] | 0.847 [0.839, 0.854] | 0.002 [-0.008, 0.013] | 0.847 [0.839, 0.855] | 0.002 [-0.007, 0.013] | 0.847 [0.839, 0.855] | 0.002 [-0.007, 0.013] | 0.847 [0.839, 0.855] | 0.002 [-0.007, 0.013] |
| mean_response_length | 11.429 [11.201, 11.695] | 11.359 [11.121, 11.628] | -0.070 [-0.417, 0.260] | 11.364 [11.125, 11.636] | -0.065 [-0.409, 0.265] | 11.351 [11.113, 11.617] | -0.077 [-0.421, 0.253] | 11.311 [11.072, 11.573] | -0.117 [-0.469, 0.225] | 11.305 [11.075, 11.561] | -0.124 [-0.475, 0.214] | 11.318 [11.081, 11.580] | -0.111 [-0.458, 0.228] | 11.264 [11.035, 11.517] | -0.165 [-0.504, 0.159] | 11.265 [11.036, 11.521] | -0.163 [-0.501, 0.162] | 11.251 [11.028, 11.498] | -0.178 [-0.524, 0.144] |
| unique_token_ratio | 0.983 [0.981, 0.985] | 0.984 [0.982, 0.986] | 0.001 [-0.002, 0.004] | 0.984 [0.982, 0.986] | 0.001 [-0.002, 0.004] | 0.984 [0.982, 0.986] | 0.001 [-0.002, 0.004] | 0.984 [0.982, 0.986] | 0.001 [-0.001, 0.004] | 0.984 [0.982, 0.986] | 0.001 [-0.001, 0.004] | 0.984 [0.982, 0.986] | 0.001 [-0.001, 0.004] | 0.985 [0.983, 0.987] | 0.002 [-0.001, 0.005] | 0.985 [0.983, 0.987] | 0.002 [-0.001, 0.005] | 0.985 [0.983, 0.987] | 0.002 [-0.001, 0.005] |
| exact_context_copy_rate | 0.284 [0.261, 0.307] | 0.281 [0.259, 0.304] | -0.003 [-0.035, 0.030] | 0.281 [0.259, 0.304] | -0.003 [-0.035, 0.030] | 0.281 [0.259, 0.304] | -0.003 [-0.035, 0.030] | 0.283 [0.260, 0.307] | -0.001 [-0.033, 0.032] | 0.283 [0.260, 0.307] | -0.001 [-0.033, 0.032] | 0.283 [0.260, 0.307] | -0.001 [-0.033, 0.032] | 0.282 [0.261, 0.306] | -0.002 [-0.034, 0.031] | 0.282 [0.261, 0.306] | -0.002 [-0.034, 0.031] | 0.282 [0.260, 0.307] | -0.002 [-0.033, 0.031] |
| repetition_rate | 0.005 [0.002, 0.009] | 0.005 [0.001, 0.008] | -0.001 [-0.005, 0.004] | 0.005 [0.001, 0.008] | -0.001 [-0.005, 0.004] | 0.005 [0.001, 0.008] | -0.001 [-0.005, 0.004] | 0.004 [0.001, 0.007] | -0.001 [-0.006, 0.003] | 0.004 [0.001, 0.007] | -0.001 [-0.006, 0.003] | 0.004 [0.001, 0.007] | -0.001 [-0.006, 0.003] | 0.004 [0.001, 0.007] | -0.001 [-0.006, 0.003] | 0.004 [0.001, 0.007] | -0.001 [-0.006, 0.003] | 0.004 [0.001, 0.007] | -0.001 [-0.006, 0.003] |
| cycle_detection_rate | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] |
| cycle_harm_rate | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] | 0.000 [0.000, 0.000] | -0.001 [-0.002, 0.000] |
| context_affinity | 0.319 [0.302, 0.336] | 0.314 [0.297, 0.331] | -0.004 [-0.027, 0.018] | 0.315 [0.297, 0.331] | -0.004 [-0.027, 0.019] | 0.314 [0.297, 0.330] | -0.005 [-0.028, 0.018] | 0.315 [0.298, 0.331] | -0.004 [-0.027, 0.019] | 0.315 [0.298, 0.331] | -0.004 [-0.027, 0.019] | 0.314 [0.297, 0.331] | -0.004 [-0.027, 0.019] | 0.316 [0.299, 0.332] | -0.003 [-0.027, 0.020] | 0.316 [0.299, 0.332] | -0.003 [-0.026, 0.020] | 0.316 [0.299, 0.332] | -0.003 [-0.027, 0.020] |
| context_affinity_without_copy | 0.230 [0.212, 0.248] | 0.225 [0.206, 0.244] | -0.005 [-0.030, 0.020] | 0.226 [0.206, 0.244] | -0.005 [-0.030, 0.020] | 0.225 [0.208, 0.243] | -0.005 [-0.031, 0.021] | 0.225 [0.208, 0.244] | -0.005 [-0.031, 0.022] | 0.226 [0.208, 0.245] | -0.004 [-0.030, 0.023] | 0.225 [0.208, 0.244] | -0.005 [-0.030, 0.021] | 0.228 [0.210, 0.247] | -0.002 [-0.029, 0.024] | 0.228 [0.210, 0.247] | -0.002 [-0.029, 0.024] | 0.228 [0.210, 0.246] | -0.002 [-0.030, 0.023] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | 0.166 [0.158, 0.176] | 0.167 [0.158, 0.175] | 0.000 [-0.012, 0.013] | 0.167 [0.158, 0.176] | 0.000 [-0.012, 0.012] | 0.167 [0.158, 0.176] | 0.000 [-0.012, 0.013] | 0.168 [0.160, 0.178] | 0.002 [-0.010, 0.014] | 0.168 [0.159, 0.178] | 0.002 [-0.010, 0.014] | 0.168 [0.160, 0.178] | 0.002 [-0.010, 0.014] | 0.170 [0.161, 0.180] | 0.004 [-0.009, 0.016] | 0.170 [0.161, 0.179] | 0.003 [-0.009, 0.015] | 0.170 [0.161, 0.179] | 0.003 [-0.009, 0.015] |
| historical_meme_rate | 0.376 [0.328, 0.424] | 0.368 [0.323, 0.416] | -0.008 [-0.072, 0.059] | 0.368 [0.323, 0.416] | -0.008 [-0.072, 0.059] | 0.365 [0.320, 0.413] | -0.011 [-0.075, 0.059] | 0.376 [0.331, 0.424] | 0.000 [-0.067, 0.072] | 0.376 [0.331, 0.424] | 0.000 [-0.067, 0.072] | 0.376 [0.328, 0.424] | 0.000 [-0.064, 0.072] | 0.373 [0.325, 0.421] | -0.003 [-0.067, 0.067] | 0.373 [0.325, 0.421] | -0.003 [-0.067, 0.067] | 0.373 [0.325, 0.421] | -0.003 [-0.067, 0.069] |

C0: distinct-2 = 0.407 (basis 15643), distinct-3 = 0.568 (basis 14143) — type/token ratios, comparable only at equal basis; latency p50/p95 = 11.7/21.4 ms; cache_hit_rate: 51%; mean normalized entropy: 0.117 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a03_log: distinct-2 = 0.408 (basis 15538), distinct-3 = 0.571 (basis 14038) — type/token ratios, comparable only at equal basis; latency p50/p95 = 12.5/22.1 ms; cache_hit_rate: 51%; mean normalized entropy: 0.114 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 11.2%, shift 0.0117; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a03_pow50: distinct-2 = 0.408 (basis 15546), distinct-3 = 0.571 (basis 14046) — type/token ratios, comparable only at equal basis; latency p50/p95 = 12.3/22.1 ms; cache_hit_rate: 51%; mean normalized entropy: 0.114 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 11.2%, shift 0.0117; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a03_pow75: distinct-2 = 0.408 (basis 15527), distinct-3 = 0.571 (basis 14027) — type/token ratios, comparable only at equal basis; latency p50/p95 = 13.1/24.6 ms; cache_hit_rate: 51%; mean normalized entropy: 0.114 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 11.2%, shift 0.0116; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a05_log: distinct-2 = 0.407 (basis 15467), distinct-3 = 0.571 (basis 13967) — type/token ratios, comparable only at equal basis; latency p50/p95 = 12.4/23.6 ms; cache_hit_rate: 51%; mean normalized entropy: 0.108 (branching 1.45); mean applied temperature: 2.77; temporal blend: coverage 11.2%, shift 0.0194; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a05_pow50: distinct-2 = 0.407 (basis 15457), distinct-3 = 0.570 (basis 13957) — type/token ratios, comparable only at equal basis; latency p50/p95 = 12.9/25.6 ms; cache_hit_rate: 51%; mean normalized entropy: 0.108 (branching 1.45); mean applied temperature: 2.77; temporal blend: coverage 11.2%, shift 0.0195; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a05_pow75: distinct-2 = 0.407 (basis 15477), distinct-3 = 0.571 (basis 13977) — type/token ratios, comparable only at equal basis; latency p50/p95 = 13.6/24.8 ms; cache_hit_rate: 51%; mean normalized entropy: 0.108 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 11.2%, shift 0.0194; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a07_log: distinct-2 = 0.407 (basis 15396), distinct-3 = 0.572 (basis 13896) — type/token ratios, comparable only at equal basis; latency p50/p95 = 12.4/21.6 ms; cache_hit_rate: 51%; mean normalized entropy: 0.100 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 11.3%, shift 0.0274; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a07_pow50: distinct-2 = 0.407 (basis 15398), distinct-3 = 0.573 (basis 13898) — type/token ratios, comparable only at equal basis; latency p50/p95 = 12.7/22.6 ms; cache_hit_rate: 51%; mean normalized entropy: 0.100 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 11.3%, shift 0.0275; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C2a07_pow75: distinct-2 = 0.407 (basis 15376), distinct-3 = 0.573 (basis 13876) — type/token ratios, comparable only at equal basis; latency p50/p95 = 13.7/26.8 ms; cache_hit_rate: 51%; mean normalized entropy: 0.100 (branching 1.45); mean applied temperature: 2.78; temporal blend: coverage 11.3%, shift 0.0274; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.312 [0.264, 0.357] | 0.008 [0.000, 0.019] | 0.378 [0.346, 0.408] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.296 [0.251, 0.344] | 0.008 [0.000, 0.019] | 0.419 [0.389, 0.451] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.059 [0.037, 0.083] | 0.003 [0.000, 0.008] | 0.170 [0.137, 0.205] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.469 [0.419, 0.520] | 0.003 [0.000, 0.008] | 0.293 [0.266, 0.320] |
| C2a03_log | generic | 375 | 1.000 [1.000, 1.000] | 0.304 [0.259, 0.347] | 0.005 [0.000, 0.013] | 0.378 [0.346, 0.409] |
| C2a03_log | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.301 [0.256, 0.349] | 0.008 [0.000, 0.019] | 0.416 [0.385, 0.447] |
| C2a03_log | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.061 [0.040, 0.085] | 0.003 [0.000, 0.008] | 0.159 [0.126, 0.193] |
| C2a03_log | topical | 375 | 1.000 [1.000, 1.000] | 0.456 [0.405, 0.507] | 0.003 [0.000, 0.008] | 0.289 [0.262, 0.318] |
| C2a03_pow50 | generic | 375 | 1.000 [1.000, 1.000] | 0.304 [0.259, 0.347] | 0.005 [0.000, 0.013] | 0.378 [0.346, 0.409] |
| C2a03_pow50 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.301 [0.256, 0.349] | 0.008 [0.000, 0.019] | 0.416 [0.385, 0.447] |
| C2a03_pow50 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.061 [0.040, 0.085] | 0.003 [0.000, 0.008] | 0.159 [0.126, 0.193] |
| C2a03_pow50 | topical | 375 | 1.000 [1.000, 1.000] | 0.456 [0.405, 0.507] | 0.003 [0.000, 0.008] | 0.290 [0.263, 0.318] |
| C2a03_pow75 | generic | 375 | 1.000 [1.000, 1.000] | 0.304 [0.259, 0.347] | 0.005 [0.000, 0.013] | 0.379 [0.347, 0.410] |
| C2a03_pow75 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.301 [0.256, 0.349] | 0.008 [0.000, 0.019] | 0.414 [0.382, 0.445] |
| C2a03_pow75 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.061 [0.040, 0.085] | 0.003 [0.000, 0.008] | 0.159 [0.126, 0.193] |
| C2a03_pow75 | topical | 375 | 1.000 [1.000, 1.000] | 0.456 [0.405, 0.507] | 0.003 [0.000, 0.008] | 0.289 [0.262, 0.317] |
| C2a05_log | generic | 375 | 1.000 [1.000, 1.000] | 0.304 [0.259, 0.349] | 0.005 [0.000, 0.013] | 0.378 [0.345, 0.408] |
| C2a05_log | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.304 [0.256, 0.352] | 0.008 [0.000, 0.019] | 0.419 [0.388, 0.451] |
| C2a05_log | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.064 [0.040, 0.088] | 0.000 [0.000, 0.000] | 0.156 [0.121, 0.189] |
| C2a05_log | topical | 375 | 1.000 [1.000, 1.000] | 0.459 [0.411, 0.509] | 0.003 [0.000, 0.008] | 0.289 [0.261, 0.318] |
| C2a05_pow50 | generic | 375 | 1.000 [1.000, 1.000] | 0.304 [0.259, 0.349] | 0.005 [0.000, 0.013] | 0.377 [0.344, 0.408] |
| C2a05_pow50 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.304 [0.256, 0.352] | 0.008 [0.000, 0.019] | 0.419 [0.388, 0.451] |
| C2a05_pow50 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.064 [0.040, 0.088] | 0.000 [0.000, 0.000] | 0.159 [0.124, 0.193] |
| C2a05_pow50 | topical | 375 | 1.000 [1.000, 1.000] | 0.459 [0.411, 0.509] | 0.003 [0.000, 0.008] | 0.289 [0.261, 0.318] |
| C2a05_pow75 | generic | 375 | 1.000 [1.000, 1.000] | 0.304 [0.259, 0.349] | 0.005 [0.000, 0.013] | 0.378 [0.345, 0.408] |
| C2a05_pow75 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.304 [0.256, 0.352] | 0.008 [0.000, 0.019] | 0.418 [0.387, 0.451] |
| C2a05_pow75 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.064 [0.040, 0.088] | 0.000 [0.000, 0.000] | 0.156 [0.121, 0.189] |
| C2a05_pow75 | topical | 375 | 1.000 [1.000, 1.000] | 0.459 [0.411, 0.509] | 0.003 [0.000, 0.008] | 0.289 [0.262, 0.318] |
| C2a07_log | generic | 375 | 1.000 [1.000, 1.000] | 0.307 [0.261, 0.352] | 0.005 [0.000, 0.013] | 0.385 [0.352, 0.418] |
| C2a07_log | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.309 [0.261, 0.357] | 0.008 [0.000, 0.019] | 0.424 [0.394, 0.456] |
| C2a07_log | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.067 [0.043, 0.091] | 0.000 [0.000, 0.000] | 0.154 [0.121, 0.187] |
| C2a07_log | topical | 375 | 1.000 [1.000, 1.000] | 0.445 [0.397, 0.499] | 0.003 [0.000, 0.008] | 0.284 [0.256, 0.310] |
| C2a07_pow50 | generic | 375 | 1.000 [1.000, 1.000] | 0.307 [0.261, 0.352] | 0.005 [0.000, 0.013] | 0.385 [0.352, 0.418] |
| C2a07_pow50 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.309 [0.261, 0.357] | 0.008 [0.000, 0.019] | 0.424 [0.394, 0.456] |
| C2a07_pow50 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.067 [0.043, 0.091] | 0.000 [0.000, 0.000] | 0.154 [0.121, 0.187] |
| C2a07_pow50 | topical | 375 | 1.000 [1.000, 1.000] | 0.445 [0.397, 0.499] | 0.003 [0.000, 0.008] | 0.284 [0.257, 0.311] |
| C2a07_pow75 | generic | 375 | 1.000 [1.000, 1.000] | 0.307 [0.261, 0.352] | 0.005 [0.000, 0.013] | 0.385 [0.352, 0.417] |
| C2a07_pow75 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.307 [0.261, 0.357] | 0.008 [0.000, 0.019] | 0.421 [0.391, 0.453] |
| C2a07_pow75 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.067 [0.043, 0.091] | 0.000 [0.000, 0.000] | 0.154 [0.121, 0.187] |
| C2a07_pow75 | topical | 375 | 1.000 [1.000, 1.000] | 0.448 [0.400, 0.501] | 0.003 [0.000, 0.008] | 0.285 [0.258, 0.313] |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal[C2a03_log]**: fail — freshness Δ 0.000 [-0.012, 0.013]; historical_meme Δ -0.008 [-0.072, 0.059]; copy Δ -0.003 [-0.035, 0.030]; affinity_without_copy Δ -0.005 [-0.030, 0.020]; p95 22.1 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a03_pow50]**: fail — freshness Δ 0.000 [-0.012, 0.012]; historical_meme Δ -0.008 [-0.072, 0.059]; copy Δ -0.003 [-0.035, 0.030]; affinity_without_copy Δ -0.005 [-0.030, 0.020]; p95 22.1 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a03_pow75]**: fail — freshness Δ 0.000 [-0.012, 0.013]; historical_meme Δ -0.011 [-0.075, 0.059]; copy Δ -0.003 [-0.035, 0.030]; affinity_without_copy Δ -0.005 [-0.031, 0.021]; p95 24.6 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a05_log]**: fail — freshness Δ 0.002 [-0.010, 0.014]; historical_meme Δ 0.000 [-0.067, 0.072]; copy Δ -0.001 [-0.033, 0.032]; affinity_without_copy Δ -0.005 [-0.031, 0.022]; p95 23.6 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a05_pow50]**: fail — freshness Δ 0.002 [-0.010, 0.014]; historical_meme Δ 0.000 [-0.067, 0.072]; copy Δ -0.001 [-0.033, 0.032]; affinity_without_copy Δ -0.004 [-0.030, 0.023]; p95 25.6 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a05_pow75]**: fail — freshness Δ 0.002 [-0.010, 0.014]; historical_meme Δ 0.000 [-0.064, 0.072]; copy Δ -0.001 [-0.033, 0.032]; affinity_without_copy Δ -0.005 [-0.030, 0.021]; p95 24.8 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a07_log]**: fail — freshness Δ 0.004 [-0.009, 0.016]; historical_meme Δ -0.003 [-0.067, 0.067]; copy Δ -0.002 [-0.034, 0.031]; affinity_without_copy Δ -0.002 [-0.029, 0.024]; p95 21.6 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a07_pow50]**: fail — freshness Δ 0.003 [-0.009, 0.015]; historical_meme Δ -0.003 [-0.067, 0.067]; copy Δ -0.002 [-0.034, 0.031]; affinity_without_copy Δ -0.002 [-0.029, 0.024]; p95 22.6 ms (budget 150) — freshness did not rise significantly
- **phase3_temporal[C2a07_pow75]**: fail — freshness Δ 0.003 [-0.009, 0.015]; historical_meme Δ -0.003 [-0.067, 0.069]; copy Δ -0.002 [-0.033, 0.031]; affinity_without_copy Δ -0.002 [-0.030, 0.023]; p95 26.8 ms (budget 150) — freshness did not rise significantly
- **phase5_promotion**: insufficient data — seeded generation does not exist before Phase 5
- **phase6_anticycle**: insufficient data — observed cycle_detection_rate=0.001 [0.000, 0.002] (threshold 0.05); cycle_harm_rate has only its automatic component until a manual round (doc 05 §5) is conducted
- **phase7_order4**: insufficient data — shadow data: 364 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 21.4 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression_pass (C0, informational at Phase 0)**: fail — memes never reproduced (indices): [13, 14, 15] (list of 27 memes, prompt set 308b7deaea0f)

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
