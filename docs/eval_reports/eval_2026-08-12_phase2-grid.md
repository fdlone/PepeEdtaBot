# Eval report 2026-08-12 snapshot=phase2-grid prompts=308b7deaea0f seeds=42,1337,2026

Revision: `d3dc7b6`. Generations per configuration: 500.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **C1branch_15**: 1500 generations
- **C1branch_25**: 1500 generations
- **C1gain_neg03**: 1500 generations
- **C1gain_neg06**: 1500 generations
- **C1gain_pos03**: 1500 generations
- **C1gain_pos06**: 1500 generations

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI], `*` = significant (interval excludes 0, doc 05 §4).

| metric | C0 | C1branch_15 | Δ C1branch_15 vs C0 | C1branch_25 | Δ C1branch_25 vs C0 | C1gain_neg03 | Δ C1gain_neg03 vs C0 | C1gain_neg06 | Δ C1gain_neg06 vs C0 | C1gain_pos03 | Δ C1gain_pos03 vs C0 | C1gain_pos06 | Δ C1gain_pos06 vs C0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.879 [0.872, 0.887] | 0.882 [0.873, 0.890] | 0.002 [-0.010, 0.013] | 0.880 [0.870, 0.889] | 0.001 [-0.012, 0.012] | 0.878 [0.871, 0.885] | -0.001 [-0.012, 0.009] | 0.878 [0.871, 0.885] | -0.001 [-0.011, 0.009] | 0.879 [0.872, 0.886] | -0.001 [-0.011, 0.009] | 0.878 [0.871, 0.885] | -0.001 [-0.012, 0.008] |
| mean_response_length | 10.822 [10.599, 11.092] | 10.848 [10.611, 11.102] | 0.026 [-0.342, 0.403] | 10.975 [10.731, 11.231] | 0.153 [-0.200, 0.533] | 10.807 [10.593, 11.077] | -0.015 [-0.387, 0.363] | 10.889 [10.667, 11.143] | 0.067 [-0.290, 0.449] | 10.841 [10.625, 11.107] | 0.019 [-0.331, 0.389] | 10.848 [10.623, 11.125] | 0.026 [-0.331, 0.396] |
| unique_token_ratio | 0.986 [0.984, 0.988] | 0.986 [0.984, 0.987] | -0.000 [-0.003, 0.002] | 0.984 [0.982, 0.986] | -0.002 [-0.005, 0.001] | 0.986 [0.984, 0.988] | -0.000 [-0.003, 0.003] | 0.986 [0.984, 0.988] | 0.000 [-0.002, 0.003] | 0.986 [0.985, 0.988] | 0.000 [-0.002, 0.003] | 0.986 [0.984, 0.988] | 0.000 [-0.002, 0.003] |
| exact_context_copy_rate | 0.222 [0.203, 0.243] | 0.199 [0.180, 0.219] | -0.023 [-0.055, 0.005] | 0.183 [0.165, 0.203] | -0.039 [-0.069, -0.011] * | 0.221 [0.201, 0.241] | -0.001 [-0.033, 0.028] | 0.212 [0.192, 0.233] | -0.010 [-0.041, 0.019] | 0.221 [0.201, 0.242] | -0.001 [-0.033, 0.029] | 0.225 [0.205, 0.245] | 0.003 [-0.028, 0.031] |
| repetition_rate | 0.001 [0.000, 0.003] | 0.002 [0.000, 0.005] | 0.001 [-0.002, 0.004] | 0.003 [0.001, 0.005] | 0.001 [-0.001, 0.005] | 0.001 [0.000, 0.003] | 0.000 [-0.003, 0.003] | 0.001 [0.000, 0.002] | -0.001 [-0.003, 0.001] | 0.001 [0.000, 0.002] | -0.001 [-0.003, 0.001] | 0.001 [0.000, 0.002] | -0.001 [-0.003, 0.001] |
| cycle_detection_rate | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.003] | 0.001 [-0.001, 0.003] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.280 [0.265, 0.295] | 0.246 [0.231, 0.261] | -0.034 [-0.055, -0.013] * | 0.235 [0.220, 0.250] | -0.046 [-0.066, -0.025] * | 0.282 [0.267, 0.298] | 0.002 [-0.019, 0.022] | 0.280 [0.264, 0.295] | -0.001 [-0.020, 0.020] | 0.280 [0.265, 0.295] | -0.001 [-0.021, 0.020] | 0.283 [0.269, 0.298] | 0.003 [-0.019, 0.023] |
| context_affinity_without_copy | 0.209 [0.195, 0.225] | 0.177 [0.163, 0.192] | -0.032 [-0.055, -0.011] * | 0.169 [0.155, 0.182] | -0.040 [-0.062, -0.020] * | 0.212 [0.196, 0.230] | 0.003 [-0.020, 0.026] | 0.213 [0.197, 0.229] | 0.004 [-0.019, 0.027] | 0.208 [0.192, 0.225] | -0.001 [-0.024, 0.022] | 0.210 [0.193, 0.226] | 0.001 [-0.022, 0.025] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — | insufficient data | — |

C0: distinct-2 = 0.649 (basis 14733), distinct-3 = 0.805 (basis 13234) — type/token ratios, comparable only at equal basis; latency p50/p95 = 19.2/28.6 ms; cache_hit_rate: 41%; mean normalized entropy: 0.211 (branching 2.93); mean applied temperature: 2.77; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C1branch_15: distinct-2 = 0.651 (basis 14772), distinct-3 = 0.804 (basis 13273) — type/token ratios, comparable only at equal basis; latency p50/p95 = 14.1/27.0 ms; cache_hit_rate: 42%; mean normalized entropy: 0.211 (branching 2.97); mean applied temperature: 2.73; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C1branch_25: distinct-2 = 0.660 (basis 14963), distinct-3 = 0.811 (basis 13464) — type/token ratios, comparable only at equal basis; latency p50/p95 = 11.0/25.5 ms; cache_hit_rate: 42%; mean normalized entropy: 0.211 (branching 2.94); mean applied temperature: 2.69; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C1gain_neg03: distinct-2 = 0.645 (basis 14711), distinct-3 = 0.800 (basis 13212) — type/token ratios, comparable only at equal basis; latency p50/p95 = 19.0/28.8 ms; cache_hit_rate: 41%; mean normalized entropy: 0.211 (branching 2.94); mean applied temperature: 2.77; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C1gain_neg06: distinct-2 = 0.654 (basis 14833), distinct-3 = 0.811 (basis 13333) — type/token ratios, comparable only at equal basis; latency p50/p95 = 19.5/29.3 ms; cache_hit_rate: 41%; mean normalized entropy: 0.214 (branching 2.96); mean applied temperature: 2.76; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C1gain_pos03: distinct-2 = 0.649 (basis 14762), distinct-3 = 0.806 (basis 13263) — type/token ratios, comparable only at equal basis; latency p50/p95 = 19.2/28.8 ms; cache_hit_rate: 41%; mean normalized entropy: 0.211 (branching 2.93); mean applied temperature: 2.77; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C1gain_pos06: distinct-2 = 0.650 (basis 14772), distinct-3 = 0.806 (basis 13273) — type/token ratios, comparable only at equal basis; latency p50/p95 = 19.6/30.4 ms; cache_hit_rate: 41%; mean normalized entropy: 0.211 (branching 2.93); mean applied temperature: 2.77; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.248 [0.203, 0.293] | 0.003 [0.000, 0.008] | 0.327 [0.298, 0.357] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.232 [0.192, 0.275] | 0.003 [0.000, 0.008] | 0.385 [0.358, 0.415] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.133 [0.101, 0.168] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.405 [0.360, 0.456] | 0.000 [0.000, 0.000] | 0.261 [0.236, 0.287] |
| C1branch_15 | generic | 375 | 1.000 [1.000, 1.000] | 0.229 [0.187, 0.272] | 0.003 [0.000, 0.008] | 0.288 [0.258, 0.316] |
| C1branch_15 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.213 [0.173, 0.253] | 0.003 [0.000, 0.008] | 0.346 [0.318, 0.376] |
| C1branch_15 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.107 [0.080, 0.135] |
| C1branch_15 | topical | 375 | 1.000 [1.000, 1.000] | 0.347 [0.299, 0.395] | 0.003 [0.000, 0.008] | 0.231 [0.206, 0.254] |
| C1branch_25 | generic | 375 | 1.000 [1.000, 1.000] | 0.211 [0.168, 0.253] | 0.003 [0.000, 0.008] | 0.270 [0.241, 0.299] |
| C1branch_25 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.192 [0.152, 0.232] | 0.003 [0.000, 0.008] | 0.327 [0.299, 0.357] |
| C1branch_25 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.106 [0.079, 0.134] |
| C1branch_25 | topical | 375 | 1.000 [1.000, 1.000] | 0.325 [0.280, 0.376] | 0.005 [0.000, 0.013] | 0.222 [0.197, 0.244] |
| C1gain_neg03 | generic | 375 | 1.000 [1.000, 1.000] | 0.253 [0.208, 0.299] | 0.003 [0.000, 0.008] | 0.329 [0.300, 0.358] |
| C1gain_neg03 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.224 [0.184, 0.267] | 0.003 [0.000, 0.008] | 0.382 [0.355, 0.412] |
| C1gain_neg03 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.138 [0.104, 0.173] |
| C1gain_neg03 | topical | 375 | 1.000 [1.000, 1.000] | 0.403 [0.357, 0.453] | 0.000 [0.000, 0.000] | 0.266 [0.243, 0.291] |
| C1gain_neg06 | generic | 375 | 1.000 [1.000, 1.000] | 0.251 [0.208, 0.293] | 0.000 [0.000, 0.000] | 0.333 [0.303, 0.360] |
| C1gain_neg06 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.221 [0.179, 0.269] | 0.003 [0.000, 0.008] | 0.382 [0.354, 0.411] |
| C1gain_neg06 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.136 [0.104, 0.172] |
| C1gain_neg06 | topical | 375 | 1.000 [1.000, 1.000] | 0.373 [0.331, 0.424] | 0.000 [0.000, 0.000] | 0.253 [0.230, 0.278] |
| C1gain_pos03 | generic | 375 | 1.000 [1.000, 1.000] | 0.251 [0.205, 0.296] | 0.000 [0.000, 0.000] | 0.327 [0.298, 0.357] |
| C1gain_pos03 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.229 [0.187, 0.272] | 0.003 [0.000, 0.008] | 0.382 [0.355, 0.411] |
| C1gain_pos03 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.134 [0.103, 0.168] |
| C1gain_pos03 | topical | 375 | 1.000 [1.000, 1.000] | 0.403 [0.357, 0.453] | 0.000 [0.000, 0.000] | 0.260 [0.235, 0.285] |
| C1gain_pos06 | generic | 375 | 1.000 [1.000, 1.000] | 0.251 [0.205, 0.296] | 0.000 [0.000, 0.000] | 0.332 [0.303, 0.361] |
| C1gain_pos06 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.240 [0.197, 0.283] | 0.003 [0.000, 0.008] | 0.388 [0.360, 0.416] |
| C1gain_pos06 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.137 [0.105, 0.171] |
| C1gain_pos06 | topical | 375 | 1.000 [1.000, 1.000] | 0.405 [0.360, 0.456] | 0.000 [0.000, 0.000] | 0.259 [0.235, 0.285] |

## Gates

- **phase2_entropy[C1branch_15]**: fail — copy Δ -0.023 [-0.055, 0.005]; distinct-2 Δ 0.002 [-0.017, 0.021]; distinct-3 Δ -0.000 [-0.022, 0.025]; affinity_without_copy Δ -0.032 [-0.055, -0.011] *; p95 27.0 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; affinity without copies dropped significantly
- **phase2_entropy[C1branch_25]**: fail — copy Δ -0.039 [-0.069, -0.011] *; distinct-2 Δ 0.011 [-0.011, 0.026]; distinct-3 Δ 0.007 [-0.018, 0.029]; affinity_without_copy Δ -0.040 [-0.062, -0.020] *; p95 25.5 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; affinity without copies dropped significantly
- **phase2_entropy[C1gain_neg03]**: fail — copy Δ -0.001 [-0.033, 0.028]; distinct-2 Δ -0.003 [-0.021, 0.016]; distinct-3 Δ -0.004 [-0.025, 0.022]; affinity_without_copy Δ 0.003 [-0.020, 0.026]; p95 28.8 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C1gain_neg06]**: fail — copy Δ -0.010 [-0.041, 0.019]; distinct-2 Δ 0.005 [-0.018, 0.021]; distinct-3 Δ 0.007 [-0.020, 0.028]; affinity_without_copy Δ 0.004 [-0.019, 0.027]; p95 29.3 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C1gain_pos03]**: fail — copy Δ -0.001 [-0.033, 0.029]; distinct-2 Δ 0.000 [-0.020, 0.018]; distinct-3 Δ 0.001 [-0.023, 0.024]; affinity_without_copy Δ -0.001 [-0.024, 0.022]; p95 28.8 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase2_entropy[C1gain_pos06]**: fail — copy Δ 0.003 [-0.028, 0.031]; distinct-2 Δ 0.001 [-0.020, 0.018]; distinct-3 Δ 0.002 [-0.023, 0.024]; affinity_without_copy Δ 0.001 [-0.022, 0.025]; p95 30.4 ms (budget 150) — distinct-2 did not rise significantly; distinct-3 did not rise significantly
- **phase5_promotion**: insufficient data — seeded generation does not exist before Phase 5
- **phase6_anticycle**: insufficient data — observed cycle_detection_rate=0.001 [0.000, 0.002] (threshold 0.05); cycle_harm_rate has only its automatic component until a manual round (doc 05 §5) is conducted
- **phase7_order4**: insufficient data — shadow data: 897 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 28.6 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression_pass (C0, informational at Phase 0)**: fail — memes never reproduced (indices): [12, 13, 14, 15, 19, 20, 21, 22, 23] (list of 27 memes, prompt set 308b7deaea0f)

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
