# Eval report 2026-08-14 snapshot=phase9-grid prompts=308b7deaea0f seeds=42,1337,2026 mode=noctx

Revision: `c475e0a`. Generations per configuration: 500.
Context mode: **noctx** — no context tokens are supplied; the prompt only selects the generation and seeds the RNG.
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
| candidate_accept_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] | 1.000 [1.000, 1.000] | 0.000 [-0.000, 0.000] |
| mean_response_length | 10.510 [10.282, 10.741] | 10.309 [10.083, 10.537] | -0.201 [-0.531, 0.149] | 10.425 [10.195, 10.669] | -0.085 [-0.409, 0.261] | 10.327 [10.101, 10.569] | -0.183 [-0.519, 0.142] |
| unique_token_ratio | 0.988 [0.986, 0.990] | 0.989 [0.988, 0.991] | 0.001 [-0.001, 0.004] | 0.988 [0.986, 0.990] | 0.000 [-0.002, 0.002] | 0.988 [0.986, 0.990] | 0.000 [-0.002, 0.002] |
| exact_context_copy_rate | 0.009 [0.004, 0.013] | 0.009 [0.004, 0.014] | 0.000 [-0.007, 0.007] | 0.009 [0.005, 0.014] | 0.000 [-0.007, 0.007] | 0.008 [0.004, 0.013] | -0.001 [-0.007, 0.006] |
| repetition_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_detection_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.024 [0.019, 0.029] | 0.024 [0.019, 0.028] | -0.000 [-0.007, 0.006] | 0.023 [0.018, 0.027] | -0.001 [-0.008, 0.005] | 0.023 [0.018, 0.027] | -0.001 [-0.008, 0.006] |
| context_affinity_without_copy | 0.024 [0.019, 0.029] | 0.024 [0.019, 0.029] | 0.000 [-0.007, 0.007] | 0.023 [0.018, 0.027] | -0.001 [-0.007, 0.006] | 0.023 [0.019, 0.028] | -0.001 [-0.008, 0.006] |
| seeded_present_rate | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — | insufficient data | — | insufficient data | — |
| historical_meme_rate | 0.019 [0.005, 0.035] | 0.024 [0.011, 0.040] | 0.005 [-0.016, 0.027] | 0.021 [0.008, 0.037] | 0.003 [-0.019, 0.021] | 0.019 [0.008, 0.035] | 0.000 [-0.021, 0.019] |

C0: distinct-2 = 0.686 (basis 14265), distinct-3 = 0.836 (basis 12766) — type/token ratios, comparable only at equal basis; latency p50/p95 = 15.0/24.3 ms; cache_hit_rate: 27%; mean normalized entropy: 0.231 (branching 3.13); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b15: distinct-2 = 0.698 (basis 13963), distinct-3 = 0.844 (basis 12463) — type/token ratios, comparable only at equal basis; latency p50/p95 = 14.4/24.3 ms; cache_hit_rate: 37%; mean normalized entropy: 0.250 (branching 3.82); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b30: distinct-2 = 0.694 (basis 14138), distinct-3 = 0.841 (basis 12638) — type/token ratios, comparable only at equal basis; latency p50/p95 = 13.1/22.5 ms; cache_hit_rate: 37%; mean normalized entropy: 0.262 (branching 3.78); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

C6b50: distinct-2 = 0.696 (basis 13990), distinct-3 = 0.841 (basis 12490) — type/token ratios, comparable only at equal basis; latency p50/p95 = 13.9/23.7 ms; cache_hit_rate: 37%; mean normalized entropy: 0.275 (branching 3.77); mean applied temperature: 2.68; temporal blend: coverage 0.0%, shift 0.0000; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.018 [0.014, 0.022] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.011 [0.003, 0.021] | 0.000 [0.000, 0.000] | 0.014 [0.010, 0.018] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.043 [0.024, 0.064] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.023 [0.018, 0.028] |
| C6b15 | generic | 375 | 1.000 [1.000, 1.000] | 0.011 [0.003, 0.021] | 0.000 [0.000, 0.000] | 0.018 [0.013, 0.022] |
| C6b15 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.015 [0.011, 0.020] |
| C6b15 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.013 [0.003, 0.024] | 0.000 [0.000, 0.000] | 0.040 [0.021, 0.059] |
| C6b15 | topical | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.023 [0.020, 0.027] |
| C6b30 | generic | 375 | 1.000 [1.000, 1.000] | 0.011 [0.003, 0.021] | 0.000 [0.000, 0.000] | 0.019 [0.015, 0.024] |
| C6b30 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.014 [0.010, 0.018] |
| C6b30 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.037 [0.019, 0.056] |
| C6b30 | topical | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.023 [0.019, 0.026] |
| C6b50 | generic | 375 | 1.000 [1.000, 1.000] | 0.011 [0.003, 0.021] | 0.000 [0.000, 0.000] | 0.017 [0.013, 0.022] |
| C6b50 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.016 [0.012, 0.021] |
| C6b50 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.008 [0.000, 0.019] | 0.000 [0.000, 0.000] | 0.035 [0.018, 0.054] |
| C6b50 | topical | 375 | 1.000 [1.000, 1.000] | 0.005 [0.000, 0.013] | 0.000 [0.000, 0.000] | 0.024 [0.020, 0.028] |

## Gates

- **phase2_entropy**: insufficient data — no Phase 2 arm in this run (entropy sampling not enabled)
- **phase3_temporal**: insufficient data — no Phase 3 arm in this run (temporal blend not enabled)
- **phase4_memes**: insufficient data — no Phase 4 arm in this run (meme scoring not enabled)
- **phase5_promotion**: insufficient data — no Phase 5 arm in this run (seeded generation not enabled)
- **phase9_interp[C6b15]**: insufficient data — distinct-2 Δ 0.012 [-0.014, 0.025]; distinct-3 Δ 0.008 [-0.021, 0.026]; copy Δ 0.000 [-0.007, 0.007]; affinity_without_copy Δ 0.000 [-0.007, 0.007]; repetition Δ 0.000 [0.000, 0.000]; cycles 0.000 [0.000, 0.000]; p95 24.3 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp[C6b30]**: insufficient data — distinct-2 Δ 0.008 [-0.016, 0.023]; distinct-3 Δ 0.004 [-0.022, 0.027]; copy Δ 0.000 [-0.007, 0.007]; affinity_without_copy Δ -0.001 [-0.007, 0.006]; repetition Δ 0.000 [0.000, 0.000]; cycles 0.000 [0.000, 0.000]; p95 22.5 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase9_interp[C6b50]**: insufficient data — distinct-2 Δ 0.009 [-0.015, 0.024]; distinct-3 Δ 0.005 [-0.022, 0.026]; copy Δ -0.001 [-0.007, 0.006]; affinity_without_copy Δ -0.001 [-0.008, 0.006]; repetition Δ 0.000 [0.000, 0.000]; cycles 0.000 [0.000, 0.000]; p95 23.7 ms (budget 150) — missing: connectedness round (M3R-020 solo protocol) — distinct-2 did not rise significantly; distinct-3 did not rise significantly; gate requires both context modes, this run measured noctx only (ctx not measured)
- **phase6_anticycle**: close — cycle_detection_rate 0.000 [0.000, 0.000] wholly below the 0.05 threshold — cycles are not frequent, the rate×harm conjunction cannot hold, so Phase 6 closes without implementation (M2R-600/610 not built); the manual harm round is not required (ADR-015)
- **phase7_order4**: fail — shadow order-4 share 0.0% vs threshold 10% over 1501 eligible steps (estimator=window — conservative lower bound); the exact-copy condition is checked at Phase 7 proposal time
- **performance.generation_p95**: pass — C0 p95 = 24.3 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression_pass (C0, informational at Phase 0)**: fail — memes never reproduced (indices): [3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26] (list of 27 memes, prompt set 308b7deaea0f)

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
