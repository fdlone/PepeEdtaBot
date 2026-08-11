# Eval report 2026-08-11 snapshot=phase1-gate prompts=308b7deaea0f seeds=42,1337,2026

Revision: `d38e3e7`. Generations per configuration: 500.
- IDF for affinity metrics is computed over the snapshot's retained message window (full history is not stored) — window-relative, identical across configurations (audit §3).

## Config matrix

- **C0**: 1500 generations
- **CF**: 1500 generations — results shared with C0
- unavailable (feature not implemented yet): C1, C2, C3, C4, C5

## Metrics table

Value [95% CI] per configuration; delta vs C0 [95% CI], `*` = significant (interval excludes 0, doc 05 §4).

| metric | C0 | CF | Δ CF vs C0 |
|---|---|---|---|
| generation_success_rate | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| candidate_accept_rate | 0.879 [0.872, 0.887] | 0.879 [0.872, 0.887] | 0.000 [-0.010, 0.010] |
| mean_response_length | 10.822 [10.599, 11.092] | 10.822 [10.599, 11.092] | 0.000 [-0.355, 0.369] |
| unique_token_ratio | 0.986 [0.984, 0.988] | 0.986 [0.984, 0.988] | 0.000 [-0.002, 0.003] |
| exact_context_copy_rate | 0.222 [0.203, 0.243] | 0.222 [0.203, 0.243] | 0.000 [-0.032, 0.029] |
| repetition_rate | 0.001 [0.000, 0.003] | 0.001 [0.000, 0.003] | 0.000 [-0.003, 0.003] |
| cycle_detection_rate | 0.001 [0.000, 0.002] | 0.001 [0.000, 0.002] | 0.000 [-0.002, 0.002] |
| cycle_harm_rate | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| context_affinity | 0.280 [0.265, 0.295] | 0.280 [0.265, 0.295] | 0.000 [-0.020, 0.021] |
| context_affinity_without_copy | 0.209 [0.195, 0.225] | 0.209 [0.195, 0.225] | 0.000 [-0.021, 0.021] |
| seeded_present_rate | insufficient data | insufficient data | — |
| seeded_win_rate_given_present | insufficient data | insufficient data | — |
| freshness_reflection | insufficient data | insufficient data | — |

C0: distinct-2 = 0.649 (basis 14733), distinct-3 = 0.805 (basis 13234) — type/token ratios, comparable only at equal basis; latency p50/p95 = 26.0/48.2 ms; cache_hit_rate: 41%; shadow order-4 share: 0.0% (estimator=window); storage_delta: n/a.

CF: distinct-2 = 0.649 (basis 14733), distinct-3 = 0.805 (basis 13234) — type/token ratios, comparable only at equal basis; latency p50/p95 = 26.0/48.2 ms; cache_hit_rate: insufficient data; shadow order-4 share: insufficient data; storage_delta: n/a.

## Per-category breakdown

| config | category | n | success | copy | repetition | affinity |
|---|---|---|---|---|---|---|
| C0 | generic | 375 | 1.000 [1.000, 1.000] | 0.248 [0.203, 0.293] | 0.003 [0.000, 0.008] | 0.327 [0.298, 0.357] |
| C0 | meme-bait | 375 | 1.000 [1.000, 1.000] | 0.232 [0.192, 0.275] | 0.003 [0.000, 0.008] | 0.385 [0.358, 0.415] |
| C0 | short-degenerate | 375 | 1.000 [1.000, 1.000] | 0.003 [0.000, 0.008] | 0.000 [0.000, 0.000] | 0.133 [0.101, 0.168] |
| C0 | topical | 375 | 1.000 [1.000, 1.000] | 0.405 [0.360, 0.456] | 0.000 [0.000, 0.000] | 0.261 [0.236, 0.287] |

## Gates

- **phase5_promotion**: insufficient data — seeded generation does not exist before Phase 5
- **phase6_anticycle**: insufficient data — observed cycle_detection_rate=0.001 [0.000, 0.002] (threshold 0.05); cycle_harm_rate has only its automatic component until a manual round (doc 05 §5) is conducted
- **phase7_order4**: insufficient data — shadow data: 897 eligible steps (need >= 1000 for a verdict; estimator=window)
- **performance.generation_p95**: pass — C0 p95 = 48.2 ms (budget 150 ms)
- **performance.lookup_p95**: insufficient data — distribution-lookup instrumentation lands in Phase 1
- **meme_regression_pass (C0, informational at Phase 0)**: fail — memes never reproduced (indices): [12, 13, 14, 15, 19, 20, 21, 22, 23] (list of 27 memes, prompt set 308b7deaea0f)

## Manual eval summary

Not conducted in this run (first required at the Phase 4 gate).

## Verdict per phase

- Phase 0: baseline frozen on this snapshot/prompts/seeds; later phases measure against these numbers. Temporal metrics report `insufficient data` until Phase 3 accumulates timestamps (audit §10.1).
