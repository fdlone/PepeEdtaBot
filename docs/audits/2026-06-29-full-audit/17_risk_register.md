# 17 — Risk Register

> Independent audit, consolidation milestone. Frames the findings from [07]–[15] as **risks** — likelihood × impact, current mitigation, residual exposure, and owner action. Complements the debt register ([15], which is fix-oriented) with an exposure-oriented view. No production code modified.
>
> Likelihood/Impact: Low / Med / High. **Exposure** = combined residual risk after existing mitigations.

## 0. Risk summary

The system carries **no high-exposure risk**. The notable risks are a **fail-open secret default** (R1) and a **possibly-inert CI dependency gate** (R2) — both Medium exposure, both cheaply closed ([18] QW2/QW3). The rest are Low: scaling/efficiency, a latent concurrency assumption, and observability blind spots that slow incident diagnosis rather than cause incidents.

## 1. Register

| ID | Risk | Likelihood | Impact | Existing mitigation | Exposure | Action |
|---|---|---|---|---|---|---|
| **R1** | Deployer keeps placeholder/equal `/pivo` secrets (pass `len≥16`) → PII protected by public secrets | Med | Med (PII confidentiality for that deployment) | `.env.example` warns; `BOT_TOKEN` fails closed | **Med** | QW3 — reject `change_me*`, require distinct secrets ([08] S2) |
| **R2** | CI dependency gate (`safety check`) silently not running (deprecated v2 CLI) → vulnerable dep merges | Med | Med | Bandit + manual review still run; deps are few/pinned | **Med** | QW2 — switch to `pip-audit -r` ([08] S1) |
| **R3** | DB write cost grows with model (per-write double SUM + redundant indexes) | Med (over time) | Low–Med (latency on busy chats) | Small data today; indexed; cached reads | **Low–Med** | D1/D2 ([07]/[09]) |
| **R4** | Single connection + global lock caps throughput | Low (current traffic) | Med (if scaled) | Single-instance design; fast hot paths | **Low** | Defer; revisit if traffic grows ([09] P1) |
| **R5** | In-memory throttle/cooldown/cache state breaks if multi-worker/multi-instance is ever added | Low | Med (incorrect throttling/cooldown) | Single-process today; correct by construction | **Low** | QW8 — document invariant ([10] A4) |
| **R6** | Lost type safety on 10 modules (ignore_errors) lets a type regression slip in | Low | Low–Med | 9/10 already strict-clean in practice; tests catch most | **Low** | QW1 — re-enable strict ([11] Q5) |
| **R7** | Silent exception swallow hides a recurring failure | Low | Low | Scope is only the typing chat-action | **Low** | QW4 — log it ([08] S6) |
| **R8** | Generation-core complexity (`markov.py`) causes a regression during future edits | Low–Med | Med (reply quality/correctness) | 71 generation tests; pure-function design | **Low–Med** | Extract sub-steps + keep tests green ([11]/[16]) |
| **R9** | Invalid config combo (`backoff_min_order ≥ markov_order`) silently degrades generation | Low | Low | Documented in `.env.example` | **Low** | C3 cross-field validation ([13]) |
| **R10** | Observability gap delays incident diagnosis (no metrics, Docker-only health) | Med | Low | Privacy-aware debug logs; `GenerationTrace` | **Low** | L3/L4 ([14]) |
| **R11** | Unmeasured coverage erodes over time | Low | Low | 366 tests today, CI-run | **Low** | T3 — coverage in CI ([12]) |
| **R12** | Secret rotation invalidates existing `/pivo` subscriptions unexpectedly | Low | Low | Documented behavior in `.env.example` | **Low** (accepted) | None — by design |

## 2. Accepted risks (no action)

- **R4 / R12** are deliberate design trade-offs (single-instance simplicity; deterministic secret-derived hashing). Accept and document; do not engineer around them prematurely.
- **f-string DDL in migration 005** ([08] S5): inputs are DB-internal `PRAGMA` names, not user data → not a real injection risk. Accepted.
- **`/pivo` non-crypto `random`** ([08] §3): correct choice for text/gameplay variety; not a security risk. Accepted.

## 3. Risk trend

All Medium-exposure risks (R1, R2) are **closeable this week** with quick wins. After [18] Batch A, the register is uniformly **Low** exposure, with the only remaining structural items (R4 lock, R8 complexity) gated on growth/feature pressure rather than standing exposure. See [16_refactoring_plan.md] for sequencing and [19_long_term_strategy.md] for the scale-up posture.
