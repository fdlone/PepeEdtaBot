# 15 — Technical Debt Register

> Independent audit, consolidation milestone. Aggregates every finding from [07]–[14] into a single prioritized debt register. Each item: description, impact, risk, effort, priority, and the source doc. No production code modified during the audit.
>
> Priority key: **P1** do first (cheap + real risk) · **P2** soon (real value) · **P3** opportunistic · **P4** nice-to-have. Effort: **XS** <1h · **S** ~½ day · **M** 1–2 days · **L** >2 days.

## 0. Overall debt posture

This is a **low-debt, well-built codebase**. There are **no Critical/High security issues**, the async model is correct by construction, tests are broad and green (366 passing), and the documentation is accurate. The debt that exists is concentrated in three buckets: (a) a handful of **cheap correctness/hygiene fixes** (typing, CI gate, secret validation), (b) **DB write-path efficiency** (redundant indexes, per-write aggregation), and (c) **maintainability of the generation core** (`markov.py` complexity). Nothing here is fire-fighting; it is steady hardening.

## 1. Debt register (prioritized)

| ID | Item | Impact | Risk | Effort | Prio | Source |
|---|---|---|---|---|---|---|
| **S1** | CI `safety check -r requirements.lock` uses deprecated Safety v2 CLI → dependency gate likely not running | Vulnerable dep could merge unnoticed | Med | XS | **P1** | [08] |
| **S2/C1** | Placeholder `/pivo` secrets (`change_me…`) pass `len≥16` validation → fail-open | Public crypto secrets protect PII if deployer forgets | Med | S | **P1** | [08][13] |
| **Q5/C2** | `bot_messages.format_config_message(state: object)` blocks strict typing for whole `app` (9/10 "excluded" modules already strict-clean) | Type safety disabled on 10 modules unnecessarily | Low | XS | **P1** | [11][13] |
| **D1/P4** | 8 redundant secondary indexes duplicate PK-index prefixes (planner never uses them) | Write amplification, larger DB | Low | S | **P2** | [07][09] |
| **D2/P2** | Double `SUM(cnt)` full-chat aggregation on every learned message | Write cost grows with model size | Med | M | **P2** | [07][09] |
| **Q6** | `database.py` repeats `if self.markov is None: raise …` ~12× verbatim | DRY/readability; god-object smell | Low | S | **P2** | [11] |
| **Q1/S6/L5** | Silent `except Exception: pass` around typing chat-action (no log) | Hidden failures | Low | XS | **P2** | [08][11][14] |
| **S4/P5** | `get_chat_administrators` uncached per admin command | Latency + Telegram API rate use | Low | S | **P3** | [08][09] |
| **T3** | No coverage measurement / regression ratchet in CI | Strong coverage is unmeasured/undefended | Low | S | **P3** | [12] |
| **markov complexity** | `_generate_text_once` (C901=36), `_select_contextual_state` (29), `load_settings` (25) | Hard to modify safely | Med | L | **P3** | [11] |
| **C3** | No cross-field validation `backoff_min_order < markov_order`, reply-context bounds | Invalid combo degrades generation silently | Low | S | **P3** | [13] |
| **L3** | No metrics/tracing | Blind to runtime behavior/perf | Low–Med | M | **P3** | [14] |
| **Q7** | Duplicate `parse_bool` (registry vs runtime_config, different contracts) | DRY | Low | S | **P3** | [11] |
| **Q9** | Module-global `random` in pivo builder & `_helpers` (non-injectable) | Testability (forces monkeypatch) | Low | S | **P3** | [11] |
| **T1/T2** | `registry` & `throttling` pruning undertested | Contract regressions slip through | Low | S | **P3** | [12] |
| ~~**Q8**~~ ✅ | Dead `get_random_pivo_message` (test-only caller) — **RESOLVED 2026-07-07** (removed + `build_pivo_mentions` + tests) | Confusing dead code | Low | XS | ~~P4~~ done | [11] |
| **L2** | `errors.py` uses `getLogger(__name__)` not `"chat_markov"` | Log-level control inconsistency | Low | XS | **P4** | [14] |
| **L4** | Health check is Docker `SELECT 1` only (not Telegram-liveness) | False-healthy if session dies | Low | M | **P4** | [14] |
| **P6** | `ResponseGenerator` instantiated per message | Minor allocation on hot path | Low | S | **P4** | [09] |
| **A4** | In-memory throttle/cooldown/cache state safe only single-instance (latent) | Breaks silently if multi-worker added | Low (now) | XS (doc) | **P4** | [10] |
| **P1** | Single connection + global lock serializes all DB I/O | Throughput ceiling | Low (now) | L | **P4** | [09] |
| **S5/P7/misc** | f-string DDL in migration (safe), blocking startup read, `os.path`→`pathlib` | Cosmetic | Negligible | XS | **P4** | [07][08][11] |

## 2. Debt by theme

- **Security/config hardening (P1):** S1, S2/C1 — both cheap, both close real fail-open/gap risks. Do first.
- **Type safety (P1):** Q5 — one annotation re-enables strict across 10 modules.
- **DB efficiency (P2):** D1 (drop indexes), D2 (incremental volume) — the only items that scale poorly with data growth.
- **Maintainability (P2–P3):** Q6, Q1, markov complexity, Q7/Q9 — reduce friction for future changes.
- **Observability/testing (P3):** T3 (coverage), L3 (metrics), T1/T2 — turn "good by inspection" into "defended by tooling".
- **Latent/architectural (P4):** P1 (lock), A4 (single-instance invariant) — only act when traffic/topology demands it.

## 3. What is explicitly NOT debt

- The `registry` single-source-of-truth pattern — a design **asset** to emulate ([05]/[13]).
- The privacy-aware logging (HKDF masking) and crypto via `cryptography`/`hmac` — keep as-is ([08]/[14]).
- The atomic, batched write transaction and WAL setup ([07]).
- The no-spawned-tasks async model — correct by construction; do not add concurrency without need ([10]).

See [16_refactoring_plan.md](16_refactoring_plan.md) for sequencing, [17_risk_register.md](17_risk_register.md) for risk framing, [18_quick_wins.md](18_quick_wins.md) for the same-day fixes.
