# 18 — Quick Wins

> Independent audit, consolidation milestone. The high-payoff/low-effort subset of the debt register ([15]) — changes that are safe, mechanical, covered by the existing 366-test suite, and deliverable in a single short session each. No production code modified during the audit.

## 0. How to use this list

Every item below is **XS–S effort** with **no architectural ripple**. Do them top-to-bottom; each is independently shippable and gated by the existing tests + `ruff`/`mypy` (run the CI trio after each: `ruff check`, `mypy app/`, `unittest discover tests`).

## 1. The list

### QW1 — Re-enable strict typing across the `app` package · XS · ([11] Q5, [13] C2)
Change `app/presentation/bot_messages.py:56` `state: object` → `state: RuntimeState` (import under `TYPE_CHECKING` to avoid the cycle, or use a small `Protocol`). Then **delete the entire `ignore_errors=true` override block** in `pyproject.toml` — 9 of the 10 listed modules already pass `--strict`; only this one annotation blocks it.
**Verify:** `mypy app/` clean with the override removed.
**Payoff:** restores full static type safety on 10 core modules in one edit.

### QW2 — Fix the CI dependency-vulnerability gate · XS · ([08] S1)
Replace `.github/workflows/ci.yml:43` `safety check -r requirements.lock` (deprecated Safety v2 CLI) with `pip-audit -r requirements.lock` (already a dev dep, no account needed).
**Payoff:** the dependency gate actually runs again.

### QW3 — Reject placeholder / identical `/pivo` secrets · S · ([08] S2, [13] C1)
In `app/config/settings.py` (~line 90-95) add: reject secrets that start with `change_me`, and require `pivo_hmac_secret != pivo_encryption_secret`.
**Verify:** add a `test_settings` case for each rejection.
**Payoff:** closes the fail-open path where forgotten placeholder secrets protect real PII.

### QW4 — Log the swallowed chat-action exception · XS · ([08] S6, [11] Q1, [14] L5)
`app/handlers/_helpers.py:25`: change `except Exception: pass` → `except Exception as exc: logger.debug("chat action failed: %s", exc)`.
**Payoff:** no more silent failures; satisfies the Bandit `B110` / Ruff `S110` note.

### QW5 — Drop the dead `get_random_pivo_message` (or wire it) · XS · ([11] Q8)
`app/domain/pivo.py:118` has no production caller (only `tests/test_pivo.py`). Either remove it and its test, or make `PivoService` call it as the single message-building entry point. Decide and act.
**Payoff:** removes confusing dead code + its cycle-breaking lazy import.

### QW6 — Unify the `errors.py` logger name · XS · ([14] L2)
`app/handlers/errors.py:10`: `logging.getLogger(__name__)` → `logging.getLogger("chat_markov")`.
**Payoff:** the global `LOG_LEVEL` / logger-tree tweaks now affect error logging too.

### QW7 — Drop redundant indexes via a migration · S · ([07] D1, [09] P4)
Add `app/migrations/008_drop_redundant_indexes.(sql|py)` dropping the 8 indexes that duplicate PK-index prefixes (keep `idx_messages_normalized_lookup`). **First** confirm each repo query's plan with `EXPLAIN QUERY PLAN`; the audit already verified the representative transition lookup uses `sqlite_autoindex_*`.
**Verify:** `test_migrator` + a plan assertion (= [12] T4).
**Payoff:** lighter writes, smaller DB, zero read regression.

### QW8 — Document the single-instance invariant · XS · ([10] A4)
Add a short comment by `ThrottlingMiddleware._last_used` and the LRU caches noting they are safe only under one event loop / one process, and that adding background tasks or workers requires synchronization.
**Payoff:** prevents a future change from silently introducing races.

## 2. Suggested batching

| Batch | Items | Theme | Effort |
|---|---|---|---|
| **A — security/config** | QW2, QW3, QW4 | close fail-open + restore CI gate | ~½ day |
| **B — typing/hygiene** | QW1, QW5, QW6, QW8 | strict typing + dead code + invariants | ~½ day |
| **C — DB** | QW7 | redundant-index migration | ~½ day |

All three batches together are roughly **1.5 days** and clear every P1 plus several P2 items from [15]. Anything beyond this (markov complexity, incremental volume, metrics) is in [16_refactoring_plan.md] / [19_long_term_strategy.md].
