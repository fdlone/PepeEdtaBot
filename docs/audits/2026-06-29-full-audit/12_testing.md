# 12 — Testing

> Independent audit, source-only. Tests were executed (`python -m unittest discover tests`) to confirm they pass and to size the suite. Coverage is assessed qualitatively (no coverage tool installed in the venv). No production code modified.
>
> Cross-refs: async correctness in [10_async_review.md](10_async_review.md); DB write path in [07_database.md](07_database.md); items feed [18_quick_wins.md](18_quick_wins.md).

## 0. Summary

Testing is a **clear strength**. The suite is **366 tests across 23 files**, all passing in ~57s, run on the standard library `unittest` (no pytest dependency) including **async tests via `IsolatedAsyncioTestCase`**. Test files map almost 1:1 to source modules, and CI runs the suite on Python 3.12 / 3.13 / 3.14. The gaps are narrow: **no coverage measurement in CI**, and a few infrastructure modules (`registry`, repositories, `throttling` internals) are exercised only indirectly. There is no regression/coverage ratchet, but the breadth is high for a project this size.

```
Ran 366 tests in 56.833s
OK
```

## 1. Suite shape

| Dimension | Observation |
|---|---|
| Framework | stdlib `unittest`; async via `IsolatedAsyncioTestCase` (10+ files) |
| Size | 366 test methods, 23 files |
| Runtime | ~57s locally (acceptable; dominated by async DB tests) |
| CI | `python -m unittest discover tests -v` on 3.12/3.13/3.14 ([.github/workflows/ci.yml]) |
| Fixtures | `tests/fixtures/` (synthetic corpora for generation tests) |
| Coverage tool | **none** (no `coverage`/`pytest-cov` in venv; `pytest.exe` present but unused by CI) |

Largest, most valuable suites: `test_markov_and_text.py` (71 — the generation core), `test_handlers.py` (48 — Telegram handler behavior, mocked bot), `test_migrator.py` (29 — migration idempotency/ordering), `test_filters.py` (26), `test_pivo.py` (22), `test_settings.py` (21).

## 2. Coverage map (module ↔ test)

**Directly tested** (dedicated file, verified by `from app.… import` in tests): markov + text, candidate_scorer, context_state_matcher, reply_policy, response_generator, privacy_filter, lexicon, database/db logic, migrator, settings, runtime_config, runtime_state, log_masking, bot_messages, filters, all handlers (`common`/`admin`/`pivo`/`learning`/`errors`/`_helpers`), learning_service, pivo (domain), pivo_service, pivo_message_builder, pivo_parser, main (wiring).

**Indirectly covered (no dedicated test):**
- `app/config/registry.py` — exercised through `settings`/`runtime_config` tests, but the registry is the anti-duplication keystone; a direct test of `RUNTIME_FIELDS`/`validate_cross_fields`/`try_apply` would lock the contract. **Gap T1.**
- `app/repositories/*` (markov/messages/chat_members/pivo_usage) — covered via `test_db_logic` through the facade, not unit-tested in isolation. Acceptable (thin SQL wrappers).
- `app/middlewares/throttling.py` — referenced in `test_filters`/`test_main` (wiring), but the **TTL/overflow pruning logic** (`_prune_state`, the 64-tick cadence, max-keys eviction) appears undertested. **Gap T2.**

## 3. Quality observations

- **Negative paths are tested.** The run surfaces a deliberate `RuntimeError: telegram failed` (a mocked `side_effect`), i.e. error-handling branches are asserted, not just happy paths. `test_error_handler.py` covers the global error handler.
- **Async edges covered.** DB atomicity, migrator transactions, and handler coroutines run under `IsolatedAsyncioTestCase` — the async surface from [10] is tested, not just sync helpers.
- **Privacy/crypto tested.** `test_log_masking.py` (fail-fast before init, determinism) and `test_privacy_filter.py` (18) protect the privacy guarantees from [08].
- **Determinism.** Generation tests use synthetic corpora fixtures; the module-global `random` ([11] Q9) is monkeypatched in tests — which is exactly why Q9 (inject RNG) would make these tests cleaner.

## 4. Gaps & recommendations

| # | Gap | Priority | Recommendation |
|---|---|---|---|
| T1 | `registry.py` not unit-tested directly | P3 | Add a test asserting every `RUNTIME_FIELDS` spec round-trips (parse→validate→apply) and that `validate_cross_fields` rejects `backoff_min_order >= markov_order` (= [13] C3). |
| T2 | `throttling` TTL/overflow pruning undertested | P3 | Unit-test `_prune_state`: TTL expiry, max-keys eviction order, the 64-tick cadence. |
| T3 | No coverage measurement / regression ratchet | P3 | Add `coverage run -m unittest` + a CI threshold (or at least a report artifact). Low effort, high long-term value. |
| T4 | D1/D2 DB changes (when made) need tests | P4 | When dropping redundant indexes ([07] D1) or making volume incremental ([07] D2), add `EXPLAIN QUERY PLAN`/volume-correctness tests to prevent regressions. |

No test is flaky or skipped, and the suite is fast enough to run on every commit. The main lever is **measuring** coverage (T3) so the existing strong breadth becomes a defended baseline rather than an assumption.
