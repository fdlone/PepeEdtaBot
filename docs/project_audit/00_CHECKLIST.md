# PepeEdtaBot — Audit Progress Checklist

> Source of truth for the deep project audit driven by `docs/Project Audit.md`.
> Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[!]` blocked / needs input.
> Updated by Claude across sessions. Last update: 2026-06-29.

## Ground rules (from TZ)
- Audit from source only; ignore prior audit conclusions (Phase 0).
- No production-code edits, no refactoring, no file deletions during audit.
- Every finding needs evidence: file, class/function, reason, impact, confidence, fix.
- Use tooling first (Trail of Bits skills + Ruff/Astral), then manual validation.

## Tooling setup
- [x] Trail of Bits marketplace present (`~/.claude/plugins/marketplaces/trailofbits`).
- [x] Relevant ToB plugins enabled in `~/.claude/settings.json` (modern-python, audit-context-building, static-analysis, insecure-defaults, supply-chain-risk-auditor, sharp-edges, agentic-actions-auditor, differential-review, variant-analysis, fp-check, semgrep-rule-creator).
- [ ] Session restarted so ToB skills register as native tools (recommended before deep security work).
- [x] Verified local tools: ruff, mypy, pip-audit, pytest in `.venv/Scripts`; `uv` 0.11.6 (semgrep via `uvx`); no global codeql.

## Milestones / Phases
### M1 — Discovery + Architecture (docs 01–06) — task #1 ✅ DONE
- [x] Phase 0: prior audits quarantined to `docs/_pre_audit_archive/` (6 docs) + README marker; ARCHITECTURE/OPERATIONS/README kept as functional docs only
- [x] Phase 1: full repository discovery + module inventory (purpose, interfaces, deps, callers)
- [x] Phase 2: architecture reconstruction (high-level, dep graph, execution/startup/shutdown, Telegram update lifecycle, async/data flow; coupling, cycles, dead code)
- [x] 01_project_overview.md
- [x] 02_repository_map.md
- [x] 03_architecture.md
- [x] 04_execution_flow.md
- [x] 05_module_inventory.md
- [x] 06_dependency_graph.md
- [~] **Checkpoint with user** (awaiting review before M2)

### M1 findings parked for later phases
- mypy `strict=true` for `app.*` but `ignore_errors=true` for 10 core modules (settings, database, registry, runtime_*, reply_policy, text, pivo, pivo_templates, bot_messages) → effective strict coverage is partial. → M2/[11]
- `domain.pivo.get_random_pivo_message` has no production caller (lazy import to break cycle) → possible dead code. → M2/[11]
- `runtime_config.parse_bool` duplicates `registry._parse_bool` (different return contract). → M2/[11]
- `database.py` god-object tendency; ~10 repeated `if self.x is None: raise` guards. → M2/[11]
- `pivo_message_builder`/`_helpers.reply_humanized` use module-global `random` (non-injectable). → M2/[11]
- core ⇄ infrastructure bidirectional package coupling (markov→database, database→core.text). → [11]/[16]
- requirements.lock has no hashes; `aiofiles` present though no runtime import seen. → M3 deps
- Single-instance assumption (all throttle/cooldown state in-memory). → [19] scalability

### M2 — Code Quality (doc 11) — task #2 ✅ DONE
- [x] Run Ruff; classify each diagnostic (style/maintainability/correctness/bug/perf/security) — project profile clean; ALL=902 classified
- [x] Run mypy; record typing findings — strict clean as configured; ignore_errors list over-broad (9/10 modules already strict-clean)
- [x] Apply ToB `modern-python` review — substantially met (UP clean, py312, lockfile)
- [x] Manual quality pass (SOLID/DRY/KISS/YAGNI, god objects, dead code, smells) — Q1–Q9 logged
- [x] 11_code_quality.md
- [~] **Checkpoint with user** (awaiting review before M3)

### M2 key findings (detail in 11_code_quality.md)
- Q5 (quick win): `bot_messages.format_config_message(state: object)` is the ONLY blocker for full strict; 9/10 excluded modules already pass `--strict`. Type it + drop `ignore_errors` block.
- Q6: `database.py` repeats `if self.markov is None: raise RuntimeError(...)` ~12× → extract `_require_init()`.
- Q1: silent `except Exception: pass` in `_helpers.py:25` (typing chat-action) → log debug.
- Complexity hotspots: `markov._generate_text_once` C901=36, `_select_contextual_state`=29, `settings.load_settings`=25.
- Q7 parse_bool dup, Q8 dead `get_random_pivo_message` (test-only), Q9 non-injectable module-global `random`.

### M3 — Security + Dependencies + Configuration (docs 08, 13) — task #3 ✅ DONE
- [x] ToB `static-analysis`: Bandit gate clean medium/medium (Low=24 benign B311/B110) + Semgrep (uvx, p/python+security-audit+secrets, 237 rules) → 4 findings ALL false-positive (logger-credential-leak matched "tokens"/"context" words; values are counts/booleans at debug, chat_id masked)
- [x] ToB `insecure-defaults` → S2: placeholder pivo secrets pass len>=16 check (fail-open)
- [x] ToB `supply-chain-risk-auditor` + `pip-audit`: runtime clean; 2 CVEs (joserfc/msgpack) are DEV-only transitive, not in requirements.lock
- [x] ToB `sharp-edges` → S2/S3 (length-only secret check; non-KDF Fernet key)
- [x] ToB `agentic-actions-auditor` on ci.yml: no AI actions, no pull_request_target, clean
- [x] ToB `fp-check`: confirmed pip-audit hits are FP for production
- [x] Manual: secrets/authz/SQL/injection/RCE/deserialization/logging — all clean (long-poll, no subprocess/eval/pickle, parameterized SQL, HKDF log masking)
- [x] Phase 7 dependency audit — 4 runtime deps, pinned w/ upper bounds; lock has no hashes (documented); aiofiles is aiogram-transitive (not unused)
- [x] Phase 9 configuration audit — registry single-source-of-truth, fail-fast validation; C1/C2/C3 findings
- [x] 08_security.md
- [x] 13_configuration.md
- [~] **Checkpoint with user** (awaiting review before M4)

### M3 key findings (detail in 08_security.md / 13_configuration.md)
- Posture STRONG. 0 Critical/High. No subprocess/eval/exec/pickle/yaml. All SQL parameterized. Long-poll only (no webhook surface).
- S1 (Med): CI `safety check -r requirements.lock` uses deprecated Safety v2 CLI → dependency gate likely broken. Replace with `pip-audit -r`.
- S2/C1 (Med): `.env.example` placeholder pivo secrets pass len>=16 → fail-open PII exposure if deployer forgets. Reject `change_me*` + require hmac!=encryption.
- S3 (Low): Fernet key = single-pass sha256(secret); log_masking correctly uses HKDF — make consistent.
- S4 (Low): get_chat_administrators uncached per admin command (perf/API-rate, → M4).
- markov.db NOT tracked (stale prior note corrected). pip-audit cp1251 crash on lock header under Windows locale.

### M4 — Performance + Async + Database (docs 07, 09, 10) — task #4 ✅ DONE
- [x] Phase 5 performance — no blocking I/O in request path; write batched; 2-tier LRU caching; P1–P7 logged
- [x] Phase 6 async review — request-driven only (no create_task/workers/schedulers); single lock deadlock-free (verified); no leaks/races; graceful shutdown
- [x] Phase 8 database — live-schema introspection + EXPLAIN QUERY PLAN; WAL, FK on, parameterized; D1 redundant indexes, D2 per-write double SUM
- [x] 07_database.md
- [x] 09_performance.md
- [x] 10_async_review.md
- [~] **Checkpoint with user** (awaiting review before M5)

### M4 key findings (detail in 07/09/10)
- D1/P4 (Med): 8 redundant secondary indexes duplicate PK-index prefixes; EXPLAIN QUERY PLAN proves planner uses sqlite_autoindex_* not idx_*. Pure write amplification → drop in a migration (keep idx_messages_normalized_lookup).
- D2/P2 (Med): two SUM(cnt) full-chat aggregations on EVERY learned message (volume3/volume2) → O(rows-per-chat), grows with model. Maintain incrementally or compute lazily.
- P1 (Med scalability): single aiosqlite connection + single asyncio.Lock serializes ALL DB I/O; WAL concurrent-reader benefit unused. Fine single-instance; the ceiling.
- P3 (Low-Med): per-step transition queries on cold cache (LRU-mitigated). P5: uncached get_chat_administrators (=S4). P6: ResponseGenerator per-message alloc.
- Async CORRECT by construction: no spawned tasks, no races (loop-confined state), deadlock-free (read delegates don't take lock; write path doesn't re-enter). A4 latent: in-memory throttle/cache state unsafe IF multi-worker/multi-instance ever added — document invariant.
- Schema evolved via 7 migrations (007: pivo_chat_members→chat_members; 006 pivo_daily_usage; 005 dropped messages.text; 003 anonymized author_id). Live markov.db: ~101 msgs.

### M5 — Logging + Testing + Doc accuracy (docs 12, 14) — task #5 ✅ DONE
- [x] Phase 10 logging/observability — single basicConfig logger, env-driven level, privacy-aware; L1–L5
- [x] Phase 11 testing — 366 tests/23 files ALL PASS (~57s), unittest + IsolatedAsyncioTestCase; T1–T4 gaps
- [x] Phase 12 documentation accuracy — README/ARCH/OPS verified accurate vs code (compose.yaml/entrypoint exist, commands/routers match); folded into 14 §5
- [x] 12_testing.md
- [x] 14_logging.md
- [~] **Checkpoint with user** (awaiting review before M6)

### M5 key findings (detail in 12/14)
- Testing STRONG: 366 tests pass, broad module↔test 1:1 mapping, async edges covered, negative paths tested. Gaps: T1 registry not unit-tested directly; T2 throttling TTL/overflow pruning undertested; T3 NO coverage measurement/ratchet in CI (pytest.exe present but unused; no coverage tool); T4 add tests when D1/D2 land.
- Logging: privacy-aware (HKDF chat_id mask, no content/tokens/secrets logged — confirms M3). Observability MINIMAL: L3 no metrics/tracing; L4 health is Docker SELECT 1 only (not Telegram-liveness); L5 error log-only no Sentry; L2 errors.py uses getLogger(__name__) not "chat_markov".
- Docs ACCURATE (Phase 12): all README references/commands/routers/privacy claims match code; README honestly documents the non-KDF Fernet derivation (=S3). Minor: D3 /config handler has no GroupOnly/AdminOrOwner (readonly non-secret display, not a security issue).

### M6 — Debt + Roadmap + Summary (docs 15–20) — task #6 ✅ DONE
- [x] Phase 13 technical debt (description/impact/risk/effort/priority) — 15_technical_debt.md, all M2-M5 findings consolidated
- [x] Phase 14 phased refactoring roadmap — 16_refactoring_plan.md (6 phases, dependency-ordered)
- [x] 15_technical_debt.md
- [x] 16_refactoring_plan.md
- [x] 17_risk_register.md (R1-R12, exposure framing)
- [x] 18_quick_wins.md (QW1-QW8, ~1.5 day batches)
- [x] 19_long_term_strategy.md (3-horizon, invariants to preserve)
- [x] 20_executive_summary.md (scorecard + top findings + cross-ref table)
- [x] Cross-reference all docs (20 has full deliverable index; all docs link [NN])
- [~] **Final review with user** (all 21 docs 00-20 complete; awaiting user sign-off)

## AUDIT COMPLETE — all milestones M1-M6 done (2026-06-29). 21 docs (00_CHECKLIST + 01-20). No production code modified.
Top actionable (P1, ~1.5d): S1 fix CI safety→pip-audit; S2 reject placeholder secrets; Q5 re-enable strict typing. Then D1 (drop redundant indexes), D2 (incremental volume). Verdict: strong, low-debt, 0 Critical/High.

## REMEDIATION (PR #49, branch chore/close-audit-findings)
- [x] **P1** (commit ee5c649): S1 CI pip-audit, S2 reject placeholder secrets, Q5 re-enable full strict typing.
- [x] **P2** (commit 46b6c19): D1 drop 8 redundant indexes (migration 008), D2 incremental per-chat model volume (migration 009).
- [x] **P3 batch 1** (commit 89e0774): Q6 `Database._require()` guard helper; S4/P5 admin-id TTL cache; Q7 `parse_bool` reuses `_parse_bool`; Q1/S6/L5 debug-log instead of silent except; L2 errors.py logger name.
- [x] **P3 batch 2** (commit 7564543): C3 confirmed (validate_cross_fields already enforces all 3 invariants at boot + /set) + T1 direct registry unit tests; T2 (throttling pruning) already covered.
- [x] **Q9** (commit 6e64886): inject RNG into pivo_message_builder + reply_humanized.
- [x] **T3** (commit 738873c): coverage measurement + ratchet (fail_under=87) in CI.
- [~] **S3** — DEFERRED by user (2026-06-30): HKDF for Fernet key would break decryption of existing /pivo PII; Low/Info, README documents the sha256 derivation. Keep as-is.
- [x] **R8** — markov complexity refactor under characterization tests (DONE, 2026-06-30):
  - [x] Characterization safety net: `tests/test_markov_generation_characterization.py` (25 tests) pins `_generate_text_once`/`generate_text_with_trace` output across all branches + pure-helper & mocked-matcher unit tests.
  - [x] `_generate_text_once`: **C901 36 → ≤10**. Extracted `_pick_seed_start`, `_pick_global_start`, `_pick_contextual_start` + `_contextual_match_counts`, `_finalize_attempt`, `_run_generation_loop` (commits cbb0712, 32c20b1, c07b7bc, 0a19148, f6f78f5). Removed a now-unreachable defensive guard.
  - [x] `_select_contextual_state`: **C901 29 → ≤10** (commit c57a744). Extracted `weighted_population_choice`, `_build_exact3/2_candidates`, unified `_build_fuzzy3/2_candidates` (casefold+prefix); prefix path locked with mocked-matcher unit tests.
  - Note: remaining `weighted_next_choice`=16 and `_run_generation_loop`=15 are inherent algorithmic branching, NOT in audit scope — left as-is.
- [ ] Deferred (only if scaling): L3 metrics, P1 single-lock, A4 multi-instance.

## Open questions / blockers
- (none yet)

## Notes / running log
- 2026-06-29: Scaffold created. Repo snapshot: `app/` layered (config, core, domain, filters, handlers, infrastructure, middlewares, migrations, presentation, repositories, services), `main.py` entrypoint, `markov.db` (sqlite, committed), `tools/`, `tests/`. ~79 Python files (excl. venv). Telegram bot (aiogram-style) + Markov text generation; `db_prod_copy/` and `markov.db` present in repo.
