# Follow-up Audit & Code Review — 2026-07-04

Scope: full re-review of security, performance and code quality after the
2026-06-29 deep audit (`2026-06-29-full-audit/`) and its remediation
(PR #49 + follow-ups). Reviewed at branch `feat/dialogue-gen-stage3-m3-m4`
(Stage 3 M1-M4 included).

## Checks run

- `ruff check app/ tests/ tools/ main.py` — clean.
- `mypy app/` (strict, 57 files) — clean.
- `unittest discover tests` — 564 tests OK before fixes, 569 OK after.
- `bandit -r app tools main.py` — 0 medium/high; 15× Low B311
  (non-crypto `random` in text generation) — benign.
- `pip-audit -r requirements.lock` — no known vulnerabilities
  (aiogram 3.29.0, aiohttp 3.14.1, cryptography 49.0.0).
- Manual: PivoSecurity (Fernet/HMAC), log_masking (HKDF), secrets loading,
  privacy invariants (normalized_text only, `author_id=0`, chat_id masking),
  AdminOrOwner, ThrottlingMiddleware, SQL parametrization in all repositories,
  Docker entrypoint/HEALTHCHECK, generation pipeline complexity,
  RuntimeState/throttle pruning, migrations atomicity.

## Prior findings re-verified

- S1/S2/Q5/D1/D2/Q6/Q7/Q9/T1/T3/R8 remediations confirmed in code.
- S3 (Fernet key = single-pass sha256(secret)) — still deferred by user
  decision (2026-06-30): switching to a KDF would break decryption of the
  existing /pivo payloads; secrets are documented as long random strings.
- D3 (`/config` without GroupOnly/AdminOrOwner) — still present, accepted as
  informational (read-only, no secrets in RuntimeState).

## Findings and statuses

| # | Severity | Finding | Status |
|---|---|---|---|
| N1 | Medium | Mention-triggered replies bypass the chat cooldown AND the hourly cap (`should_reply_to_message`: `if mentioned: return True`) — one user could force a generation+reply per message. | **Fixed** — `MENTION_COOLDOWN_SEC` (default 5 s, runtime-mutable via `/set`): per-user per-chat gate; a gated mention is demoted to the unprompted-reply path. `app/handlers/learning.py`, `app/config/registry.py`, `RuntimeState.last_mention_reply_ts`. |
| N2 | Low | Docs drift: README/ARCHITECTURE claimed CI runs `safety` (it runs `pip-audit` since S1); AdminOrOwner comment said "few-second cache" while TTL=60 s. | **Fixed** — docs and comment updated. |
| N3 | Low | `/pivo` recorded `pivo_pool_usage` anti-repeat state inside `build_call_message`, before the daily-quota check — over-quota and failed calls rotated the template pools. | **Fixed** — `build_call_message` is now side-effect free and returns picks; `PivoService.record_pool_usage` is called only after the reply is delivered. |
| N4 | Low | `/clear confirm` wiped model/messages/emoji but not the chat's /pivo data (subscriptions, daily quotas, pool anti-repeat). | **Fixed** — `PivoService.clear_chat_data(chat_id)` wired into `/clear`; confirmation and README texts updated. |
| N5 | Info | `.mcp.json` untracked and not ignored. | **Fixed** — added to `.gitignore`. |
| N6 | Info | Throttle-notify reply is itself unthrottled (1:1 amplification only). Docker HEALTHCHECK runs as root (no `USER`) and `sqlite3.connect` could create a root-owned file in edge cases. | **Fixed** (follow-up commit) — notify reply rate-limited per key (`notify_cooldown_sec`, default 30 s); HEALTHCHECK opens the DB read-only via URI (`mode=ro`), so it can no longer create a root-owned file. |
| N7 | Process | CLAUDE.md/AGENTS.md referenced `PROJECT_AUDIT.md` / `PROJECT_AUDIT_CODEX.md`, archived on 2026-06-29. | **Fixed** — both now point at `docs/audits/` (this structure). Note: both files are gitignored (local agent instructions), so the fix lives outside git. |

## Changed files (fix commit)

`app/config/{registry,settings,runtime_state}.py`, `app/handlers/{learning,pivo,admin}.py`,
`app/services/pivo_service.py`, `app/repositories/{chat_members,pivo_usage,pivo_pool_usage}_repo.py`,
`app/infrastructure/database.py`, `app/presentation/bot_messages.py`,
`app/filters/admin_or_owner.py`, `.env.example`, `.gitignore`, `README.md`,
`docs/ARCHITECTURE.md`, tests (`test_handlers`, `test_pivo`, `test_runtime_state`,
`test_fallback_phrases`).

## Docs restructure (same session)

- `docs/project_audit/` → `docs/audits/2026-06-29-full-audit/`
  (+ `docs/Project Audit.md` → `TZ.md` inside it).
- `docs/_pre_audit_archive/` (7 deprecated pre-audit docs) — deleted; history
  remains in git (last present at commit `a5f5b4e`).
- `docs/audits/README.md` — new index with statuses.
- CLAUDE.md / AGENTS.md / ARCHITECTURE.md references updated.

## Not run / limitations

- `safety` CLI not run — CI migrated to `pip-audit` (S1); matched CI instead.
- `pip-audit` on Windows requires `PYTHONUTF8=1` (cp1251 locale crash on the
  lock-file header) — CI on ubuntu is unaffected.

## Remaining / recommended

- Deferred from 2026-06-29 audit (only if scaling): L3 metrics, P1 single-lock
  DB ceiling, A4 multi-instance state.
- S3 stays as-is per user decision.

## Session update — 2026-07-04 (CI trigger fix)

### Completed
- CI never ran for PR #56 because `ci.yml`'s `pull_request` trigger was filtered
  to `branches: [main]`, while #56 is stacked on #55 (base
  `feat/dialogue-gen-stage3-m3-m4`, not `main`). The base-ref filter excluded it.
- Removed the `branches: [main]` filter from the `pull_request` trigger so CI
  runs on PRs into any base branch (stacked PRs included). `push` still gated to
  `main` to avoid duplicate full runs on feature branches.

### Changed files
- `.github/workflows/ci.yml`

### Tests/checks run
- None (workflow-only change); CI itself will exercise it on the next PR event.

## Session update — 2026-07-04 (L1 running jokes / hot n-grams)

### Completed
- Implemented L1 from `docs/DIALOGUE_GENERATION_ACTION_PLAN.md` on branch
  `feat/dialogue-gen-stage4-l1` (based on `chore/audit-followup-fixes`; must be
  rebased onto `main` after PRs #55/#56 merge, before opening its PR).
  Plan: `docs/superpowers/plans/2026-07-04-l1-hot-ngrams.md`. Commits pushed
  per task; per-task review checkpoints (diff review + bandit) applied.
- Review findings fixed along the way: casefolded stopword check (the
  case-preserved profile `normalize_lower=false` would have let capitalized
  stopword-only n-grams through) and position-major n-gram extraction (the
  size-major loop starved trigrams on long messages once the 24/message cap hit).
- Housekeeping (same session, before L1): pruned stale remote-tracking refs of
  17 already-deleted merged branches (`git fetch --prune`); verified
  `bot.log`/`markov.db*`/`__pycache__`/`.mcp.json` are all properly ignored.

### Changed files
- New: `app/core/hot_ngrams.py`, `app/repositories/chat_hot_ngrams_repo.py`,
  `app/migrations/012_chat_hot_ngrams.sql`, `tests/test_hot_ngrams.py`,
  `tests/test_chat_hot_ngrams_repo.py`,
  `docs/superpowers/plans/2026-07-04-l1-hot-ngrams.md`.
- Modified: `app/infrastructure/database.py` (repo wiring, delegates,
  `decay_chat_hot_ngrams` at init, `clear_chat` wipe),
  `app/services/learning_service.py`, `app/handlers/learning.py`
  (record on learn + unprompted-reply seeding), `app/config/{registry,settings,
  runtime_state}.py` (3 knobs), `app/repositories/__init__.py`, `.env.example`,
  `README.md`, `docs/ARCHITECTURE.md`, `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`,
  tests (`test_handlers`, `test_learning_service`, `test_db_logic`,
  `test_migrator`, `test_runtime_state`, `test_runtime_config`,
  `test_bot_messages`, `test_fallback_phrases`).

### Audit findings updated
- No new security findings. Privacy contour unchanged: `chat_hot_ngrams` is
  per-chat aggregate (no author), same normalized-token source as the word
  model, raw `chat_id` key like the model tables, wiped by `/clear`, n-gram
  text never logged (only its length at DEBUG).

### Tests/checks run
- `unittest discover tests` — 598 tests OK (was 569 before this branch).
- `ruff check app/ tests/ tools/ main.py` — clean; `mypy app/` (strict) — clean.
- `bandit -r app tools main.py` — 0 medium/high, 15× Low B311 baseline
  unchanged (the new `random.random`/`random.choice` uses are non-crypto by
  design).
- `tools/eval_generation.py` vs `tools/generation_baseline.json` — all
  non-latency metrics byte-identical (the channel is off in eval and no
  generation code path changed).
- `EXPLAIN QUERY PLAN` on the hot-ngram query — PK searches only, no
  `SCAN transitions` (pinned as a test).

### Not run / limitations
- `pip-audit` — no dependency changes in this branch; last clean run 2026-07-04
  (see the main section above).
- Live-chat behaviour of the seeding (perceived "running joke" quality) can
  only be judged in situ after deployment, per the action plan's checkpoint rule.

### Remaining work
- Rebase `feat/dialogue-gen-stage4-l1` onto `main` after #55/#56 merge, open PR.
- Observe Stage 3 + L1 in the live chat before starting L3 (then L2, which
  needs its own privacy review).

## Session update — 2026-07-04 (L3 rare events & false starts)

### Completed
- Implemented L3 from `docs/DIALOGUE_GENERATION_ACTION_PLAN.md` on branch
  `feat/dialogue-gen-stage4-l3` (stacked on `feat/dialogue-gen-stage4-l1`,
  PR #58 → L1 branch). Plan: `docs/superpowers/plans/2026-07-04-l3-rare-events.md`.
  Commit+push per task with per-task review checkpoints (user requirement).
- L1 PR #56-style note: PR #57 for L1 was opened earlier this session (CI green,
  stacked on #56); user merges the chain manually after #55/#56.
- Review fix along the way: corrected the sentence-boundary regex comment
  (abbreviation dots can split — accepted for a ~0.17% cosmetic event).

### Changed files
- Modified: `app/core/reply_flavor.py` (roll_rare_event/apply_rare_event +
  verdict/filler pools), `app/handlers/_helpers.py` (`reply_humanized_sequence`,
  `reply_humanized` delegates), `app/handlers/learning.py` (event roll at the
  generated-reply send site), `app/config/{registry,settings,runtime_state}.py`
  (3 knobs + `rare_events_today` daily budget, pruned in `forget_chat`),
  `.env.example`, `README.md`, `docs/ARCHITECTURE.md`,
  `docs/DIALOGUE_GENERATION_ACTION_PLAN.md`, tests (`test_reply_flavor`,
  `test_handlers` incl. `_fake_message.answer` + real cap-method binding,
  `test_runtime_state`, `test_runtime_config`, `test_bot_messages`,
  `test_fallback_phrases`).

### Audit findings updated
- No new security findings. No new data stored; `rare_events_today` is
  in-memory (chat_id → (day, count)), pruned with the chat state. Events only
  reshape already-generated reply text; fallback phrases and eval untouched.

### Tests/checks run
- `unittest discover tests` — 616 tests OK (598 after L1).
- ruff, `mypy app/` (strict) — clean.
- `bandit -r app tools main.py` — 0 medium/high (Low B311 baseline).

### Not run / limitations
- `pip-audit` — no dependency changes; last clean run earlier today.
- Perceived quality of rare events (frequency feel, filler naturalness) is
  judgeable only in the live chat.

### Remaining work
- Merge chain: #55 → #56 → #57 (L1) → #58 (L3), bases collapse automatically.
- L2 per-user quirks — last Stage 4 item; requires a dedicated privacy review
  before implementation.
