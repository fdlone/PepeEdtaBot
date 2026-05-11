# Project Audit Action Plan

Audit date: 2026-05-11  
Repository: `PepeEdtaBot`  
Branch audited: `main`  
Mode: audit-only; no source-code fixes were made.

## Goal

Make the project easier to maintain, safer to operate, cleaner for future development, and more predictable for feature expansion.

The plan focuses on closing audit findings without changing business logic unless an item explicitly requires a later implementation decision.

## Audit scope

Checked:

- technology stack and architecture;
- project layering and module boundaries;
- configuration and environment files;
- dependency files and local virtual environment state;
- tests and local test run;
- documentation and audit documents;
- Docker and GitHub Actions;
- repository hygiene, ignored files, local artifacts, temporary files;
- security-sensitive patterns, secrets, logs, DB artifacts, Telegram integration boundaries.

Safe checks performed:

- file inventory via `rg --files` and PowerShell listing;
- git status, tracked/untracked file checks, ignore checks;
- text search for TODO/FIXME/HACK and secret-like identifiers;
- local unit test run: `.\.venv\Scripts\python.exe -m unittest discover tests`;
- local package inventory: `.\.venv\Scripts\python.exe -m pip list`;
- read-only review of `README.md`, `docs/ARCHITECTURE.md`, `PROJECT_AUDIT.md`, `PROJECT_AUDIT_CODEX.md`, Docker/CI files and main Python modules.

Checks not performed:

- `ruff` and `mypy`: not installed in current `.venv`;
- Docker build: would require Docker daemon and may use local build context;
- live Telegram smoke: would call external Telegram APIs and send real messages;
- production DB data audit: local DB contents were not opened or inspected;
- dependency vulnerability scan: would require up-to-date external advisory data.

## Important note about `PROJECT_AUDIT.md`

`PROJECT_AUDIT.md` was reviewed but intentionally not modified.

Discrepancies with current state:

- It says the backlog is empty, but this audit found new P1/P2/P3 maintenance items.
- Section `0.1` still says all `200` tests are green; current local run reports `199`.
- The Docker snippet in section `9.3` shows `ENTRYPOINT ["/app/docker-entrypoint.sh"]`, while the actual `Dockerfile` uses `/usr/local/bin/docker-entrypoint.sh`.
- Historical references to branches containing `codex` are valid history, but new branch names must not use `codex`.
- The main architecture description is broadly correct, but docs around test counts and some security wording should be synchronized.

## Priorities

- P0: critical; should be fixed first.
- P1: important; should be fixed before active feature expansion.
- P2: maintainability and quality improvements.
- P3: optional but useful improvements.

No P0 issues were found.

## Action items

### AUD-001 — Exclude local artifacts from Docker build context

Priority: P1  
Category: devops / filesystem / security  
Recommended execution order: 1
Status: Deferred

Status note:
User clarified that this repository checkout is a local working/test environment,
not the production Docker build environment. Local Docker build from this folder
is not needed and is not possible on the current machine. No `.dockerignore`
change was made. Revisit only if local Docker builds from this workspace become
part of the workflow.

Problem description:
`.dockerignore` does not exclude `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg`, or broad local backup/temp patterns. Docker build context can therefore include local DB copies and screenshots.

Why it matters:
Local DB copies may contain sensitive or private data. Even if not committed to git, they can still be sent to the Docker daemon and copied into an image by `COPY . .`.

Recommended fix:
Add explicit patterns to `.dockerignore`, for example `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg`, `*.sqlite`, backup/archive/temp patterns as appropriate.

Affected files/modules:
`.dockerignore`

Risks:
Low. Overly broad ignores could accidentally exclude files needed at runtime, so keep patterns targeted.

Acceptance criteria:
Docker build context excludes local DB copies, temp test folders, screenshots, `.env`, virtualenvs and caches. Docker image still builds.

Safe without changing business logic:
Yes.

### AUD-002 — Decide storage policy for `db_prod_copy/`

Priority: P1  
Category: security / filesystem  
Recommended execution order: 2
Status: Done

Status note:
User chose to keep `db_prod_copy/` in the local working/test checkout and make
the local-only policy explicit. Added `db_prod_copy/` to `.gitignore`. No files
were deleted, moved, renamed, or inspected for data contents. `.dockerignore`
was not changed because local Docker builds from this workspace are not part of
the workflow on the current machine.

Problem description:
`db_prod_copy/markov.db` and companion WAL/SHM files exist inside the repo workspace. They are ignored by file extension but look like production data.

Why it matters:
Production DB copies should not live in a working tree by default. They increase accidental disclosure risk and can pollute Docker build context unless ignored there too.

Recommended fix:
Move production DB snapshots to a secure location outside the repository, or document a local-only workflow and explicitly ignore the folder.

Affected files/modules:
`db_prod_copy/`, `.gitignore`, `.dockerignore`, possibly developer docs.

Risks:
Medium. The copy may be needed for migration verification or backup. Do not delete without manual confirmation.

Acceptance criteria:
Production DB copies are either outside the repo workspace or explicitly managed as ignored local artifacts. No Docker context leakage.

Safe without changing business logic:
Yes, if only ignore/docs/storage policy changes are made.

### AUD-003 — Synchronize local `.env` with `.env.example`

Priority: P2  
Category: devops / configuration  
Recommended execution order: 3
Status: Needs decision

Status note:
User chose report-only handling for now. Local `.env` is a secret-bearing,
git-ignored working file and was not modified. Missing keys were identified by
name only; secret values were not printed or changed. Manual follow-up is needed
if local startup should use the current full configuration.

Problem description:
The local `.env` appears outdated by key list and comments. It lacks visible current keys such as `PIVO_HMAC_SECRET`, `PIVO_ENCRYPTION_SECRET`, `MAX_REPLY_TOKENS`, `REPETITION_PENALTY_STRENGTH`, `LOG_LEVEL`, and `BOT_TEXT_ALIASES`.

Why it matters:
The current app requires `/pivo` secrets. An outdated `.env` can prevent local startup or cause confusing runtime behavior.

Recommended fix:
Manually merge `.env.example` into `.env`, preserving real local secret values. Do not commit `.env`.

Affected files/modules:
Local `.env`, `.env.example` as reference.

Risks:
Medium. Mishandling can overwrite real secrets. Make a secure backup outside git before editing.

Acceptance criteria:
`load_settings()` succeeds with the local `.env`; all required keys are present.

Safe without changing business logic:
Yes.

### AUD-004 — Recreate or resync local `.venv`

Priority: P2  
Category: dependencies / devops  
Recommended execution order: 4
Status: Done

Status note:
User chose to install dev dependencies into the existing local `.venv`.
Ran `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`.
No dependency files were changed. The first sandboxed attempt timed out; the
approved rerun completed successfully and installed/synchronized locked runtime
packages plus `ruff` and `mypy`. Verification passed:
`ruff check app/ tests/`, `mypy app/`, and `unittest discover tests -v`.

Problem description:
Local `.venv` differs from `requirements.lock` and does not contain `ruff` or `mypy`.

Why it matters:
Local tests do not run on the same dependency versions as CI/Docker. Linters cannot be run locally.

Recommended fix:
Recreate `.venv` or run `pip install -r requirements-dev.txt` in a clean virtual environment.

Affected files/modules:
Local `.venv`, `requirements-dev.txt`, `requirements.lock`.

Risks:
Low. A local environment rebuild can disrupt local-only experiments.

Acceptance criteria:
`python -m pip list` matches lock/runtime expectations; `python -m ruff check app/ tests/`, `python -m mypy app/`, and `python -m unittest discover tests -v` run locally.

Safe without changing business logic:
Yes.

### AUD-005 — Resolve `/pivo` group versus DM behavior

Priority: P2  
Category: architecture / code-quality / docs  
Recommended execution order: 5
Status: Done

Status note:
User chose the group-only contract: the bot is created for group chats and
private chat command handling is not an active feature. Added `GroupOnly()` to
all `/pivo*` handlers, added a regression test that locks the router
registration, and documented that `/pivo*` commands are available only in
groups/supergroups.

Problem description:
`/pivo` handlers are available without `GroupOnly()`, while the product is documented as a group-chat bot.

Why it matters:
Unclear command scope makes future behavior harder to reason about and test. Private chat subscriptions can create data rows with private-chat hashes.

Recommended fix:
Choose one contract:

- group-only: add `GroupOnly()` to `/pivo*` handlers and tests for private chat rejection;
- DM-supported: document private-chat behavior and add tests that lock it in.

Affected files/modules:
`app/handlers/pivo.py`, `tests/test_handlers.py`, `tests/test_filters.py`, `README.md`.

Risks:
Medium. Changing scope can alter user-visible behavior.

Acceptance criteria:
Product contract is explicit in code and docs; regression tests cover private chat behavior.

Safe without changing business logic:
No, if adding `GroupOnly()`. Yes, if only documenting current behavior.

### AUD-006 — Fix README crypto/privacy drift

Priority: P2  
Category: docs / security  
Recommended execution order: 6
Status: Done

Status note:
User chose the minimal factual correction. README now describes the actual
`/pivo` scheme: HMAC-SHA256 under `PIVO_HMAC_SECRET`, Fernet key derived as
SHA-256 of `PIVO_ENCRYPTION_SECRET`, and HKDF only for `chat_id` log masking.
No code or dependency changes were made for this item.

Problem description:
`README.md` says `/pivo` uses HKDF domain labels for HMAC/encryption and warns that DEBUG can print raw `chat_id`. Current code uses direct HMAC plus `sha256(PIVO_ENCRYPTION_SECRET)` for `/pivo`, and learning logs mask `chat_id`.

Why it matters:
Security documentation must describe the actual implementation. Incorrect crypto docs can lead to bad operational decisions.

Recommended fix:
Update README to say:

- `/pivo` indexes IDs with HMAC-SHA256 under `PIVO_HMAC_SECRET`;
- `/pivo` encrypts payload using Fernet key derived as `sha256(PIVO_ENCRYPTION_SECRET)`;
- HKDF is currently used for log masking in `app/log_masking.py`;
- known logging limitations, if any, are precise.

Affected files/modules:
`README.md`

Risks:
Low. Documentation-only.

Acceptance criteria:
README matches `pivo.py` and `app/log_masking.py`.

Safe without changing business logic:
Yes.

### AUD-007 — Fix quickstart dependency instructions

Priority: P2  
Category: docs / dependencies  
Recommended execution order: 7
Status: Done

Status note:
User chose to split runtime and development setup. README now keeps
`requirements.txt` for minimal local runtime startup and explicitly recommends
`requirements-dev.txt` for development/local checks because it installs
`requirements.lock` plus CI tools. No dependency files were changed.

Problem description:
Quickstart installs `requirements.txt`, while CI/Docker use `requirements.lock` through `requirements-dev.txt`.

Why it matters:
New developers can unknowingly run a different dependency set than CI.

Recommended fix:
Split README instructions into runtime install and development install. Recommend `pip install -r requirements-dev.txt` for contributors.

Affected files/modules:
`README.md`

Risks:
Low.

Acceptance criteria:
A new developer can follow README and run the same checks as CI.

Safe without changing business logic:
Yes.

### AUD-008 — Remove or de-emphasize brittle test-count numbers

Priority: P3  
Category: docs  
Recommended execution order: 8
Status: Done

Status note:
User chose to remove the exact test-count metric from stable architecture docs.
`docs/ARCHITECTURE.md` now uses stable wording instead of the stale
`208 unit-тестов` count. `PROJECT_AUDIT.md` was not edited.

Problem description:
`docs/ARCHITECTURE.md` says `208 unit-тестов`, but current local run reports `199`. `PROJECT_AUDIT.md` also contains historical inconsistent counts.

Why it matters:
Stale metrics reduce trust in docs.

Recommended fix:
Replace exact counts in stable docs with approximate wording, or update all docs whenever test count changes.

Affected files/modules:
`docs/ARCHITECTURE.md`, optionally `README.md`. Do not edit `PROJECT_AUDIT.md` unless separately requested.

Risks:
Low.

Acceptance criteria:
Architecture docs no longer contradict current test output.

Safe without changing business logic:
Yes.

### AUD-009 — Clarify `AGENTS.md` repository policy

Priority: P3  
Category: git / repository hygiene  
Recommended execution order: 9
Status: Done

Status note:
User chose local-only policy for `AGENTS.md`. Verification showed that
`AGENTS.md` is already not tracked by `git ls-files` and is ignored by
`.gitignore`. Attempted `git rm --cached AGENTS.md` found no tracked path to
remove, so no index change was needed.

Problem description:
`AGENTS.md` is tracked, but `.gitignore` lists it under local agent instructions.

Why it matters:
Tracked ignored files confuse contributors: edits still show up even though the file appears ignored.

Recommended fix:
Decide whether `AGENTS.md` is a repository document. If yes, remove it from `.gitignore`. If no, remove it from the index in a separate intentional change.

Affected files/modules:
`AGENTS.md`, `.gitignore`.

Risks:
Medium if removing from index because repo instruction behavior may change.

Acceptance criteria:
Git policy matches actual tracked state.

Safe without changing business logic:
Yes.

### AUD-010 — Decide what to do with untracked screenshots

Priority: P2  
Category: git / filesystem  
Recommended execution order: 10
Status: Done

Status note:
User chose to keep local screenshot workflow and add a targeted ignore pattern.
Added `Screenshot_*.jpg` to `.gitignore`. No screenshot files were deleted,
moved, renamed, or added to git.

Problem description:
`Screenshot_1.jpg` and `Screenshot_2.jpg` are untracked and not ignored.

Why it matters:
They can be accidentally committed or included in Docker build context.

Recommended fix:
Either move/delete them after confirmation, or add a targeted ignore pattern if screenshots are local-only.

Affected files/modules:
`Screenshot_1.jpg`, `Screenshot_2.jpg`, `.gitignore`, `.dockerignore`.

Risks:
Low to medium. The screenshots may contain useful or sensitive evidence.

Acceptance criteria:
`git status` no longer shows accidental screenshot artifacts, or they are intentionally tracked with documentation.

Safe without changing business logic:
Yes.

### AUD-011 — Add Docker build smoke check

Priority: P3  
Category: tests / devops  
Recommended execution order: 11
Status: Done

Status note:
User chose to add a CI Docker build smoke check and fix the related Docker
documentation drift immediately. Added a GitHub Actions `docker-build` job that
runs `docker build -t pepe-bot:latest .` without starting the bot or requiring
secrets. Updated README Docker commands to use `pepe-bot` /
`pepe-bot:latest`, matching `compose.yaml`. Local Docker build was not run on
this machine.

Problem description:
CI runs lint, type check and unit tests, but does not verify that the Docker image builds.

Why it matters:
Dockerfile drift can break deployment while tests stay green.

Recommended fix:
Add a CI job or manual script for `docker build` after `.dockerignore` cleanup. Do not start the bot or call Telegram APIs.

Affected files/modules:
`.github/workflows/ci.yml`, `Dockerfile`, `.dockerignore`.

Risks:
Medium. Docker builds can slow CI and require cache tuning.

Acceptance criteria:
Docker image build is verified without needing production secrets.

Safe without changing business logic:
Yes.

### AUD-012 — Refactor `db.py` stats and clear helpers when touching DB code

Priority: P3  
Category: architecture / code-quality  
Recommended execution order: 12
Status: Deferred

Status note:
User chose to defer this refactor until the next real DB-related feature or
bugfix. No DB code was changed for this item. This follows the original plan
wording: refactor stats/clear helpers when touching DB code, not as a standalone
SQL refactor during repository hygiene work.

Problem description:
`db.py` still contains dense cross-domain SQL in `get_stats`, `clear_chat`, and `save_message_and_update_model`.

Why it matters:
The file is no longer a god-file, but future DB features still need to touch repetitive SQL blocks.

Recommended fix:
When making a DB-related feature, extract small internal helpers or repository methods for stats/clear operations. Keep public API stable.

Affected files/modules:
`db.py`, `app/repositories/*`, tests around DB stats and clear.

Risks:
Medium. DB refactors can introduce subtle behavior changes.

Acceptance criteria:
Behavior stays identical; existing tests pass; new helper tests cover moved logic.

Safe without changing business logic:
Yes, if done as pure refactor with tests.

### AUD-013 — Add explicit local environment verification docs

Priority: P3  
Category: docs / devops  
Recommended execution order: 13

Problem description:
README gives commands but does not provide a concise "verify your local dev env matches CI" checklist.

Why it matters:
The current `.venv` drift shows that developers can run partial checks unknowingly.

Recommended fix:
Add a short local verification section: Python version, install dev deps, run ruff/mypy/unittest, optional `pip list`/lock sanity.

Affected files/modules:
`README.md`

Risks:
Low.

Acceptance criteria:
A new developer can set up and validate a clean environment without guessing.

Safe without changing business logic:
Yes.

## Quick wins

- Add `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg` to `.dockerignore`.
- Decide whether screenshots should be deleted, ignored, or documented.
- Update `README.md` quickstart to use `requirements-dev.txt` for development.
- Correct README `/pivo` crypto wording.
- Update `docs/ARCHITECTURE.md` test count or remove exact count.
- Recreate local `.venv` from `requirements-dev.txt`.
- Sync local `.env` with `.env.example`.

## File deletion candidates

Do not delete anything without manual confirmation.

| Path | Reason | Confidence | Verify before deletion | Safe now? |
|---|---|---|---|---|
| `Screenshot_1.jpg` | Untracked screenshot, not referenced by code/docs. | Medium | Confirm it is not needed for issue evidence or documentation. | Requires manual confirmation. |
| `Screenshot_2.jpg` | Untracked screenshot, not referenced by code/docs. | Medium | Confirm it is not needed for issue evidence or documentation. | Requires manual confirmation. |
| `.test_tmp/` | Old local test temp folders, ignored by git. | High | Confirm no current process/test uses it. | Requires manual confirmation. |
| `__pycache__/` | Generated Python cache. | High | Confirm no process is running. | Usually safe, but not part of this audit task. |
| `markov.db` | Local SQLite DB in repo root; ignored but may be stale. | Medium | Confirm it is not the only useful local test/training dataset. | Requires manual confirmation. |
| `db_prod_copy/` | Looks like production DB copy; sensitive local artifact. | Medium | Confirm backup, retention and migration-test needs. Prefer secure relocation over deletion. | Requires manual confirmation. |
| `.idea/` | Local IDE metadata. | Medium | Confirm user does not rely on project-level IDE config. | Requires manual confirmation. |
| `.venv/` | Recreateable and currently out of sync. | Medium | Confirm no local-only packages are needed. | Requires manual confirmation. |

## Recommended execution sequence

1. Fix critical repository hygiene/security exposure risks:
   update `.dockerignore`, decide policy for `db_prod_copy/`, screenshots and local DB files.
2. Clean confusing local artifacts:
   handle `.test_tmp/`, `Screenshot_*.jpg`, root `markov.db`, `.venv` drift.
3. Dependency and environment cleanup:
   recreate `.venv`, install `requirements-dev.txt`, run `ruff`, `mypy`, `unittest`.
4. Product/architecture decision:
   decide `/pivo` private-chat behavior and encode it in code or docs.
5. Documentation sync:
   fix README crypto/privacy, quickstart, local dev verification; update architecture test metric.
6. DevOps improvements:
   add Docker build smoke after Docker context cleanup.
7. Optional refactors:
   reduce repetitive SQL in `db.py` when DB code is next touched.
