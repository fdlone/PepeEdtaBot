# Proposal: bump-aiogram-and-checkout

## Why

The runtime lock pins `aiogram==3.29.0`, which PyPI has yanked for a real
defect — "severe (exponential) slowdown when validating nested unions"
(RichBlock entity parsing). A yanked pin also breaks fresh resolver-based
installs. Separately, CI's `actions/checkout@v4` runs on Node 20, which GitHub
is deprecating on hosted runners. Both were queued as background hygiene in
`docs/v2/00_STATUS.md`.

## What Changes

- `requirements.lock`: `aiogram==3.29.0` → `aiogram==3.30.0` (includes the
  3.29.1 fix for the yank reason plus Bot API 10.2 support; no breaking
  changes documented for 3.29.x→3.30.0). Transitive pins refreshed only if the
  resolver actually moves them. The range in `requirements.txt`
  (`>=3.7.0,<4.0.0`) already admits it and stays untouched.
- `.github/workflows/ci.yml`: `actions/checkout@v4` → `@v7` in both jobs.
  v5 fixed the Node 20 deprecation (Node 24); v6/v7 changes (credentials in a
  separate file, fork-PR block for `pull_request_target`/`workflow_run`) do
  not apply to this workflow — it triggers on `push: main` + `pull_request`
  only and never pushes from CI.

## Capabilities

### New Capabilities

<!-- none: dependency and CI hygiene, no behavior contract changes -->

### Modified Capabilities

<!-- none -->

## Impact

- **Code**: none — lock file and workflow file only.
- **Runtime**: aiogram minor upgrade; validated by the full test suite, the
  bot import/startup check and `pip_audit` (the same checks CI runs).
- **Risk**: aiogram minor releases have broken nothing here since 3.7; the
  suite exercises handlers/middleware against the installed version. Rollback
  is reverting the lock line.
