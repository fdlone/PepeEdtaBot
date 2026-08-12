# Tasks: bump-aiogram-and-checkout

## 1. Dependency bump

- [x] 1.1 `requirements.lock`: `aiogram==3.30.0`; refresh transitive pins only if the resolver moves them; keep the hand-written lock header (pip freeze erases it) — resolver moved nothing else, one-line edit
- [x] 1.2 Install the new pin into the venv and run the full test suite + ruff + mypy — 1095 tests OK, ruff clean, mypy clean
- [x] 1.3 `python -m pip_audit -r requirements.lock` clean (the CI gate) — no known vulnerabilities (locally under PYTHONUTF8=1: the audit tool reads the lock's Russian header as cp1251 on Windows; CI's Ubuntu is unaffected)
- [x] 1.4 Bot startup check: importing `main` / building the app wires aiogram 3.30.0 without deprecation errors

- [x] 1.5 Refresh the remaining lock pins via the documented strategy (clean venv + pip freeze, header preserved, setuptools excluded as before); full suite + ruff/mypy (latest, as CI resolves) + pip_audit + import check green on the result

## 2. CI action

- [x] 2.1 `actions/checkout@v4` → `@v7` in both jobs of `.github/workflows/ci.yml`
- [x] 2.2 CI green on the PR (this is also the live proof for the checkout bump)

## 3. Close-out

- [x] 3.1 Drop the hygiene line from `docs/v2/00_STATUS.md` next-session list — deferred on purpose: the phase-5 branch rewrote that exact list and both branches are unmerged; a one-line edit after both land avoids a guaranteed conflict
- [x] 3.2 Archive this change after merge
