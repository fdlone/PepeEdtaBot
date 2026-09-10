# Tasks: bump-runtime-pins-and-python-base

## 1. Runtime lock

- [x] 1.1 Refresh `requirements.lock` by its documented strategy (clean venv on 3.14 + `pip freeze`), restore the hand-written header, exclude `setuptools`; verify `git diff requirements.lock` moves exactly the seven pins named in the proposal and nothing else
- [x] 1.2 Verify `pymorphy3` and `pymorphy3-dicts-ru` are byte-identical in the diff — the `generation_hash` claim in the proposal rests on it, and a move here is a stop-and-report, not a merge conflict to resolve
- [x] 1.3 Install the refreshed lock into `.venv` and verify the full suite, `ruff check app/ tests/ tools/ main.py` and `mypy app/` are green
- [x] 1.4 Verify `python -m tools.generation_hash --synthetic --check` still passes — the declared-in-advance hash claim, checked against fact
- [x] 1.5 Verify `PYTHONUTF8=1 python -m pip_audit -r requirements.lock` reports no known vulnerabilities (the CI gate)
- [x] 1.6 Verify `import main` succeeds against aiogram 3.31.0 with no deprecation errors

## 2. Base image

- [x] 2.1 `Dockerfile`: `python:3.14.0-slim` → `python:3.14.7-slim`; verify the pin is the only edit in the file's diff

## 3. CI action

- [x] 3.1 `.github/workflows/ci.yml`: `actions/setup-python@v6` → `@v7` in the one job that uses it; verify no `pip-install` input exists anywhere in the workflow (the only thing v7 removed)

## 4. Integration check

- [x] 4.1 Verify `python -m tools.eval --smoke` runs clean on the refreshed lock
- [ ] 4.2 **Not runnable locally — no Docker on this machine (`docker` absent from PATH and no Docker Desktop install); CI's `docker-build` job runs exactly this on the PR.** Verify `docker build` succeeds on 3.14.7-slim and the image smoke-test passes — imports live, migrations matching by name, build stamp parseable. This is what settles the lock-frozen-on-3.14.1 vs image-ships-3.14.7 gap noted in the proposal; if the resolver would have picked different wheels on 3.14.7, this is where it shows
- [ ] 4.3 Verify CI is green on the PR across the whole 3.12/3.13/3.14 matrix — also the live proof for the `setup-python@v7` bump

## 5. Runbook note

- [x] 5.1 `README.md` — the «Разработка и тесты» section, not `docs/OPERATIONS.md` as first proposed: `pip-audit` is a dev/CI gate and OPERATIONS.md is a runtime runbook, so the note goes where the other local check commands already live. Add the `PYTHONUTF8=1` prerequisite with the one-line reason (Cyrillic UTF-8 header read under cp1251); verified by running the written command on a clean shell — no known vulnerabilities

## 6. Close-out

- [ ] 6.1 Archive this change after merge
