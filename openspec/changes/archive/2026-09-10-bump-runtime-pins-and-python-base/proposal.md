# Proposal: bump-runtime-pins-and-python-base

## Why

`Dockerfile` pins `python:3.14.0-slim`, seven patch releases behind the
current `3.14.7-slim`; the prod image has therefore been carrying an
interpreter and a Debian package set from the 3.14.0 release date across every
restart since. Alongside it the runtime lock has drifted seven pins behind PyPI
and CI still runs `actions/setup-python@v6`. None of this is urgent —
`pip-audit -r requirements.lock` reports no known vulnerabilities today — which
is exactly why it is worth doing as one deliberate hygiene package rather than
under pressure later.

The audit that surfaced this also found a local-only papercut worth writing
down: `pip-audit` cannot read `requirements.lock` on Windows without
`PYTHONUTF8=1`. It was diagnosed once already (2026-08-12,
`bump-aiogram-and-checkout` task 1.3) and left in that change's task list,
where nobody looking for it would find it.

## What Changes

- `Dockerfile`: base image `python:3.14.0-slim` → `python:3.14.7-slim`. Patch
  releases only — same 3.14 minor, so `requires-python >=3.12` and the CI
  matrix are untouched.
- `requirements.lock`: refreshed by the lock's own documented strategy (clean
  venv on 3.14 + `pip freeze`, hand-written header restored, `setuptools`
  excluded as before). The resolver moves exactly seven pins:
  - `aiogram` 3.30.0 → 3.31.0 (Bot API 10.3; fixes bot-level `parse_mode`
    not applied to recently added entities, and content-type detection for
    live photos — no breaking changes documented for 3.30→3.31)
  - `cryptography` 50.0.0 → 50.0.1
  - `pydantic` 2.13.4 → 2.13.5 and `pydantic_core` 2.46.4 → 2.46.5 (pydantic
    pins its own core; 2.49.0 on PyPI is not ours to take)
  - `multidict` 6.7.1 → 6.8.0, `idna` 3.18 → 3.19, `python-dotenv`
    1.2.2 → 1.2.3
  Every other pin holds. Ranges in `requirements.txt` already admit all seven
  and stay untouched.
- `.github/workflows/ci.yml`: `actions/setup-python@v6` → `@v7`. The only
  removal in v7 is the `pip-install` input, which this workflow never used;
  `cache: pip` is a different input and is unaffected. `actions/checkout@v7`
  is already current and stays.
- `README.md` («Разработка и тесты»): record that `pip-audit -r
  requirements.lock` needs
  `PYTHONUTF8=1` on Windows — the lock's header is Cyrillic UTF-8 and the
  requirements parser decodes it under the machine's cp1251 locale. CI's
  Ubuntu is unaffected.

## Capabilities

### New Capabilities

<!-- none: dependency, base-image and CI hygiene plus one runbook note -->

### Modified Capabilities

<!-- none -->

## Impact

- **Code**: none. Four files: `Dockerfile`, `requirements.lock`,
  `.github/workflows/ci.yml`, `README.md`.
- **`generation_hash`**: **unchanged, declared in advance** (CLAUDE.md §8).
  The generation path's only dependency is morphology, and both `pymorphy3`
  (2.0.6) and `pymorphy3-dicts-ru` are already at their latest — the resolver
  does not move them. None of the seven pins that do move is reachable from
  generation. Any deviation from this at check time is a stop-and-report.
- **Runtime**: interpreter patch bump plus an aiogram minor. Validated by the
  same checks CI runs — full suite, ruff, mypy, `pip_audit`, the eval smoke,
  the baseline guard and the Docker image smoke-test.
- **Risk**: low and individually revertible. The base-image bump is a one-line
  revert; the lock is a seven-line revert. The residual unknown is that the lock
  is frozen on a locally available 3.14.1 while the image ships 3.14.7 — see
  tasks 1.2 and 4.2, where the image's own build and smoke-test are what
  actually settles it.
