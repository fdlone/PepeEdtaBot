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
| N6 | Info | Throttle-notify reply is itself unthrottled (1:1 amplification only). Docker HEALTHCHECK runs as root (no `USER`) and `sqlite3.connect` could create a root-owned file in edge cases. | **Open (accepted)** — low impact; revisit if throttling scope grows or healthcheck changes. |
| N7 | Process | CLAUDE.md/AGENTS.md referenced `PROJECT_AUDIT.md` / `PROJECT_AUDIT_CODEX.md`, archived on 2026-06-29. | **Fixed** — both now point at `docs/audits/` (this structure). |

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

- N6 items (accepted, low): throttle-notify amplification; root HEALTHCHECK.
- Deferred from 2026-06-29 audit (only if scaling): L3 metrics, P1 single-lock
  DB ceiling, A4 multi-instance state.
- S3 stays as-is per user decision.
