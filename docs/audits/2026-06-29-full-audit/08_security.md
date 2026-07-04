# 08 — Security

> Independent audit, source-only. Tooling: Bandit (the project's own CI gate), pip-audit, plus manual review through the Trail of Bits lenses (`static-analysis`, `insecure-defaults`, `supply-chain`, `sharp-edges`, `agentic-actions`). Findings classified Critical / High / Medium / Low with file:line evidence, exploitation scenario, and fix. No production code modified.
>
> Cross-refs: config detail in [13_configuration.md](13_configuration.md); dependency pinning in [06_dependency_graph.md](06_dependency_graph.md); items feed [15_technical_debt.md](15_technical_debt.md) / [17_risk_register.md](17_risk_register.md).

## 0. Summary

**Overall posture: strong.** This is a long-polling Telegram bot with a SQLite backend and no HTTP server, so the classic web attack surface (webhook handler, SSRF, CSRF, XSS, deserialization, template injection) is **absent by design**. The dangerous-primitive scan is clean: **no `subprocess`, `os.system`, `eval`, `exec`, `pickle`, `marshal`, or `yaml.load`** anywhere in `app/`. All SQL is parameterized. Secrets come from the environment with validation. Crypto uses the `cryptography` library and `hmac`/`hkdf` correctly.

| Severity | Count | Items |
|---|---|---|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 2 | S1 (CI `safety` step likely broken), S2 (insecure-default placeholder secrets pass validation) |
| Low | 4 | S3 (Fernet key derivation not a KDF), S4 (admin lookup not cached — DoS-adjacent), S5 (f-string DDL in migration), S6 (silent blind-except) |
| Info | — | Long-poll only; non-crypto `random` by design; chat_id log masking is exemplary |

## 1. Tooling results

### 1.1 Bandit — clean at the gate
`bandit -r app tools main.py -x tests --severity-level medium --confidence-level medium` (the exact CI invocation):
```
No issues identified.   (Medium: 0, High: 0)
```
At the **Low** threshold Bandit reports 24 issues, **all benign**: 23 × `B311` (non-crypto `random` — by design for text/gameplay variety; see §3) and 1 × `B110` (`try/except/pass` at `app/handlers/_helpers.py:25`, also flagged as Q1 in [11_code_quality.md] → S6). No action required on the Bandit set.

### 1.2 Semgrep (ToB `static-analysis`) — 4 findings, all false positives
`uvx semgrep scan --config p/python --config p/security-audit --config p/secrets` (237 rules, 66 files; auto-config skipped because it requires telemetry). **4 findings, all the same rule** `python-logger-credential-disclosure` (CWE-532, WARNING, likelihood LOW). Each was verified (fp-check) and is a **false positive** — the rule matched the literal words "tokens"/"context" in format strings, but the interpolated values are **counts/booleans, never content or secrets**, all at `logger.debug`, with `chat_id` masked:

| Location | Logged value | Verdict |
|---|---|---|
| `app/core/markov.py:1020` | generation trace = attempt/jump **counts**, enums, booleans | FP — n-gram "tokens" are counts, not auth tokens |
| `app/core/response_generator.py:172` | `tokens=len(candidate.split())`, `context=bool(...)` | FP — count + boolean |
| `app/handlers/learning.py:149` | `len`, token **count** | FP — counts |
| `app/handlers/learning.py:226` | `context_tokens=len(context_tokens)` | FP — count |

No user message text or credentials are logged anywhere these fire. Confirms the §3 logging assessment. (One `tools/eval_generation.py:161` taint fixpoint timeout on `hardcoded-token` — a tool script, not production; no finding emitted.)

### 1.3 pip-audit — runtime is clean (the 2 hits are dev-only)
`pip-audit` against the venv reports:
```
joserfc 1.6.5  CVE-2026-48990     fix 1.6.7
msgpack 1.1.2  GHSA-6v7p-g79w-8964 fix 1.2.1
```
**Neither package is a runtime dependency.** `requirements.lock` runtime closure is `aiogram 3.29.0`, `aiosqlite 0.22.1`, `python-dotenv 1.2.2`, `cryptography 49.0.0` (+ aiogram transitive: `aiohttp`, `aiofiles`, etc.). `joserfc`/`msgpack` are transitive deps of the **dev** security tooling (`safety`/`pip-audit`) installed in the same venv. **Production image is unaffected.** Optional hygiene: bump the dev tools. (`pip-audit -r requirements.lock` could not be run directly — it crashes on the file's Cyrillic header under the Windows cp1251 locale; run it in CI/Linux or pass `--encoding utf-8`.)

## 2. Findings

### S1 — CI `safety check` step is likely broken/fragile · **Medium** · `.github/workflows/ci.yml:43`
`safety check -r requirements.lock` uses the **deprecated** Safety v2 CLI. Safety ≥3 (pinned `safety>=3.6.0` in `requirements-dev.txt`) replaced it with `safety scan` and now generally **requires authentication / an account**, so this step may be silently failing or erroring in CI — meaning the intended dependency-vulnerability gate may not actually be running.
- **Impact:** false sense of dependency coverage; a vulnerable runtime dep could merge unnoticed.
- **Fix:** replace with `pip-audit -r requirements.lock` (already a dev dep, no account needed) or migrate to `safety scan` with a CI token. Confidence: **High** that the CLI is deprecated; **Medium** on whether CI currently errors (depends on Safety's grace handling).

### S2 — Placeholder secrets satisfy validation (insecure default) · **Medium** · `app/config/settings.py:90-95`, `.env.example`
Validation requires only `len(secret) >= 16`. The `.env.example` placeholders `change_me_to_a_long_random_hmac_secret` / `change_me_to_a_long_random_encryption_secret` are **longer than 16 chars and therefore pass**. A deployer who copies `.env.example`, sets a real `BOT_TOKEN`, but forgets to replace the pivo secrets gets a **running bot with predictable, public crypto secrets** protecting `/pivo` identity data.
- **Impact:** the HMAC/Fernet protecting encrypted user IDs/usernames would use a value committed in the public repo → `/pivo` PII is effectively unprotected for that deployment.
- **Why fail-open:** `BOT_TOKEN` fails closed (no default, bot won't start), but the pivo secrets fail *open* against the well-known placeholder.
- **Fix:** reject known placeholder values (e.g. anything starting with `change_me`) and/or add an entropy/distinctness check (`hmac_secret != encryption_secret`). The `.env.example` already documents "два разных значения" but the code does not enforce it. Confidence: **High.**

### S3 — Fernet key derived by single-pass SHA-256, not a KDF · **Low** · `app/domain/pivo.py:33-36`
```python
key = base64.urlsafe_b64encode(hashlib.sha256(encryption_secret.encode()).digest())
self._fernet = Fernet(key)
```
The encryption key is one unsalted SHA-256 of the secret. For a high-entropy server-side secret this is acceptable (no brute-forceable password involved), but it is **inconsistent** with `app/log_masking.py`, which correctly uses **HKDF-SHA256** with a domain label. The single-pass derivation also means HMAC and encryption keys are simple transforms of two independent secrets (fine, since the secrets differ).
- **Fix (consistency, not urgent):** derive the Fernet key via HKDF with an `info=b"pivo:fernet"` label, mirroring `log_masking`. Confidence: **High** on the observation; **Low** on real-world risk.

### S4 — `get_chat_administrators` called per admin-command, uncached · **Low** · `app/filters/admin_or_owner.py:24`
Every admin/owner-gated command performs a live Telegram API call to enumerate chat admins. The owner short-circuits first (good), but for non-owner admins this is an unbounded outbound call per invocation. Combined with command cooldowns it is not a practical DoS, but it adds latency and API-rate exposure.
- **Fix:** short-TTL cache of admin lists per chat. Confidence: **High.** (Also a [09_performance.md] item.)

### S5 — f-string-built DDL in migration · **Low / Info** · `app/migrations/005_drop_messages_text.py:27`
```python
await conn.execute(f"DROP INDEX IF EXISTS [{name}]")
```
The interpolated `name` comes from `PRAGMA index_list` (database-internal metadata), **not user input**, so this is not injectable in practice. Flagged only because it is the lone non-parameterized SQL string in the codebase. **No fix required**; bracket-quoting is already applied.

### S6 — silent blind `except Exception: pass` · **Low** · `app/handlers/_helpers.py:25`
The typing-indicator chat action is wrapped in `except Exception: pass` so a failed `send_chat_action` cannot block the reply (intentional and correct), but it swallows **all** exceptions with no log line.
- **Fix:** `logger.debug(...)` the exception. Same item as Q1 in [11_code_quality.md]. Confidence: **High.**

## 3. Areas reviewed and found clean (evidence)

- **Injection / SQL:** every `execute`/`executemany` uses `?` placeholders with tuple params (`app/infrastructure/database.py`, `app/repositories/*`). `_fetch_int(sql, params)` is only ever called with **literal** SQL constants and a parameterized `chat_id` (`database.py:347-361`). No string-concatenated user data in queries.
- **RCE / command exec / deserialization:** no `subprocess`, `os.system`, `eval`, `exec`, `pickle`, `marshal`, `yaml.load`, `__import__` of user data in `app/`.
- **Webhook / SSRF / CSRF / XSS:** none applicable — `main.py:98,121` uses `delete_webhook` + `start_polling` (long-poll); there is no inbound HTTP server and no outbound URL fetching of user-controlled targets.
- **Secrets in source / VCS:** `.env` is git-ignored; `*.db`, `data/`, `db_prod_copy/` are git-ignored; `markov.db` is **not** tracked (`git ls-files` confirms no `.db`). `.env.example` carries placeholders + a `secrets.token_urlsafe(32)` generation hint, not real values. (Supersedes the stale note that `markov.db` was committed.)
- **Sensitive logging:** `app/log_masking.py` masks `chat_id` via HKDF-SHA256 with fail-fast (`LogMaskingNotInitialized`) — **exemplary**. No message text, tokens, or raw user IDs are logged (`main.py:118` logs only the bot username at startup).
- **Authorization:** `is_admin_or_owner` checks `owner_id` then live chat-admin membership, restricts to group/supergroup, and **fails closed** on error (`admin_or_owner.py:16-28`). `/pivo` quota and identity use HMAC-hashed chat/user keys (privacy by design).
- **Crypto primitives:** `cryptography` Fernet (AES-CBC+HMAC) for `/pivo` data at rest; `hmac.new(..., sha256)` for identity hashing; HKDF for log masking. The 23 `random.*` uses are **non-security** (reply variety, typing delay) — correct choice; the one place needing unpredictability would be the secrets, which come from the environment.

## 4. ToB skill mapping

| Skill / lens | Applied to | Result |
|---|---|---|
| `static-analysis` (Bandit + Semgrep) | `app`, `tools`, `main.py` | Bandit clean at medium/medium (§1.1); Semgrep 4 findings all FP (§1.2) |
| `insecure-defaults` | `settings.py`, `.env.example` | **S2** (placeholder secrets pass validation) |
| `supply-chain` + pip-audit | `requirements.lock` / dev deps | Runtime clean; **S1** CI gate fragile; dev-tool CVEs only |
| `sharp-edges` | crypto + config API | **S2/S3** (length-only secret check; non-KDF key derivation) |
| `agentic-actions-auditor` | `.github/workflows/ci.yml` | No AI-agent actions present; standard `checkout@v4`/`setup-python@v5`, no `pull_request_target`, no untrusted-input→shell injection. Clean. |
| `fp-check` | pip-audit hits | Confirmed **false positives for production** (dev-only transitive deps) |

## 5. Prioritized security backlog

| Priority | Item | Effort | Note |
|---|---|---|---|
| **P1** | S1 — fix/replace CI `safety check` (use `pip-audit -r requirements.lock`) | XS | Restores the dependency gate |
| **P1** | S2 — reject placeholder/equal pivo secrets | S | Prevents fail-open PII exposure |
| **P3** | S3 — HKDF for Fernet key (consistency with log_masking) | S | Hardening |
| **P3** | S4 — cache chat-admin lookups | S | Perf + API-rate; see [09] |
| **P4** | S6 — log the swallowed exception | XS | Observability; = Q1 in [11] |
| **P4** | bump dev tools (`safety`/`pip-audit`) to clear joserfc/msgpack | XS | Dev hygiene |
