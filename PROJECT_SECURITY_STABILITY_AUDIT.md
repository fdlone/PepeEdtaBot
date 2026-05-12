# Security and Stability Audit

Date: 2026-05-12  
Branch: `audit-security-stability-review`  
Mode: audit plus dev-tooling update. Production bot code, tests, runtime configs, and deployment files were not changed. Dev-only security tools were added to `requirements-dev.txt` after the initial audit.

## 1. Executive Summary

Overall risk level: **MEDIUM** for current small-group production usage, **HIGH** before expanding to larger or less trusted chats.

No committed Telegram token, `.env`, or hardcoded production secret was found in tracked files. SQL access is parameterized, admin-only commands use `AdminOrOwner`, `/pivo*` is group-only, `/pivo` subscription data is HMAC-indexed and Fernet-encrypted, and the current unit/lint/type gates pass locally. Dependency scanning with Safety found current vulnerabilities in locked `cryptography==45.0.7`.

Top 5 risks:

| Rank | Risk | Severity | Why it matters |
|---|---|---:|---|
| 1 | `/pivo` explicit mentions have no target-count limit | HIGH | Any group user can create several high-fanout mention messages per day. |
| 2 | Locked `cryptography==45.0.7` is reported vulnerable by Safety | HIGH | Safety reports 3 CVEs affecting the locked version. |
| 3 | Long-running process keeps unbounded per-chat/per-user dictionaries | MEDIUM | Memory grows with every distinct chat/user/command seen until restart. |
| 4 | Novelty prefix filter fetches all normalized messages and builds an in-memory prefix set | MEDIUM | Large chats can cause RAM/CPU spikes during generation. |
| 5 | SQLite Markov tables and WAL can grow without retention/compaction policy | MEDIUM | Disk usage grows with continuous operation; WAL/journal handling is only implicit. |

Production readiness: acceptable for the current controlled chat if operators monitor disk, restarts, and errors. Before expanding usage, fix `/pivo` fanout limits, add bounded runtime-state cleanup, add DB growth policy, and harden Docker context hygiene.

## 2. Repository and Runtime Overview

| Item | Observed state |
|---|---|
| Bot framework | `aiogram v3`, long polling in `main.py` |
| Python | Local `.venv`: Python 3.14.0; Docker: `python:3.14.0-slim`; CI matrix: 3.12 / 3.13 / 3.14 |
| Storage | SQLite via `aiosqlite`; WAL enabled in `Database.init()` |
| DB path | Default `data/markov.db`; local root DB files are ignored by Git |
| Secrets | `.env` through `python-dotenv`; required: `BOT_TOKEN`, `PIVO_HMAC_SECRET`, `PIVO_ENCRYPTION_SECRET`; optional `OWNER_ID` |
| Deployment | Dockerfile + Compose; restart policy `unless-stopped`; no systemd files found |
| Background tasks | No custom background tasks or schedulers; aiogram polling owns the long-running loop |
| External services | Telegram Bot API only |
| Runtime mutable settings | `/set` updates in-memory `RuntimeState` only; reset on restart |

## 3. Findings Table

| ID | Severity | Area | File / Location | Finding | Evidence | Impact | Recommendation | Fix complexity |
|---|---|---|---|---|---|---|---|---|
| AUD-001 | HIGH | Telegram abuse / spam | `app/services/pivo_parser.py`, `app/services/pivo_service.py`, `app/handlers/pivo.py` | Explicit `/pivo` mentions are not capped. | Parser collects every entity/plain mention; service uses `explicit_mentions` directly; handler only applies daily call quota. | A regular user can send up to 3 high-fanout mention messages/day; admins up to 5; owner unlimited. | Add max explicit mentions per call, max subscriber mentions per call, and clear user feedback when truncated/rejected. | Low |
| AUD-002 | MEDIUM | Memory / long-running stability | `runtime_state.py`, `app/middlewares/throttling.py` | Runtime dictionaries are unbounded. | `last_reply_ts`, `learned_messages`, `recent_short_replies`, `_last_used` never expire keys. | RAM grows with every unique chat/user/command over months, especially if the bot is added to many chats or receives spam. | Add TTL cleanup or bounded LRU keyed by chat/user/command. | Medium |
| AUD-003 | MEDIUM | RAM / CPU / DB load | `app/services/learning_service.py`, `app/repositories/messages_repo.py` | Prefix novelty cache loads all normalized messages for a chat. | `get_all_normalized()` uses `fetchall()`; `_build_prefix_cache()` tokenizes all rows into a set. Cache is invalidated on each learned message. | Large chats can spike memory and CPU during generation; SQLite connection lock is held during fetch. | Store prefix hashes incrementally or cap sampled rows/prefixes; add a regression benchmark. | Medium |
| AUD-004 | MEDIUM | Disk / DB lifecycle | `db.py`, `app/migrations/001_initial.sql`, `README.md` | Learning data has no retention or compaction policy. | Messages and transition tables grow indefinitely; only `pivo_daily_usage` has retention. WAL is enabled but no explicit checkpoint/backup/maintenance guidance exists. | Continuous operation can consume disk and degrade queries; WAL files can surprise operators after crashes or long runs. | Define retention/maintenance policy: per-chat max rows, manual vacuum/checkpoint runbook, backup/restore guidance. | Medium |
| AUD-005 | MEDIUM | Docker / secrets hygiene | `.dockerignore`, `.gitignore`, `Dockerfile` | `.dockerignore` misses local artifacts that `.gitignore` excludes. | `.gitignore` excludes `data/`, `db_prod_copy/`, `.test_tmp/`, `Screenshot_*.jpg`; `.dockerignore` does not. `Dockerfile` uses `COPY . .`. | Local Docker build context can include DB copies, temp files, screenshots, or other local data. | Mirror sensitive/local ignore patterns into `.dockerignore`. | Low |
| AUD-006 | MEDIUM | Operations / logging | `main.py`, Docker/Compose | No application log rotation, structured logs, metrics, or health signal tied to bot health. | `logging.basicConfig()` writes plain stdout; Docker healthcheck only checks Python interpreter. | Production failures may be visible only after manual log review; disk/log growth depends on host runtime settings. | Add operator runbook now; later add structured logs/metrics and meaningful liveness if endpoint exists. | Low/Medium |
| AUD-007 | MEDIUM | Resilience | `main.py`, handlers | No global Telegram API error policy beyond local `reply_humanized` and `/pivo` quota refund. | `dp.start_polling(bot)` has no project-level error middleware; handler reply failures outside `/pivo` can bubble. | Telegram outages or rate limits can create noisy logs, dropped commands, or unclear user behavior. | Add centralized error middleware/logging policy and targeted tests for Telegram failures. | Medium |
| AUD-008 | LOW | Configuration drift | `.env`, `.env.example`, `settings.py` | Local `.env` is missing some newer optional keys by name. | Key-name-only check found no `LOG_LEVEL`, `BOT_TEXT_ALIASES`, `REPETITION_PENALTY_STRENGTH`, or `REPLY_CONTEXT_*` in local `.env`; defaults exist. | Current startup can still work, but local/prod behavior may differ from documented `.env.example`. | Manually sync `.env` with `.env.example` without printing or committing values. | Low |
| AUD-009 | HIGH | Dependency security | `requirements.lock` | `cryptography==45.0.7` is reported vulnerable by Safety. | `safety check -r requirements.lock` found CVE-2026-34073, CVE-2026-26007, and CVE-2026-39892 affecting `cryptography==45.0.7`. | The bot uses `cryptography` for `/pivo` Fernet payload encryption and HKDF log masking; even if the exact vulnerable code paths need confirmation, the locked dependency is now a supply-chain risk. | Upgrade `cryptography` to a fixed version, regenerate `requirements.lock`, and rerun full checks. | Low/Medium |
| AUD-010 | LOW | Secrets design | `app/log_masking.py`, `main.py` | Log masking derives from `PIVO_HMAC_SECRET`. | `log_masking.init_masking(settings.pivo_hmac_secret)` couples logging masks to `/pivo` secret rotation. | Rotating `/pivo` secret also rotates log correlation IDs; intended in comments but operationally relevant. | Consider separate `LOG_MASKING_SECRET` only if stable log correlation across `/pivo` rotation becomes necessary. | Low |
| AUD-011 | LOW | Static analysis | `db.py`, `markov.py`, `app/handlers/_helpers.py`, `app/handlers/learning.py`, `app/services/*` | Bandit reports only Low findings. | 47 Low findings: `assert` usage, non-cryptographic `random`, and one intentional broad `except/pass` around Telegram chat action. | No direct exploitable issue found by Bandit, but `assert` should not guard runtime invariants under optimized Python. | Triage Bandit findings; replace DB/service asserts with explicit runtime errors in a future cleanup. | Low |

## 4. Deep Dive

### Security

No tracked `.env`, `markov.db`, WAL/SHM files, `db_prod_copy/`, screenshots, or local temp files were found by `git ls-files`. `.gitignore` covers these paths. `.env.example` uses placeholders and generation guidance, not real secrets.

`settings.py` loads secrets from environment and validates presence/length. It does not print secret values. Startup exceptions mention variable names only. `main.py` creates `Bot(token=settings.bot_token)` and never logs the token.

SQL injection risk is low: reviewed SQL calls use positional parameters for user-controlled values. The only dynamic DDL found is migration-only index dropping in `005_drop_messages_text.py`; it operates on names returned by SQLite metadata and brackets the identifier.

No shell command execution, unsafe `eval`, unsafe deserialization, arbitrary file reads from user input, or external URL fetching were found in runtime bot code.

### Telegram Abuse and Spam Risks

`/pivo` is opt-in for stored subscribers when no explicit mentions are supplied. Users without username are supported through `tg://user?id=...` HTML mentions, with display name escaped. Username changes are handled on the next `/pivo_on` because subscription upsert refreshes encrypted username/display name.

The main abuse gap is explicit mentions. `parse_pivo_command()` accepts Telegram mention entities, text mentions, and plain `@username` patterns. `PivoService.build_call_message()` bypasses DB lookup when explicit mentions exist and sends them all. Daily quota limits frequency, but not fanout per message.

Admin commands `/set`, `/setprob`, and `/clear` are protected by `GroupOnly()` and `AdminOrOwner()`. The filter is fail-closed when Telegram admin lookup fails. `/config` is not admin-only and exposes runtime tuning values, but not secrets.

### Secrets and Configuration

Sensitive config:

| Secret | Loaded in | Used for | Logging risk |
|---|---|---|---|
| `BOT_TOKEN` | `settings.py` | Telegram bot auth | Not logged by project code |
| `PIVO_HMAC_SECRET` | `settings.py` | HMAC chat/user hashes and log-mask key derivation | Not logged |
| `PIVO_ENCRYPTION_SECRET` | `settings.py` | Fernet key material via SHA-256 | Not logged |
| `OWNER_ID` | `settings.py` | Owner authorization | Not secret, not logged |

No sensitive tracked files were found. Local `.env` exists and is ignored; values were not inspected. If `.env` or DB copies were ever shared outside this checkout, rotate `BOT_TOKEN`; rotate `PIVO_*` only with a migration plan because old `/pivo` subscriptions become unreadable.

### Database and Storage

SQLite uses WAL and a single `aiosqlite` connection with an `asyncio.Lock`. This serializes DB access and avoids concurrent writes inside this process. It also means expensive reads such as full `fetchall()` block other DB work while the lock is held.

Indexes exist for common Markov lookups and chat member lookup. `messages.normalized_text` has a lookup index from migration `002`, which supports exact-match duplicate checks.

Retention exists only for `pivo_daily_usage`. The learned corpus and transition counters grow forever. This is product-relevant because the bot is designed to run continuously and learn from every group message up to 500 sanitized chars.

### Memory and Resource Usage

CPU risks:

| Resource | Risk | Trigger | Impact | Evidence in code | Severity | Recommendation |
|---|---|---|---|---|---|---|
| CPU | Full prefix cache rebuild | First generation after every new message in a large chat | Tokenizes all stored messages repeatedly | `LearningService._build_prefix_cache()` | MEDIUM | Incrementally store prefixes or cap rows |
| CPU | Multiple generation retries | Mentioned bot in a sparse/low-diversity model | Up to handler retry budget times generator retry budget | `MAX_GENERATION_ATTEMPTS + MAX_TRAINING_PREFIX_RETRY_ATTEMPTS`; `VALIDATION_RETRY_ATTEMPTS` | LOW/MEDIUM | Keep caps; add timing metrics |
| CPU | Admin lookup per non-owner `/pivo` | `/pivo` by non-owner | Telegram API call and latency | `cmd_pivo()` calls `is_admin_or_owner()` | LOW | Cache admin status briefly if needed |

RAM risks:

| Resource | Risk | Trigger | Impact | Evidence in code | Severity | Recommendation |
|---|---|---|---|---|---|---|
| RAM | Unbounded runtime dicts | Many chats/users over long uptime | Slow memory growth | `RuntimeState`, `ThrottlingMiddleware` | MEDIUM | TTL/LRU cleanup |
| RAM | Full normalized-message fetch | Large learned corpus | Spike during generation | `MessagesRepo.get_all_normalized()` | MEDIUM | Prefix table or bounded sampling |
| RAM | Markov caches | High state variety | Bounded by `cache_limit` per cache | `MarkovGenerator.cache_limit=1024` | LOW | Current bound is acceptable |

Disk risks:

| Resource | Risk | Trigger | Impact | Evidence in code | Severity | Recommendation |
|---|---|---|---|---|---|---|
| Disk | DB grows forever | Continuous learning | Disk pressure, slower backups | No retention for `messages`/transitions | MEDIUM | Retention/compaction policy |
| Disk | WAL growth surprises | Long process or crash | Extra disk files, restore complexity | `PRAGMA journal_mode=WAL` | MEDIUM | Document checkpoint/backup process |
| Disk | Logs grow by runtime policy | Docker stdout host config | Disk pressure outside app | Plain stdout logging | MEDIUM | Configure host log rotation |

Network risks:

| Resource | Risk | Trigger | Impact | Evidence in code | Severity | Recommendation |
|---|---|---|---|---|---|---|
| Network | Telegram mention bursts | `/pivo` with many explicit mentions | Chat spam, rate-limit risk | No target-count cap | HIGH | Cap mentions and rate-limit `/pivo` |
| Network | Telegram admin API latency | `/pivo` by non-owner | Command latency/fail-closed to user quota | `is_admin_or_owner()` | LOW | Optional short cache |
| Network | Telegram API outages | replies/send actions fail | Dropped commands or noisy exceptions | No global error policy | MEDIUM | Central error middleware/runbook |

### Deployment and Operations

Docker runs the bot through `docker-entrypoint.sh`, fixes `/app/data` ownership, then drops privileges via `runuser`. Compose uses `restart: unless-stopped`, so crashes and reboots can restart the bot when Docker itself is managed.

The healthcheck only validates that Python can start inside the container. It does not verify polling, Telegram connectivity, DB writeability, or stuck event loop behavior. This is acceptable as a minimal smoke but weak as a production liveness signal.

No systemd, Windows service, backup script, or restore runbook was found.

### Logging and Monitoring

Positive points:

| Point | Evidence |
|---|---|
| `chat_id` is masked in learning logs | `mask_chat_id()` used in `app/handlers/learning.py` |
| Pivo logs avoid raw user/chat IDs | Only generic event and mention count are logged |
| Startup/shutdown logs exist | `main.py` logs start/stop status |

Gaps:

| Gap | Operational effect |
|---|---|
| No log rotation guidance | Docker/host must be configured manually |
| No metrics | CPU/RAM/disk/error trends are not visible from the app |
| No structured logs | Harder to aggregate and alert |
| Debug log includes context seed tokens | Not raw full message, but still user-derived text fragments |

Minimum monitoring recommendation: container restarts, process RSS, CPU, DB/WAL disk usage, error log count, Telegram API exception count, and `/pivo` call count.

### Tests and CI

Current local result after adding dev security tools: 244 unit tests OK, ruff clean, mypy clean, pip check clean. Tests cover many security-relevant paths: admin filters, group-only filters, pivo parser/escaping, pivo quota/refund, log masking, migrations, runtime config validation, and Markov generation limits.

Dev security tools are now listed in `requirements-dev.txt`: `bandit`, `pip-audit`, and `safety`.

Static/security check results:

| Tool | Result | Interpretation |
|---|---|---|
| Bandit 1.9.4 | Completed with 47 Low findings, 0 Medium, 0 High | Findings are mostly `assert` usage and non-cryptographic `random`; no direct critical code-security issue found. |
| Safety 3.7.0 | Completed with 3 vulnerabilities in `cryptography==45.0.7` | Treat as a real dependency-security finding until upgraded or disproven. |
| pip-audit 2.10.0 | Installed but did not complete; timed out after 300s against PyPI and 180s against OSV | Current environment cannot produce a pip-audit verdict; keep Safety as the available dependency signal. |

Recommended missing tests before fixes:

| Priority | Test |
|---|---|
| High | `/pivo` rejects or truncates more than N explicit mentions |
| High | subscriber mention fanout is capped |
| Medium | `ThrottlingMiddleware` expires old keys |
| Medium | `RuntimeState` cleanup removes stale chat keys |
| Medium | large-corpus prefix filter benchmark or bounded-row test |
| Medium | Telegram API failure middleware behavior outside `/pivo` |
| Low | `.dockerignore` contains all local sensitive patterns from `.gitignore` |

## 5. Action Plan

### Immediate Fixes

| Item | Why first |
|---|---|
| Cap `/pivo` explicit mentions and subscriber fanout | Direct spam/mass-mention abuse path |
| Upgrade vulnerable `cryptography` lock | Safety reports 3 CVEs against the locked version |
| Mirror local sensitive patterns into `.dockerignore` | Low-risk, prevents accidental Docker context leaks |
| Add operator guidance for DB/WAL/log rotation | Needed before long unattended runtime |

### Short-Term Fixes

| Item | Why soon |
|---|---|
| Add TTL/LRU cleanup for `RuntimeState` and throttling keys | Prevents slow memory growth |
| Replace full prefix-cache rebuild with bounded/incremental design | Prevents large-chat CPU/RAM spikes |
| Add centralized Telegram API error middleware/policy | Improves production failure behavior |
| Add dependency audit tools to CI workflow | Tools are now in dev deps, but CI does not run them yet |

### Long-Term Improvements

| Item | Why later |
|---|---|
| Define learned-data retention or archive policy | Product decision: memory quality vs disk/privacy |
| Add metrics endpoint or sidecar-friendly stats | Requires deployment decision |
| Separate `LOG_MASKING_SECRET` from `/pivo` HMAC secret | Only needed if log correlation must survive `/pivo` secret rotation |
| Meaningful healthcheck | Needs a supported health endpoint or safe local DB probe |

## 6. Proposed Implementation Plan

### Fix AUD-001: `/pivo` fanout limits

| Option | Pros | Cons | Risk | Complexity |
|---|---|---|---|---|
| A. Reject if explicit mentions or subscriber mentions exceed configured hard limit | Simple, predictable, no accidental mass ping | User must retry with fewer targets | Low | Low |
| B. Truncate mentions to first N and say it was truncated | Keeps command useful | Some users may not notice missing targets | Medium | Low |

Recommendation: Option A for explicit mentions, Option B or A for subscriber list depending on product preference. Add tests first.

### Fix AUD-002: bounded runtime state

| Option | Pros | Cons | Risk | Complexity |
|---|---|---|---|---|
| A. TTL cleanup on every N updates | Minimal dependencies, simple | Cleanup happens only when traffic exists | Low | Medium |
| B. Use bounded `OrderedDict`/LRU per map | Hard memory cap | May evict active low-frequency chats | Medium | Medium |

Recommendation: Option A with conservative TTL for throttling and inactive chat state.

### Fix AUD-003: prefix cache scaling

| Option | Pros | Cons | Risk | Complexity |
|---|---|---|---|---|
| A. Store prefix hashes during message save | Fast lookup, no full scan | Migration and schema change required | Medium | Medium |
| B. Sample/cap recent normalized rows | Minimal schema change | Less complete novelty filter | Low | Low/Medium |

Recommendation: Option B first if production data is small; Option A if corpus size keeps growing.

### Fix AUD-004: DB growth policy

| Option | Pros | Cons | Risk | Complexity |
|---|---|---|---|---|
| A. Documentation/runbook only | No behavior change | Does not stop growth automatically | Low | Low |
| B. Per-chat retention limit and compaction command | Controls disk | Changes model behavior and privacy semantics | Medium | Medium/High |

Recommendation: Option A immediately, then decide whether model retention is acceptable.

### Fix AUD-005: Docker context hygiene

| Option | Pros | Cons | Risk | Complexity |
|---|---|---|---|---|
| A. Add missing patterns to `.dockerignore` | Direct and safe | None material | Low | Low |
| B. Build from clean export/context | Stronger isolation | More workflow overhead | Low | Medium |

Recommendation: Option A.

### Fix AUD-009: vulnerable `cryptography` lock

| Option | Pros | Cons | Risk | Complexity |
|---|---|---|---|---|
| A. Upgrade only `cryptography` to a fixed version and regenerate `requirements.lock` | Smallest dependency change | Transitive compatibility still needs checks | Low/Medium | Low |
| B. Refresh all runtime dependencies from `requirements.txt` and regenerate lock | Brings the stack current | Larger diff and more regression risk | Medium | Medium |

Recommendation: Option A first, then run full unit/lint/type/security checks. If `cryptography` upgrade pulls incompatible transitive changes, use Option B in a dedicated dependency PR.

## 7. Commands Run

| Command | Result |
|---|---|
| `Get-Location; Get-ChildItem -Force` | Inspected repository root and local artifacts. |
| `git status --short --branch` | Started on `fix-short-reply-policy`; later on `audit-security-stability-review`. |
| `git branch --all` | Existing visible style: `fix-short-reply-policy`, `main`, matching remotes. |
| `git switch -c audit-security-stability-review` | First attempt failed due sandbox `.git` lock permission; approved rerun succeeded. |
| `git ls-files .env markov.db markov.db-wal markov.db-shm db_prod_copy Screenshot_*.jpg` | No tracked sensitive/local runtime files. |
| `git ls-files` | Reviewed tracked file inventory. |
| `Get-Content -Raw README.md` | Reviewed project docs. |
| `Get-Content -Raw PROJECT_AUDIT.md` | Reviewed baseline audit; not edited. |
| `Get-Content -Raw PROJECT_AUDIT_CODEX.md` | Reviewed working audit log. |
| `Get-Content -Raw docs\ARCHITECTURE.md` | Reviewed architecture docs. |
| `Get-Content -Raw .github\workflows\ci.yml` | Reviewed CI. |
| `Get-Content -Raw main.py settings.py db.py pivo.py app\log_masking.py` | Reviewed startup, config, DB facade, pivo crypto, log masking. |
| `Get-Content -Raw app\handlers\*.py app\middlewares\throttling.py` | Reviewed handlers, filters, throttling. |
| `Get-Content -Raw app\services\*.py app\repositories\*.py app\infrastructure\migrator.py` | Reviewed services, repos, migrator. |
| `Get-Content -Raw .gitignore .dockerignore Dockerfile compose.yaml requirements*.txt pyproject.toml` | Reviewed deployment/dependency configs. |
| `rg -n ... secret/token/logging patterns` | No committed secrets found; log sites reviewed. |
| `rg -n ... unsafe API patterns` | No runtime shell/eval/unsafe deserialization found. |
| `.env` key-name-only extraction | Values not printed; local `.env` has required secret keys by name but lacks some newer optional keys. |
| `git check-ignore -v ...` | Confirmed `.env`, DB files, `db_prod_copy/`, `.test_tmp/`, screenshots, AGENTS.md are Git-ignored. |
| `git ls-files --others --exclude-standard` | No visible unignored untracked files. |
| `.\.venv\Scripts\python.exe --version` | Python 3.14.0. |
| `.\.venv\Scripts\python.exe -m unittest discover tests -v` | 244 tests OK. |
| `.\.venv\Scripts\python.exe -m ruff check app/ tests/` | All checks passed. |
| `.\.venv\Scripts\python.exe -m mypy app/` | Success, 29 source files. |
| `.\.venv\Scripts\python.exe -m bandit -r ...` | Initial attempt failed before dev install: `No module named bandit`; after install, completed with 47 Low findings, 0 Medium, 0 High. |
| `.\.venv\Scripts\python.exe -m pip_audit -r requirements.lock` | Initial attempt failed before dev install; after install first failed on cache permission, then on Windows encoding, then timed out after 300s with `PYTHONUTF8=1` and local cache. |
| `.\.venv\Scripts\python.exe -m pip_audit -r requirements.lock --vulnerability-service osv` | Timed out after 180s. |
| `.\.venv\Scripts\safety.exe check -r requirements.lock` | Initial attempt failed before dev install; after install and local `USERPROFILE`, completed and found 3 vulnerabilities in `cryptography==45.0.7`. |
| `.\.venv\Scripts\python.exe -m pip list --outdated` | Completed; several packages outdated, no vulnerability assessment. |
| `.\.venv\Scripts\python.exe -m pip check` | No broken requirements found. |
| `.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` | Installed dev security tools: `bandit`, `pip-audit`, `safety` and their dependencies. |
| `.\.venv\Scripts\python.exe -m ruff check app/ tests/` | Re-run after dev-tool install: all checks passed. |
| `.\.venv\Scripts\python.exe -m mypy app/` | Re-run after dev-tool install: success, 29 source files. |
| `.\.venv\Scripts\python.exe -m unittest discover tests -v` | Re-run after dev-tool install: 244 tests OK. |
| `.\.venv\Scripts\python.exe -m pip check` | Re-run after dev-tool install: no broken requirements. |
| `docker --version` | Not available: `docker` command not found. |
| `docker build -t pepe-bot:audit .` | Not run: `docker` command not found. |
| `git remote -v` | Origin: `https://github.com/fdlone/PepeEdtaBot.git`. |
| `gh --version` | Not available: `gh` command not found. |
| `git log --oneline --decorate -5` | Confirmed audit branch is based on current `fix-short-reply-policy` head, not `origin/main`. |

## 8. Files Reviewed

Most important files reviewed:

| Area | Files |
|---|---|
| Startup/config | `main.py`, `settings.py`, `runtime_state.py`, `runtime_config.py`, `config_registry.py` |
| Handlers | `app/handlers/admin.py`, `app/handlers/common.py`, `app/handlers/learning.py`, `app/handlers/pivo.py`, `app/handlers/_helpers.py` |
| Security filters/middleware | `app/filters/admin_or_owner.py`, `app/filters/group_only.py`, `app/middlewares/throttling.py`, `app/log_masking.py` |
| Services | `app/services/learning_service.py`, `app/services/pivo_service.py`, `app/services/pivo_parser.py`, `app/services/pivo_message_builder.py` |
| Storage | `db.py`, `app/repositories/*.py`, `app/migrations/*`, `app/infrastructure/migrator.py` |
| Generation/text | `markov.py`, `text_utils.py`, `bot_policy.py`, `bot_messages.py`, `pivo.py`, `pivo_templates.py` |
| Deployment | `Dockerfile`, `docker-entrypoint.sh`, `compose.yaml`, `.dockerignore`, `.github/workflows/ci.yml` |
| Dependencies | `requirements.txt`, `requirements.lock`, `requirements-dev.txt`, `pyproject.toml` |
| Docs/audits/tests | `README.md`, `docs/ARCHITECTURE.md`, `PROJECT_AUDIT.md`, `PROJECT_AUDIT_CODEX.md`, `tests/*` |
