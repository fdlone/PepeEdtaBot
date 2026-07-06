# 14 — Logging & Observability (+ Documentation Accuracy)

> Independent audit, source-only. Covers logging configuration, levels, what is/isn't logged, metrics, tracing, health checks, and error reporting — then a Phase-12 documentation-accuracy pass (README/ARCHITECTURE/OPERATIONS vs. code). No production code modified.
>
> Cross-refs: log masking & sensitive-data review in [08_security.md](08_security.md); Semgrep logger findings in [08] §1.2; config in [13_configuration.md](13_configuration.md).

## 0. Summary

Logging is **simple, privacy-aware, and correct for a single-process bot**, but **observability is minimal** — there are no metrics, no tracing, and no application-level health endpoint (the only health signal is the Docker `HEALTHCHECK` doing a SQLite `SELECT 1`). For the current scope this is a reasonable trade-off; the gaps matter only if the bot grows or needs production monitoring. Documentation (README/ARCHITECTURE/OPERATIONS) is **accurate** — spot-checks against code passed, including referenced files and command/router wiring.

| Area | State |
|---|---|
| Logging config | Single `basicConfig` at startup, env-driven level, structured-ish format (L1) |
| Sensitive data | `chat_id` masked (HKDF); no message text/tokens/secrets logged (see [08]) — exemplary |
| Logger naming | Inconsistent: `chat_markov` everywhere except `errors.py` uses `__name__` (L2) |
| Metrics / tracing | **None** (L3) |
| Health check | Docker-level only (`SELECT 1`); no in-app readiness/liveness (L4) |
| Error reporting | Global aiogram error handler + logging; no Sentry/aggregator (L5) |
| Documentation | Accurate; minor notes (§5) |

## 1. Logging configuration

`main.run_bot` configures logging once at startup (`main.py:69-74`):
```python
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("chat_markov")
logging.getLogger("aiogram").setLevel(logging.WARNING)
```
- **Level** is env-driven (`LOG_LEVEL`, enum-validated in [13]). Sensible default `INFO`.
- **Format** includes timestamp, level, logger name, message — adequate for console/`docker logs`. Not JSON, so not directly machine-parseable (fine for current scale; a structured/JSON formatter would help if shipped to a log aggregator — see L3).
- **Noise control:** aiogram pinned to `WARNING` so framework chatter doesn't drown app logs. Good.
- **Output:** stdout/stderr via `basicConfig` (no file handler) — correct for containers (`docker logs` / `OPERATIONS.md` covers log handling).

## 2. Findings

### L1 — Single, consistent app logger · **Info (good)**
Almost all modules use `logging.getLogger("chat_markov")`, so the whole app shares one configurable logger tree. Combined with the env-driven level, an operator can dial verbosity with one variable.

### L2 — Inconsistent logger name in `errors.py` · **Low**
`app/handlers/errors.py:10` uses `logging.getLogger(__name__)` (`app.handlers.errors`) instead of `"chat_markov"`. It still works (propagates to the root handler configured by `basicConfig`), but it sits outside the `chat_markov` tree, so a future `getLogger("chat_markov").setLevel(...)` tweak wouldn't affect it. **Fix:** use `"chat_markov"` for consistency. Confidence: **High**.

### L3 — No metrics or tracing · **Low–Medium (observability gap)**
No counters/histograms (messages learned, replies generated, generation rejects, `/pivo` calls, throttle drops), no tracing. Diagnosing "why did the bot get slow/quiet" relies on `DEBUG` log spelunking. The generation pipeline already computes rich signals (the `GenerationTrace` logged at debug, [08] §1.2) that would map directly to metrics.
- **Fix (if monitoring is wanted):** expose a few counters (Prometheus client or even periodic log-summary lines). Low urgency at current scale. Confidence: **High** that they're absent.

### L4 — Health check is container-only · **Low**
`Dockerfile` `HEALTHCHECK` runs `python -c "...sqlite3.connect(DB_PATH).execute('SELECT 1')"` — it verifies the interpreter and DB file open, **not** that polling is alive or that the bot is connected to Telegram. A bot that lost its Telegram session but kept the process alive would still report healthy.
- **Fix (optional):** a liveness signal tied to "last successful poll/update" (e.g., touch a heartbeat file or expose readiness). Confidence: **High** on the limitation; impact depends on ops needs.

### L5 — Error reporting is log-only · **Low**
A global aiogram error handler (`handlers/errors.py`, `test_error_handler.py` covers it) logs unhandled exceptions; there is no external error aggregator (Sentry/etc.). Fine for a hobby/single-instance bot; note it for production hardening. The one silent `except Exception: pass` ([08] S6 / [11] Q1) is the only place an error is dropped without a log line — worth fixing for completeness.

## 3. What is logged (privacy posture — see [08])

- **Masked identifiers:** `chat_id` via `mask_chat_id` (HKDF-SHA256), fail-fast if uninitialized. **No raw chat IDs.**
- **No content:** message text, generated replies, and n-gram tokens are **not** logged; debug lines log **counts/booleans/enums** only (Semgrep's 4 "credential leak" hits were all false positives — [08] §1.2).
- **Startup:** logs only the bot username (`main.py:118`). **No token.**
This is a deliberately privacy-conscious logging design and should be preserved (README explicitly instructs: "не добавляйте raw `chat_id` в новые log-сообщения").

## 4. Recommendations (logging/observability)

| Priority | Item | Effort |
|---|---|---|
| **P4** | L2 — use `"chat_markov"` in `errors.py` | XS |
| **P4** | L5/S6 — log the swallowed chat-action exception | XS |
| **P3** | L3 — add basic counters (or periodic summary log) if monitoring is desired | M |
| **P4** | L4 — liveness tied to last successful update | M |
| **P4** | optional JSON formatter for aggregator ingestion | S |

## 5. Documentation accuracy (Phase 12)

Spot-checked the functional docs against current code; **documentation is accurate and unusually well-maintained.**

**Verified correct:**
- `README.md` references — `compose.yaml` ✓ exists, `docker-entrypoint.sh` ✓ exists, `.env.example` ✓, migration filenames (`003_anonymize_authors.py`, `005_drop_messages_text.py`) ✓.
- **Commands** listed in README (`/help /ping /pivo /pivo_on /pivo_off /pivo_privacy /stats /config /set /setprob /clear`) all match `@router.message(Command(...))` decorators in `handlers/`.
- **Routers**: README's "четыре Router'а (common, admin, pivo, learning)" + error handler matches `main.py:58-62` (5 `include_router` calls).
- **Architecture/layers** description matches the actual package layout ([02]/[05]).
- **Privacy claims** (text column dropped, author_id anonymized, `/pivo` HMAC+Fernet) match migrations and `domain/pivo.py`.
- README **accurately** states the Fernet key is "SHA-256 от PIVO_ENCRYPTION_SECRET" — i.e. it correctly documents the non-KDF derivation flagged as [08] S3 (doc is honest; the *code* is the item to harden, not the doc).

**Minor doc notes:**
- D1 — The audit's own README under `docs/_pre_audit_archive/README.md` marks the quarantined prior audits; current `docs/` cleanly separates functional docs (README/ARCHITECTURE/OPERATIONS) from this audit set. No stale audit conclusions leak into functional docs. ✓
- D2 — `AGENTS.md` exists at repo root (contributor/agent guidance) — not cross-referenced from README; harmless.
- D3 — `/config` is documented but, unlike `/set`/`/setprob`/`/clear`, its handler (`admin.py:48`) has no `GroupOnly()/AdminOrOwner()` filter. It only displays non-secret runtime tuning values, so this is not a security issue ([08]), but the README could note that `/config` is readable by anyone in the chat. **Low.**

**Verdict:** no outdated/contradictory documentation found. The docs would let a new engineer run and understand the system without reading every source file — which is the TZ's stated bar.
