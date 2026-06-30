# 13 — Configuration

> Independent audit, source-only. Covers environment variables, loading/validation, defaults, secrets management, and feature flags. Cross-refs: secret-handling risks in [08_security.md](08_security.md); the registry pattern in [05_module_inventory.md](05_module_inventory.md); typing of state in [11_code_quality.md](11_code_quality.md).

## 0. Summary

Configuration is **well-engineered**. There is one immutable `Settings` dataclass loaded once at startup (`app/config/settings.py`) and a `registry.py` that is the **single source of truth** for the ~23 runtime-mutable fields — adding a `/set`-able field is a one-line edit. Every env var is parsed with an explicit type and validated (required-ness, ranges, enums) with clear `ValueError` messages, so misconfiguration **fails fast at boot** rather than misbehaving later. There is no config file, no remote config, no dynamic `eval` of config — only environment variables.

The only configuration risks are the two carried into [08_security.md]: **S2** (placeholder secrets pass the length check → insecure default) and the **stale `ignore_errors`/`object`-typed state** consumer noted in [11_code_quality.md]. Everything else is informational.

## 1. Configuration model

| Layer | File | Role |
|---|---|---|
| Immutable settings | `app/config/settings.py` → `Settings` (frozen, slots) | Loaded once via `load_settings()`; holds secrets + non-mutable knobs. |
| Field registry | `app/config/registry.py` → `RUNTIME_FIELDS` (23 specs) | Single source of truth for runtime-mutable fields: name, type, range, default, parser. |
| Mutable runtime state | `app/config/runtime_state.py` → `RuntimeState` | Per-process mutable copy seeded from settings; mutated by `/set`. |
| `/set` application | `app/config/runtime_config.py` | Validates + applies admin `/set key value`; bounded to `ALLOWED_RUNTIME_KEYS`. |

**Strength:** the registry eliminates the usual drift between "env var → settings field → /set handler → display" by deriving all four from one spec list. This is a genuine anti-duplication win and should be preserved.

## 2. Environment variables

~40 variables, all documented in `.env.example`. Categories:

- **Secrets / identity (non-mutable):** `BOT_TOKEN` (required, fails closed), `OWNER_ID` (optional int → `None`), `PIVO_HMAC_SECRET`, `PIVO_ENCRYPTION_SECRET` (each `len >= 16`), `DB_PATH`.
- **`/pivo` limits:** `PIVO_EXPLICIT_MENTIONS_LIMIT` (≥1), `PIVO_SUBSCRIBER_FANOUT_LIMIT` (≥1).
- **Memory bounds (long-lived process):** `RUNTIME_STATE_TTL_SEC`, `RUNTIME_STATE_MAX_CHATS`, `THROTTLE_STATE_TTL_SEC`, `THROTTLE_STATE_MAX_KEYS`, `TEXT_CACHE_MAX_MESSAGES` (each int ≥1).
- **Generation tuning (runtime-mutable via registry/`/set`):** `REPLY_PROBABILITY`, `MIN_COOLDOWN_SEC`, `MIN_TOKENS_FOR_MODEL`, `MAX_REPLY_CHARS`, `MAX_REPLY_TOKENS`, `RANDOMNESS_STRENGTH`, `REPETITION_PENALTY_STRENGTH`, `MARKOV_ORDER`, `ENABLE_BACKOFF`, `BACKOFF_MIN_ORDER`, `NORMALIZE_LOWER`, `AUTO_CAPITALIZE_REPLIES`, `TYPING_MIN_MS/MAX_MS`, and the `REPLY_CONTEXT_*` family.
- **Feature flags:** `USE_REPLY_CONTEXT`, `REPLY_CONTEXT_ONLY_FOR_REPLIES`, `REPLY_CONTEXT_INCLUDE_CURRENT_MESSAGE`, `ENABLE_BACKOFF`, `FUZZY_CONTEXT_CASEFOLD`, `FUZZY_CONTEXT_PREFIX` (last two default **off** — new generation paths gated conservatively).
- **Operational:** `LOG_LEVEL` (enum-validated), `BOT_TEXT_ALIASES` (CSV → frozenset, empty → built-in defaults).

## 3. Validation quality (evidence)

`load_settings()` (`settings.py:69-176`) validates thoroughly:
- **Required:** `BOT_TOKEN` empty → `ValueError("BOT_TOKEN is required")` (`:74`).
- **Type-safe ints with friendly errors:** `OWNER_ID` (`:78`), all limit/bound vars wrap `int()` and raise `"<VAR> must be an integer"`, then enforce `>= 1` floors.
- **Enum:** `LOG_LEVEL` constrained to the 5 valid levels (`:156`).
- **Side effect:** ensures `DB_PATH` parent dir exists (`os.makedirs(..., exist_ok=True)`, `:88`) — convenient, though uses `os.path`/`os.makedirs` rather than `pathlib` (cosmetic, see [11] §1.2).
- Range/bounds for the runtime fields are enforced centrally by `registry.py` parsers (`_int_in_range`, `_float_in_range`), shared between boot-time load and `/set`.

This is **fail-fast configuration** done well.

## 4. Findings

### C1 — Secret validation is length-only (→ S2) · **Medium**
`len(secret) >= 16` accepts the public `.env.example` placeholders. See [08_security.md] **S2** for impact and fix (reject `change_me*`, require `hmac_secret != encryption_secret`). The `.env.example` text already states the requirement ("два разных значения") but the loader does not enforce it.

### C2 — `format_config_message(state: object)` weakens the config-display contract · **Low**
`app/presentation/bot_messages.py:56` types the runtime state as `object`, which (a) blocks strict typing for the whole `app` package (see [11_code_quality.md] **Q5**) and (b) means the `/config` display reaches into `state.<field>` with no static guarantee the field exists. Typing it as `RuntimeState` fixes both. Quick win.

### C3 — No validation that `BACKOFF_MIN_ORDER < MARKOV_ORDER` at the env layer · **Low / Info**
`.env.example` documents the constraint ("должен быть < MARKOV_ORDER"); `registry.validate_cross_fields` is the intended home for such cross-field checks. Confirm it covers this pair (and `REPLY_CONTEXT_LAST_TOKENS <= REPLY_CONTEXT_MAX_TOKENS`); if not, add them so an invalid combination fails at boot rather than silently degrading generation. Confidence: **Medium** (needs a quick check of `validate_cross_fields` coverage during M4/M6).

## 5. Secrets management

- **Source:** environment only (via `python-dotenv` `load_dotenv()` in `load_settings`). No secrets in code, no secret files committed.
- **VCS hygiene:** `.env` git-ignored; `.env.example` ships placeholders + a generation command (`secrets.token_urlsafe(32)`).
- **Rotation:** `.env.example` warns that rotating pivo secrets invalidates existing `/pivo` subscriptions (HMAC/Fernet are deterministic over the secret) — accurate and documented. `log_masking` intentionally re-keys on rotation too.
- **Gap:** no enforced distinctness/entropy (C1/S2).

## 6. Feature flags

Flags are clean and conservative:
- `FUZZY_CONTEXT_CASEFOLD` / `FUZZY_CONTEXT_PREFIX` default **off**; documented as additive to the exact-match generation path. This matches the recent branch work (`feat/generation-context-matching`) — new behavior is opt-in, exact path unchanged.
- `ENABLE_BACKOFF`, `USE_REPLY_CONTEXT`, `AUTO_CAPITALIZE_REPLIES` all have safe defaults documented inline in `.env.example`.
- All runtime flags are registry-backed, so they are both env-settable and `/set`-mutable with identical validation — no divergence.

## 7. Recommendations

| Priority | Item | Effort |
|---|---|---|
| **P1** | C1/S2 — reject placeholder + equal pivo secrets | S |
| **P2** | C2 — type `state: RuntimeState` in `bot_messages` (= Q5/[11]) | XS |
| **P3** | C3 — confirm/add cross-field validation (`backoff_min_order < markov_order`, reply-context bounds) | S |
| **P4** | cosmetic — `pathlib` for `DB_PATH` dir creation | XS |

No simplification needed — the configuration layer is already minimal and DRY. The registry pattern is a model the rest of the codebase could emulate.
