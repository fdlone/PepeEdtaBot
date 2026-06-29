# 06 — Dependency Graph

> Independent audit. Internal import edges extracted programmatically from source (`grep` over `from app... import` / `import app...`), then verified by reading. External deps cross-ref [01_project_overview.md](01_project_overview.md); full dependency audit in [13_configuration.md]/[15_technical_debt.md] (M3/M6).

## Internal layer dependency graph

Arrows = "imports". Read top-to-bottom (higher layers depend on lower).

```
                         main.py
        ┌──────────┬─────────┴───────┬───────────────┬──────────────┐
        ▼          ▼                 ▼               ▼              ▼
   handlers/*   middlewares    presentation       services       config
   (common,     (throttling)   (bot_messages)   (learning,      (settings,
    admin,                                        pivo_*)         runtime_state)
    pivo,                          │                 │               │
    learning,                      │                 ▼               ▼
    errors,                        │             domain/*         registry
    _helpers)                      │            (pivo,            (root config
        │                          │             pivo_templates)   source)
        ▼                          │                 │
      core/*  ◄──────────────────────────────────────┘ (services→core)
   (markov, response_generator, candidate_scorer,
    context_state_matcher, reply_policy, text,
    privacy_filter, lexicon)
        │
        ▼
   infrastructure/database  ──▶ repositories/*  ──▶ aiosqlite
        │         ▲                                    
        │         └── (database imports core.text for sanitize on write)
        ▼
   infrastructure/migrator ──▶ migrations/*
```

## Edge list (internal, module granularity)

Source of truth (verified):

| Module | Imports (internal) |
|---|---|
| `main.py` | config.runtime_state, config.settings, core.markov, domain.pivo, handlers.*, infrastructure.database, middlewares, presentation.bot_messages, services, log_masking |
| `config/runtime_config.py` | config.registry |
| `config/runtime_state.py` | config.registry, config.settings |
| `config/settings.py` | config.registry, core.reply_policy |
| `core/candidate_scorer.py` | core.lexicon, core.markov |
| `core/context_state_matcher.py` | infrastructure.database |
| `core/markov.py` | core.context_state_matcher, core.lexicon, infrastructure.database |
| `core/reply_policy.py` | core.markov |
| `core/response_generator.py` | config.runtime_state, core.candidate_scorer, core.markov, core.text, log_masking |
| `core/text.py` | core.privacy_filter |
| `domain/pivo.py` | services.pivo_message_builder *(lazy, in-function)* |
| `filters/admin_or_owner.py` | config.settings |
| `handlers/admin.py` | config.runtime_config, config.runtime_state, config.settings, core.markov, filters, handlers._helpers, infrastructure.database, presentation.bot_messages |
| `handlers/common.py` | config.runtime_state, filters, handlers._helpers, infrastructure.database, presentation.bot_messages |
| `handlers/learning.py` | config.runtime_state, core.markov, core.reply_policy, core.response_generator, core.text, handlers._helpers, log_masking, services |
| `handlers/pivo.py` | config.runtime_state, config.settings, domain.pivo, filters, handlers._helpers, services, services.pivo_parser, services.pivo_service |
| `infrastructure/database.py` | core.text, infrastructure.migrator, repositories |
| `repositories/__init__.py` | repositories.{chat_members,markov,messages,pivo_usage}_repo |
| `repositories/messages_repo.py` | core.text |
| `services/__init__.py` | services.{learning_service,pivo_service} |
| `services/learning_service.py` | core.markov, core.text, infrastructure.database |
| `services/pivo_message_builder.py` | domain.pivo_templates |
| `services/pivo_service.py` | domain.pivo, infrastructure.database, services.pivo_message_builder |
| `migrations/002_*.py` | core.text |

Modules with **no internal imports** (leaves): `config/registry`,
`core/lexicon`, `core/privacy_filter`, `domain/pivo_templates`,
`filters/group_only`, `handlers/errors`, `handlers/_helpers`,
`infrastructure/migrator`, `middlewares/throttling`, `presentation/bot_messages`,
`pivo_parser`, `markov_repo`, `chat_members_repo`, `pivo_usage_repo`,
`log_masking`.

## Cycles

**No import-time cycles.** The only back-edge against the intended inward
direction is `domain/pivo.py → services/pivo_message_builder`, and it is a
**lazy in-function import** (`app/domain/pivo.py:119`, inside
`get_random_pivo_message`) specifically to avoid an import cycle
(`pivo_service → domain.pivo → pivo_message_builder → domain.pivo_templates`).
Since the edge is not evaluated at import time, the module graph is acyclic.

Two notable bidirectional couplings at *package* granularity (not cycles, since
distinct modules):
- `core ⇄ infrastructure`: `core/markov` & `core/context_state_matcher` import
  `infrastructure/database`; `infrastructure/database` imports `core/text`.
- The `Database` facade is imported by `core`, `services`, and `handlers` —
  it is the de-facto central hub.

## Fan-in / fan-out hotspots

- **Highest fan-in (most depended-upon):**
  `core/markov` (imported by 6 modules), `infrastructure/database` (6+),
  `config/runtime_state` (5), `core/text` (5+ incl. write path & migration),
  `config/registry` (3), `config/settings` (4).
- **Highest fan-out:** `main.py` (10), `handlers/admin` (8), `handlers/pivo` (8),
  `handlers/learning` (8).

Implication: `markov.py`, `database.py`, and `text.py` are the structural load
points — changes there ripple widely, and they are the priority targets for the
test suite and any refactor (see [11_code_quality.md], [12_testing.md],
[16_refactoring_plan.md]).

## External (third-party) dependencies

Runtime (`requirements.txt` / pinned in `requirements.lock`):

| Package | Used by | Purpose |
|---|---|---|
| `aiogram` (3.29) | handlers, filters, middlewares, main, pivo_parser/service | Telegram bot framework (also pulls `aiohttp`, `magic-filter`, `pydantic`) |
| `aiosqlite` (0.22) | repositories, database | async SQLite |
| `cryptography` (49.0) | domain.pivo (Fernet), log_masking (HKDF/HMAC) | encryption + key derivation |
| `python-dotenv` (1.2) | settings | `.env` loading |

Transitive (from lock): `aiohttp`/`aiohappyeyeballs`/`aiosignal`/`multidict`/
`yarl`/`frozenlist`/`propcache`, `pydantic`/`pydantic_core`/`annotated-types`/
`typing-inspection`, `certifi`, `cffi`/`pycparser`, `idna`, `attrs`,
`magic-filter`, `aiofiles`, `typing_extensions`.

Dev/CI (`requirements-dev.txt`): `bandit`, `mypy>=2.0`, `pip-audit`,
`ruff>=0.15`, `safety`.

**Observations (carried to M3 dependency audit):**
- `requirements.lock` is `pip freeze` output **without hashes** and includes
  `aiofiles` though no runtime module imports it (transitive via aiogram? to be
  confirmed in M3).
- Standard library only otherwise: `asyncio`, `sqlite3`(via aiosqlite),
  `hashlib`, `hmac`, `base64`, `html`, `re`, `math`, `random`, `logging`,
  `dataclasses`, `collections`, `itertools`, `importlib`, `pathlib`.
- No numpy/pandas/ML libs — generation is hand-rolled, keeping the image small.

## Stdlib `random` usage (determinism note)

Generation uses an injected `random.Random` instance for reproducibility in
tests, **but** `pivo_message_builder` and `_helpers.reply_humanized` call the
**module-global** `random` (`random.choice`, `random.randint`) — non-injectable.
Minor; relevant to test determinism and noted in [11_code_quality.md] (M2).
