# 02 — Repository Map

> Independent audit, source-only. Companion to [05_module_inventory.md](05_module_inventory.md) (per-module responsibilities) and [03_architecture.md](03_architecture.md) (layering).

## Top-level layout

```
PepeEdtaBot/
├── main.py                  # composition root + long-polling loop (133 LOC)
├── app/                     # application package (layered)
│   ├── config/              # Settings, RuntimeState, runtime field registry
│   ├── core/                # generation engine + text/privacy/scoring (pure-ish)
│   ├── domain/              # /pivo crypto + message templates
│   ├── filters/             # aiogram filters: GroupOnly, AdminOrOwner
│   ├── handlers/            # aiogram routers: common, admin, pivo, learning, errors
│   ├── infrastructure/      # Database facade + migration runner
│   ├── middlewares/         # ThrottlingMiddleware
│   ├── migrations/          # NNN_*.sql / NNN_*.py schema migrations
│   ├── presentation/        # user-facing message strings/formatters
│   ├── repositories/        # per-domain SQL access
│   ├── services/            # LearningService, PivoService (+ pivo builder/parser)
│   └── log_masking.py       # HKDF-based chat_id masking for logs
├── tests/                   # stdlib unittest suite (~24 modules)
├── tools/                   # seed + offline evaluation scripts (not shipped at runtime)
├── docs/                    # documentation (audit output + functional docs)
├── Dockerfile, compose.yaml, docker-entrypoint.sh
├── requirements.txt / .lock / requirements-dev.txt
├── pyproject.toml           # ruff + mypy config (no build backend section)
├── .env.example             # documented config template
└── .github/workflows/ci.yml # lint, type, test (3.12/3.13/3.14), bandit, safety, docker build
```

## `app/` package by layer

| Dir | Files (LOC) | Role |
|---|---|---|
| `config/` | `registry.py` (170), `settings.py` (195), `runtime_state.py` (90), `runtime_config.py` (36), `runtime_config`/`__init__` | Env loading, runtime-mutable state, single source of truth for `/set` fields |
| `core/` | `markov.py` (1438), `context_state_matcher.py` (288), `response_generator.py` (207), `candidate_scorer.py` (176), `privacy_filter.py` (114), `lexicon.py` (105), `reply_policy.py` (82), `text.py` (74) | Reply generation pipeline + text normalization/PII redaction |
| `domain/` | `pivo_templates.py` (399, data), `pivo.py` (121) | `/pivo` crypto primitives, mention building, template pools |
| `filters/` | `admin_or_owner.py` (33), `group_only.py` (10), `__init__` (4) | Permission / chat-type gates |
| `handlers/` | `learning.py` (288), `admin.py` (211), `pivo.py` (147), `common.py` (42), `_helpers.py` (28), `errors.py` (19) | aiogram routers (the only Telegram-aware orchestration layer) |
| `infrastructure/` | `database.py` (385), `migrator.py` (90) | SQLite connection facade + migration runner |
| `middlewares/` | `throttling.py` (97), `__init__` (3) | Per-user/command cooldown |
| `migrations/` | `001_initial.sql`, `002`/`003`/`005` `.py`, `004`/`006`/`007` `.sql` | Schema evolution (see [07_database.md] M4) |
| `presentation/` | `bot_messages.py` (117) | Command list + message formatters |
| `repositories/` | `markov_repo.py` (193), `pivo_usage_repo.py` (120), `chat_members_repo.py` (90), `messages_repo.py` (46) | SQL by domain, all share one `asyncio.Lock` |
| `services/` | `pivo_message_builder.py` (204), `pivo_service.py` (155), `pivo_parser.py` (148), `learning_service.py` (64) | Business logic between handlers and repos |
| `log_masking.py` | (79) | Process-global HKDF key for masking `chat_id` |

## Migrations (schema source of truth)

`app/migrations/` — applied once each by `app/infrastructure/migrator.py`, recorded in `schema_migrations`:

- `001_initial.sql` — messages, starts, starts3, transitions, transitions3, transitions1, pivo_chat_members + indexes.
- `002_normalize_messages_text_column.py` — add/backfill `normalized_text`.
- `003_anonymize_authors.py` — `author_id := 0`.
- `004_chat_member_profiles.sql` — creates `chat_member_profiles` (later dropped).
- `005_drop_messages_text.py` — drop raw `messages.text`.
- `006_pivo_daily_usage.sql` — per-day `/pivo` quota table.
- `007_unify_chat_members.sql` — consolidate into `chat_members`; drop `pivo_chat_members` + unused `chat_member_profiles`.

Full schema analysis in [07_database.md] (M4).

## Non-shipped / tooling

- `tools/seed_db.py`, `tools/seed_diverse.py` — synthetic corpus seeders for local smoke tests.
- `tools/eval_generation.py` — synthetic selection-path evaluation harness.
- `tools/eval_prod.py` (+ `eval_prod_baseline.json`) — offline best-of-N evaluation against a prod-copy DB (copies DB to temp first). **Untracked/new** per initial `git status`.

## Docs

- **Functional (kept):** `README.md`, `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md` — used only for Phase 12 accuracy comparison, not as audit input.
- **Audit output (this set):** `docs/project_audit/`.
- **Quarantined prior audits:** `docs/_pre_audit_archive/` (6 docs) — deprecated, not used (Phase 0).
- **The audit brief:** `docs/Project Audit.md`.

## Ignored / local-only (not in git)

`.gitignore` excludes `.env`, `*.db`/`*.db-wal`/`*.db-shm`, `data/`,
`db_prod_copy/`, `.test_tmp/`, `.venv/`, `AGENTS.md`, `CLAUDE.md`, `.claude/`,
`uv.lock`. Confirmed: the working-tree `markov.db`, `db_prod_copy/`, `.env`,
`.test_tmp/` are present locally but **not tracked** — no committed
secrets/data from these paths. (Verified via `git ls-files`.)

## Config artifacts

- `pyproject.toml` — ruff (`E,F,I,UP`, line 100, py312) + mypy (strict `app.*` with a 10-module `ignore_errors` exemption block). No `[build-system]`/`[project]` — repo is run as a script, not installed.
- `AGENTS.md` (gitignored) — local agent instructions (answer in Russian, minimal safe changes, plan→implement→test→verify).
