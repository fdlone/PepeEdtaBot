# 11 — Code Quality

> Independent audit, source-only. Tooling-first (Ruff + mypy, then ToB `modern-python`), then manual SOLID/DRY/KISS/YAGNI review. Every finding carries file:line evidence, a category, a confidence level, and a suggested fix. No production code was modified.
>
> Cross-refs: module surface in [05_module_inventory.md](05_module_inventory.md); coupling/cycles in [06_dependency_graph.md](06_dependency_graph.md); items feed [15_technical_debt.md](15_technical_debt.md), [16_refactoring_plan.md](16_refactoring_plan.md), [18_quick_wins.md](18_quick_wins.md).

## 0. Summary

Code quality is **good** for a project of this size. The repository ships a CI-enforced Ruff profile (`E,F,I,UP`) that passes with **zero diagnostics**, and mypy `strict` is **clean** across all 50 source files. The real findings are:

- **Typing debt is mostly illusory.** 9 of the 10 modules carved out via `ignore_errors=true` already pass `--strict`; only `bot_messages.py` blocks it, via a single `state: object` annotation (20 errors, one root cause). The exclusion list is far larger than necessary. **(High confidence, quick win.)**
- **A few genuinely complex functions** in the generation pipeline (`_generate_text_once` C901=36, `_select_contextual_state`=29, `load_settings`=25) concentrate branching and are the main maintainability risk.
- **Localized DRY violations**: a verbatim init-guard repeated ~12× in `database.py`, and two `parse_bool` implementations with different contracts.
- **One dead-code item** (`get_random_pivo_message`, test-only) and **non-injectable module-global `random`** in two modules (testability smell).

None of these are correctness or security bugs (those are tracked in [08_security.md]). They are maintainability/typing items.

## 1. Ruff

### 1.1 Project profile (CI gate) — clean
`ruff check app tools` with the committed config (`pyproject.toml`: `select = ["E","F","I","UP"]`, line-length 100, target py312):

```
All checks passed!
```

So **imports are ordered, no unused names/imports, no unused variables, no pyupgrade regressions, no E-class style violations** in the enforced set. This is a healthy baseline and should remain the merge gate.

### 1.2 Broad scan (`--select ALL`) — 902 diagnostics, classified
The broad run is for audit insight only — **not** a recommendation to enable `ALL` (the TZ explicitly says don't blindly fix every lint warning). Classification by engineering impact:

| Category | Rules (count) | Verdict |
|---|---|---|
| **Noise / by-design** | `RUF001/002/003` ambiguous-unicode (207) | Cyrillic glyphs in a Russian-language bot — expected, **ignore**. |
| **Docs (style only)** | `D100–D107`, `D205/D212/D401/D205` (~230) | Missing docstrings. Style; low priority. Don't gate. |
| **Style only** | `COM812` trailing-comma (85), `TC001/002/003` (48) | Cosmetic; the formatter/CI doesn't require them. |
| **Maintainability** | `C901` (7), `PLR0912/0915/0911` (15), `PLR0913` (9) | **Real signal** — see §3.1. |
| **Maintainability** | `TRY003`/`EM101`/`EM102` (122) | Inline exception messages. Stylistic; bulk-fixing has low payoff. |
| **Potential bug** | `B905` zip-without-strict (1), `BLE001` blind-except (2), `S110` try/except/pass (1) | **Review individually** — §2. |
| **Performance** | `PTH*` os.path→pathlib (4), `ASYNC240` (2) | Minor; §2. |
| **Security-flavored** | `S311` non-crypto random (23) | False alarm for gameplay/text randomness; the one place that needs crypto (`pivo` HMAC) uses `hmac`/`secrets` already. **Ignore.** |
| **Maintainability** | `FBT001/002` boolean-positional (34), `PLR2004` magic values (51) | Style/taste; low priority. |

**Recommendation:** keep the `E,F,I,UP` gate. Optionally add `B` (bugbear) and `C901` with a generous threshold to catch the §2/§3 items in CI; everything else is opt-in cleanup, not debt.

### 1.3 Potential-bug / perf items worth a look (§2 detail)

| # | Rule | Location | Category | Note |
|---|---|---|---|---|
| Q1 | `BLE001`+`S110` | `app/handlers/_helpers.py:25` | potential bug | `except Exception: pass` around the typing chat-action. Intentional (must not block reply) but **swallows everything silently** — should log at debug. **Medium conf.** |
| Q2 | `BLE001` | `app/filters/admin_or_owner.py:26` | acceptable | Catches `Exception` around `get_chat_administrators`, logs a warning, fails closed (`return False`). Defensible — network call. **Low priority.** |
| Q3 | `B905` | `app/core/context_state_matcher.py:278` | potential bug | `zip(left, right)` without `strict=`; here the shorter-string truncation is *intended* (prefix similarity), so add `strict=False` to make intent explicit. **Low.** |
| Q4 | `ASYNC240` | `app/infrastructure/migrator.py:82` | perf (minor) | `path.read_text()` (blocking) inside async during startup. One-shot at boot, tiny files → negligible. Note for [09_performance.md]/[10_async_review.md]. **Low.** |

## 2. mypy

### 2.1 As configured — clean
`mypy app` with the committed config (`strict` for `app.*`, `ignore_errors=true` for 10 named modules):

```
Success: no issues found in 50 source files
```

### 2.2 The `ignore_errors` list is over-broad — **key finding (Q5, High confidence)**
Re-running `--strict` against the 10 excluded modules with a config that drops the override:

```
Found 20 errors in 1 file (checked 10 source files)
```

All 20 errors are in **`app/presentation/bot_messages.py`** and share **one root cause**: `format_config_message(state: object, ...)` (line 56) annotates the runtime-state argument as `object`, so every `state.<field>` access is an `attr-defined` error. The other **9 modules** (`registry`, `runtime_config`, `runtime_state`, `settings`, `reply_policy`, `text`, `pivo`, `pivo_templates`, `database`) **already pass `--strict` cleanly.**

The `pyproject.toml` comment labels these as «НЕ ТРОГАТЬ / legacy без аннотаций», but that is no longer accurate — they are strict-clean today.

**Fix (quick win):** change `state: object` → `state: RuntimeState` (or a narrow `Protocol`/`runtime_field_names`-driven loop) in `bot_messages.py`, then delete the entire `ignore_errors` override block. This restores **full strict coverage across the whole `app` package** in one small change. Tracked in [18_quick_wins.md].

> Resolves the M1 parked item "effective strict coverage is partial".

## 3. Manual quality pass (SOLID / DRY / KISS / YAGNI)

### 3.1 Complexity hotspots (maintainability) — confirmed by `C901`/`PLR091x`

| Function | File:line | Metric | Note |
|---|---|---|---|
| `_generate_text_once` | `app/core/markov.py:1039` | C901=36, branches/statements high | Core generation step; deepest nesting in the codebase. Prime refactor target ([16]). |
| `_select_contextual_state` | `app/core/markov.py:638` | C901=29 | Contextual state selection; many branches for backoff/fuzzy paths. |
| `load_settings` | `app/config/settings.py:69` | C901=25, 134 statements | Env parsing + cross-validation in one function. Splitting per concern would help. |
| `on_text_message` | `app/handlers/learning.py:82/111` | C901=18 | The hot-path handler; branch-heavy (reply policy + learning + generation). |
| `weighted_next_choice` | `app/core/markov.py` | C901=16 | Sampling logic. |
| `save_message_and_update_model` | `app/infrastructure/database.py` | C901=11 | DB write + model update; mixes concerns. |
| `_format_time_value` | `app/services/pivo_message_builder.py:159` | C901=14 | Time formatting branch table. |

`markov.py` (1438 LOC) is the dominant complexity sink and remains the top refactor candidate (see [05_module_inventory.md] note and [15_technical_debt.md]). Recommend **extraction of pure sub-steps** rather than rewrites.

### 3.2 DRY violations (confirmed)

- **Q6 — repeated init-guard in `database.py` (Medium).** The pattern
  ```python
  if self.markov is None:
      raise RuntimeError("Database not initialized: call await Database.init() first")
  ```
  appears **~12× verbatim** (lines 185–244+, 20 `is None:` checks total). A single `_require_init()` helper or a `@requires_init` property/descriptor would collapse them and remove the god-object smell of scattered guards. Resolves M1 parked "10 repeated guards".

- **Q7 — duplicate `parse_bool` (Low/Medium).** `app/config/runtime_config.py:20` `parse_bool() -> bool | None` and `app/config/registry.py:21` `_parse_bool() -> bool` implement the same parsing with **different return contracts** (None-on-unknown vs raise). Consolidate to one parser (registry's) and have `runtime_config` adapt the contract. Resolves M1 parked item.

### 3.3 Dead code

- **Q8 — `get_random_pivo_message` (Medium, test-only).** `app/domain/pivo.py:118`. The only callers are `tests/test_pivo.py`; production `PivoService` builds messages directly via `pivo_message_builder` (the lazy import inside this function exists only to break a cycle — see [06_dependency_graph.md:78]). Either wire it as the single message-building entry point or remove it and its test. Confirms M1 parked item. — ✅ **RESOLVED 2026-07-07** (branch `refactor/simplify-core-modules`): removed `get_random_pivo_message` and the equally test-only `build_pivo_mentions` along with their tests; the cycle-breaking lazy import is gone with them.

### 3.4 Testability smell (non-injectable randomness)

- **Q9 — module-global `random` (Low/Medium).** `app/services/pivo_message_builder.py` (`random.choice` ×6) and `app/handlers/_helpers.py:23` (`random.randint`) call the module-global RNG directly, so output is **not deterministically testable** without monkeypatching `random`. Inject a `random.Random` instance (or a callable) to match the rest of the pipeline, which already threads RNG where it matters. Confirms M1 parked item.

### 3.5 SOLID / abstraction notes (Low)

- `Database` (`app/infrastructure/database.py`, 385 LOC) trends toward a **god object**: connection lifecycle + Markov model store + message store + COALESCE aggregation queries in one class. Mitigated by the repository layer above it, but a future split (connection vs. markov-store vs. message-store) is worth noting for [16_refactoring_plan.md]. core⇄infrastructure bidirectional coupling (markov→database, database→core.text) is documented in [06_dependency_graph.md] and re-flagged here.
- No evidence of over-engineering/unnecessary abstractions elsewhere — the `registry` single-source-of-truth pattern is a genuine DRY *win* and should be preserved.

## 4. ToB `modern-python` review

The Trail of Bits modern-python guidance (uv + ruff + ty, py312 idioms) is **already substantially met**: the project uses ruff with `UP` (pyupgrade) clean, targets py312, and `requirements.lock` is present. Residual modern-Python items overlap with §1.3 (`PTH*` os.path→pathlib in `migrator.py`/`tools`) — cosmetic, low priority. No `# type: ignore` debt of note. ty (Astral type checker) was not run separately because mypy strict already passes; switching tools is out of scope for the audit.

## 5. Prioritized quality backlog

| Priority | Item | Effort | Payoff |
|---|---|---|---|
| **P1 (quick win)** | Q5 — type `state` in `bot_messages.py`, drop `ignore_errors` block → full strict | XS | High (restores type safety on 10 modules) |
| **P2** | Q6 — extract `_require_init()` guard in `database.py` | S | Medium (DRY, readability) |
| **P2** | Q1 — log (debug) instead of silent `pass` in `_helpers.py` | XS | Medium (observability) |
| **P3** | Q4/Q3 — `_generate_text_once`/`_select_contextual_state` extraction | M–L | Medium (maintainability, see [16]) |
| **P3** | Q7 — consolidate `parse_bool` | S | Low/Medium |
| **P3** | Q9 — inject RNG into pivo builder / `_helpers` | S | Medium (testability) |
| ~~P4~~ ✅ | Q8 — ~~resolve dead `get_random_pivo_message`~~ **DONE 2026-07-07** (removed + `build_pivo_mentions` + tests) | XS | Low |
| **P4** | Q2/Q4 — blind-except review, os.path→pathlib | XS | Low |

All P1–P2 items are safe, mechanical, and covered by existing tests — see [18_quick_wins.md] once M6 is reached.
