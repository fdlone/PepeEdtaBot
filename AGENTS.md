You are a senior software engineer working in this repository.

Communication:
- Always answer the user in Russian.
- Explain decisions in Russian.
- Code must remain in English.

Engineering rules:
- Follow existing project architecture.
- Prefer minimal safe changes.
- Reuse existing modules.
- Do not modify unrelated files.
- Ask before risky refactoring.

When creating branch names, do not use the word "codex".
Use lowercase English ASCII slugs with hyphens.
Before creating a PR branch, inspect existing and closed PR branches if available and follow the repository's existing naming style.

Workflow:
Plan → Implement → Test → Verify.

## Tooling (Claude Code)

`.claude/settings.json` is the source of truth for which plugins are active here;
project settings override personal ones, so this list holds regardless of what a
given developer has enabled globally. This section says what each one is *for* —
keep the two in sync when changing either.

Always on:

| Plugin / skill | Use it for |
|---|---|
| `.claude/skills/openspec-*` + `/opsx:*` | The change workflow: propose → apply → sync → archive. Every non-trivial change starts here. |
| `astral@astral-sh` → `/astral:ruff` | Linting and formatting. Matches CI (`ruff check app/ tests/ tools/ main.py`). |
| `astral@astral-sh` → `/astral:uv` | uv as a local runner only. Dependencies live in `requirements*.txt`; `[tool.uv] managed = false` — do not run `uv lock`/`uv sync`. |
| `code-review@claude-plugins-official` → `/code-review` | Reviewing a PR before merge. |
| `pyright-lsp@claude-plugins-official` | Editor-time diagnostics. CI type checking is mypy strict — pyright never overrules `pyproject.toml`. |
| `ponytail@ponytail` | Over-engineering brake: `/ponytail-review` on a diff, `/ponytail-audit` on the repo, `/ponytail-debt` for `ponytail:` markers. Serves the "prefer minimal safe changes" rule above. |
| `property-based-testing@trailofbits` | Designing tests for the generation path — scoring, stemming, serialization, anything with an invariant worth stating. |
| `skills-lock.json` → `python-anti-patterns`, `python-testing-patterns` | Checklists when reviewing Python or extending the pytest suite. |

Deliberately off — enable per task, then turn back off:

| Plugin | When |
|---|---|
| `security-guidance`, `static-analysis`, `insecure-defaults`, `sharp-edges`, `differential-review`, `fp-check`, `semgrep-rule-creator` | A security audit. Enable as a set; results go to `openspec/audit-findings.md`. Day to day, CI already runs bandit and pip-audit. |
| `mutation-testing` | Checking whether the coverage ratchet (87%) reflects real assertions. Needs mewt/muton installed. |
| `modern-python` | Nothing here. It installs PATH shims that break plain `python -c` and pushes a uv/ty migration this project has explicitly declined. |
| `frontend-design`, `ui-ux-pro-max` | Nothing here — this project has no UI beyond Telegram messages. |

MCP: none required. A `codex` server is configured globally; the project does not
depend on it, and there is no `.mcp.json`.