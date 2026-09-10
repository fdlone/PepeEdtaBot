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

The active plugin set lives in the *user*-scope `~/.claude/settings.json`
(`enabledPlugins`) — this project ships no `.claude/settings.json` of its own,
only a `.claude/settings.local.json` holding permissions. So the list below is
what a developer is expected to have, not something the repo enforces. This
section says what each one is *for* — keep it in sync with what is actually
installed (`claude plugin list`).

Verified against the installed set on 2026-09-10.

Always on:

| Plugin / skill | Use it for |
|---|---|
| `.claude/skills/openspec-*` + `/opsx:*` | The change workflow: propose → apply → sync → archive. Every non-trivial change starts here. |
| `code-review@claude-plugins-official` → `/code-review` | Reviewing a PR before merge. |
| `ponytail@ponytail` | Over-engineering brake: `/ponytail-review` on a diff, `/ponytail-audit` on the repo, `/ponytail-debt` for `ponytail:` markers. Serves the "prefer minimal safe changes" rule above. |
| `property-based-testing@trailofbits` | Designing tests for the generation path — scoring, stemming, serialization, anything with an invariant worth stating. Property tests use `hypothesis` (dev dependency). |
| `superpowers@claude-plugins-official` | Process skills: brainstorming before a plan, systematic debugging before a fix, TDD. |
| `code-simplifier@claude-plugins-official`, `claude-md-management@claude-plugins-official`, `context7@claude-plugins-official` | Simplification passes, keeping this file and `CLAUDE.md` honest, and fetching current library docs instead of answering from memory. |
| `tools/eval` (in-repo) | Markov 2.0R eval protocol: ablation matrix C0–CF, pre-registered gates, reports to `docs/eval_reports/`. Normative docs: `docs/v2/01–05`. Every 2.0R phase gates on it. |

Installed but of no use here: `frontend-design@claude-plugins-official` — no UI
beyond Telegram messages. `github@claude-plugins-official` was **disabled**
2026-09-10: its MCP server wants a `GITHUB_PERSONAL_ACCESS_TOKEN` that is set
nowhere, so it failed to connect in every session, and `gh` on the CLI (already
authenticated, scopes `repo`/`workflow`) is what this project actually uses for
PRs. Re-enable it only alongside setting that variable.

Not installed, and the reasons are worth keeping:

| Plugin | Verdict |
|---|---|
| `astral@astral-sh` → `/astral:ruff`, `/astral:uv` | Was documented here as always-on but was never installed, and the `astral-sh/claude-code-plugins` marketplace has been dead upstream since 2026-02-27; removed from the known marketplaces 2026-09-10. Run `ruff` from `requirements-dev.txt` directly — that is what CI does. uv stays a local runner only: dependencies live in `requirements*.txt`, `[tool.uv] managed = false`, do not run `uv lock`/`uv sync`. |
| `pyright-lsp@claude-plugins-official` | Editor-time diagnostics only. CI type checking is mypy strict — pyright would never overrule `pyproject.toml`. Install it if you want it in the editor; nothing in the repo depends on it. |
| `security-guidance`, `static-analysis`, `insecure-defaults`, `sharp-edges`, `differential-review`, `fp-check`, `semgrep-rule-creator` | A security audit set. Not available from any configured marketplace; results, when such an audit happens, go to `openspec/audit-findings.md`. Day to day, CI already runs bandit and pip-audit. |
| `mutation-testing` | Would check whether the coverage ratchet (87%) reflects real assertions. Needs mewt/muton installed. |
| `modern-python` | Nothing here. It installs PATH shims that break plain `python -c` and pushes a uv/ty migration this project has explicitly declined. |
| `python-anti-patterns`, `python-testing-patterns` | Previously listed here via `skills-lock.json`; neither was ever in that lock nor on disk. The lock itself is gone as of 2026-09-10 — it pinned 817 `mukul975/Anthropic-Cybersecurity-Skills` entries plus nine `vercel-labs/agent-skills` ones, none of which apply to a Python Telegram bot, and every one of their names loaded into the skill index each session. `.claude/skills/` and `.agents/skills/` now hold the six `openspec-*` skills and nothing else. |

MCP: none required, and none configured for this project — the `codex` server
that used to sit in `.mcp.json` was removed 2026-09-10 as unused, and there is
no `.mcp.json` again.
