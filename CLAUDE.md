# CLAUDE.md

## Project

This repository is `PepeEdtaBot`: a Python Telegram bot project located locally in the current project folder and linked to:

https://github.com/fdlone/PepeEdtaBot

The project is expected to evolve, so maintainability, clarity, safety, and long-term extensibility are high priorities.

## Communication rules

- Always communicate with the user in Russian.
- All explanations of actions, findings, risks, decisions, and results must be written in Russian.
- Source code, identifiers, function names, class names, variable names, comments, docstrings, tests, commit messages, branch names, configuration names, and technical project artifacts should be written in English unless the user explicitly asks otherwise.
- Do not hide important risks or assumptions. If something is uncertain, say so clearly in Russian.

## Session start procedure

At the beginning of every new session, before making changes:

1. Inspect the current repository state:
   - current branch;
   - `git status`;
   - recent commits if useful;
   - project structure;
   - dependency files;
   - configuration files;
   - environment examples;
   - test setup;
   - existing documentation.

2. Check attribution settings to prevent accidental AI-authored commits:
   - Check `~/.claude/settings.json` for `"includeCoAuthoredBy": false` and empty `attribution.commit` / `attribution.pr`.
   - If the settings file is missing or the values are not set correctly, **stop and warn the user in Russian before making any commits**. Do not proceed with commits until the user confirms the issue is resolved or explicitly accepts the risk.
   - Also scan the 5 most recent commits for any `Co-Authored-By: Claude` or `Generated with Claude Code` lines:
     ```
     git log -5 --format="%H %s" | head -20
     git log -5 --format="%B" | grep -i "co-authored\|claude code\|🤖"
     ```
   - If traces are found, report them to the user in Russian and ask whether to proceed or clean up first.

## Attribution & Commit Hygiene

- Do NOT include `Co-Authored-By` lines in any commit messages.
- Do NOT add `Generated with Claude Code` footers to commits or PRs.
- Do NOT add any AI attribution, signatures, or AI-related metadata anywhere.
- Commits must appear as authored solely by the human developer.

## Workflow

- All subsequent project work MUST go through the OpenSpec workflow: propose → apply → sync → archive (`/opsx:*` skills, artifacts in `openspec/`).
- Every non-trivial change starts with an OpenSpec proposal before any code is written.
- Trivial mechanical fixes (typos, formatting) may skip OpenSpec, but when in doubt, use it.

## Coding rules

- Prefer simple, readable, maintainable Python code.
- Keep business logic separated from Telegram handlers where practical.
- Avoid large monolithic functions.
- Avoid hardcoded secrets, tokens, chat IDs, user IDs, passwords, or environment-specific paths.
- Use environment variables or configuration files for runtime settings.
- Keep sensitive data protected.
- Do not introduce unnecessary dependencies.
- If adding a dependency, explain why it is needed.
- Preserve existing behavior unless the user asked to change it or a fix is clearly required.
- Make small, reviewable changes.
- Do not perform broad rewrites unless there is a strong reason.

## Security and privacy rules

- Treat bot tokens, API keys, user identifiers, chat metadata, and private user data as sensitive.
- Never print secrets in logs.
- Never commit real secrets.
- Validate and sanitize user-controlled input where relevant.
- Be careful with database queries and file operations.
- If sensitive user data is stored, prefer encryption and clear access boundaries.
- If privacy-related behavior is changed, update the relevant user-facing text.

## Testing and validation

After making changes, run the most relevant available checks, for example:

- unit tests;
- linting;
- type checks;
- import checks;
- bot startup checks;
- targeted manual validation where automated tests are absent.

If tools such as pytest, ruff, mypy, or other checks are not configured, do not invent results. Clearly state in Russian what was and was not available.

## Git rules

- Before changing files, check git status.
- Avoid modifying unrelated files.
- Do not discard user changes.
- Do not force push.
- Do not rewrite history unless the user explicitly asks.
- If there are uncommitted changes not made by you, identify them and avoid overwriting them.

## Response format to the user

When reporting progress or final results to the user:

- Write in Russian.
- Be concise but specific.
- Mention what was checked.
- Mention what was changed.
- Mention what files were affected.
- Mention what tests/checks were run.
- Mention any remaining risks or unfinished items.
- If something could not be completed, explain why clearly.

Preferred final response structure:

```
Готово.

Что сделал:
- ...

Изменённые файлы:
- ...

Проверки:
- ...

Осталось / рекомендуется:
- ...
```

## Important behavior

- Do not claim that checks passed if they were not run.
- When in doubt, preserve information and add clarification instead of deleting it.
