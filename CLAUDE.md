# CLAUDE.md

Project-specific instructions for Claude Code in the `why` repo.

> **Most of the substance is in [AGENTS.md](AGENTS.md).** Read that first — it covers architecture, commands, conventions, and hard rules. This file only contains Claude-Code-specific notes.

## Tooling

Always run from the repo root. Quality gate before claiming any task done:

```sh
pytest && ruff check src tests && mypy src
```

All three must be clean. Do not declare done on "worker said PASS" — verify by reading the run output yourself.

## Sleipnir workflow (this repo uses it)

This repo runs the Sleipnir multi-agent workflow. Active state lives in `docs/sleipnir/state.json`. When resuming work mid-flight, read that file first.

- Brainstorm phase: produce 2-3 holistic approaches, then a design doc under `docs/sleipnir/`.
- Execution phase: dispatch sleipnir-coder workers; never edit files directly as the orchestrator.
- Review phase: 3 personas (bug, security, practices); mandatory before PR.
- PR phase: use the What / Why / How / Test Steps template (see below).

## PR template

When opening a PR, the body MUST follow this shape:

```
## Related Links
## What
## Why
## How
## Test Steps
## Other Notes
```

Do not use the generic `## Summary` / `## Test plan` shape.

## Things to never do in this repo

- Skip pre-commit hooks (`--no-verify`).
- Force-push to `main`.
- Add a top-level dependency without a design note in `docs/sleipnir/` or `docs/designs/`.
- Bypass `citations.py` validation to make a test pass — fix the fixture, not the validator.
- Write per-function docstrings everywhere; module-level docstrings are the documentation primitive (see existing `src/why/*.py`).

## Memory / context budget

Module docstrings cover the per-file context. `docs/architecture.md` covers the cross-module map. Read these before exploring code with grep — they often answer the question.
