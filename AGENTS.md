# AGENTS.md

> Guide for AI coding agents (Claude Code, Cursor, Aider, etc.) working in the `why` repo. Humans should read [README.md](README.md) first.

## What this project is

`why` is a Python 3.11+ CLI (`pip install git-why` → `why` command) that explains why code is the way it is by mining git history and PR metadata, then synthesizing an explanation with an LLM. The default cloud provider is Groq; any OpenAI-compatible local server (Ollama, llama.cpp, LM Studio, vLLM, TGI) also works. See [README.md](README.md) for install and usage, and [docs/architecture.md](docs/architecture.md) for the full pipeline map.

## Where to start

1. Read [docs/architecture.md](docs/architecture.md) for the pipeline map, module roles, and key invariants.
2. Open the module relevant to the task — every file in `src/why/` has a structured module-level docstring describing its stage, inputs, outputs, and invariants.
3. Tests live in `tests/` mirroring `src/why/` — `test_<module>.py`.
4. Decision history is in `docs/designs/` (per-feature design docs) and `docs/sleipnir/` (Sleipnir agent design issues, numbered by GitHub issue).

## Repo layout

```
why/
├── src/why/               # All source code
│   ├── _backends/         # Provider dispatch (base.py, openai_compatible.py)
│   ├── _errors.py         # Shared exception hierarchy
│   ├── cli.py             # Entry point; wires flags to synthesize_why
│   ├── synth.py           # Orchestration — synthesize_why() calls every stage
│   ├── llm.py             # LLMClient, GroqBackend, retry logic
│   ├── citations.py       # Citation validation; raises CitationError
│   ├── prompts.py         # Prompt builders; pure functions
│   ├── scoring.py         # Commit ranking; pure functions
│   ├── timeline.py        # Timeline validation/repair; pure functions
│   └── ...                # target, history, commit, diff, symbols, render, github, cache
├── tests/                 # test_<module>.py mirrors src/why/
│   ├── fixtures/          # Real git repos used by history/diff/symbol tests
│   ├── conftest.py
│   └── _helpers.py        # LLM mock helpers
├── docs/                  # Engineering docs (architecture, designs, sleipnir state)
│   ├── architecture.md    # Canonical pipeline map for agents
│   ├── designs/           # Per-feature design documents
│   └── sleipnir/          # Issue-tagged design docs + state.json
├── website/               # Jekyll source for the GitHub Pages site (https://www.why.ai)
├── .github/workflows/     # ci.yml (test + lint + typecheck), pages.yml
├── pyproject.toml
├── README.md
└── AGENTS.md              # This file
```

## Commands

```sh
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests + checks (all three must pass before claiming done)
pytest                    # full test suite
ruff check src tests      # lint
mypy src                  # strict type-check

# Run the CLI locally
why <file>                # requires GROQ_API_KEY

# Single test file while iterating
pytest tests/test_<module>.py -v
```

## Conventions

- Python 3.11+, strict mypy. New code must pass `mypy src` with zero `# type: ignore` additions.
- Public-facing CLI flags require a corresponding test in `tests/test_cli.py`.
- LLM-touching code uses `LLMClient` (`src/why/llm.py`) — never instantiate `groq.Groq` or `openai.OpenAI` directly outside `_backends/`.
- Strict citation mode is auto-enabled for the `openai-compatible` provider and can be opted out with `--no-strict-citations`. For `groq`, validation still runs but issues are logged as warnings — there is no flag to make `groq` strict. The auto-enable lives in `cli.py` (search for `strict = (llm.provider == "openai-compatible")`).
- Pipeline is unidirectional: cli → target → history → commit → scoring → diff → prompts → llm → render. `synth.py` orchestrates the llm/citations/timeline call but is not a downstream stage.
- No I/O in `scoring.py`, `timeline.py`, or `prompts.py` — pure functions over data classes.
- New exception types belong in `src/why/_errors.py` to avoid circular imports.
- Inter-stage payloads use frozen dataclasses (`Target`, `Commit`, `CommitWithPR`, `ChatResult`).

## Style

- Default to no comments. Add a comment only when the WHY is non-obvious (constraint, invariant, workaround for a specific bug).
- Module-level docstrings exist on every file in `src/why/` — keep them in sync when you change a module's behavior.
- ruff selects `E, F, I, UP, B, SIM, RUF, PT`. Line length 100. Config is in `pyproject.toml`.
- Match existing patterns: small functions, dataclasses for inter-stage payloads, explicit error types in `_errors.py`.
- Do not add per-function docstrings unless the function's contract is genuinely non-obvious. Module-level docstrings are the documentation primitive here.

## Adding a feature

1. Skim [docs/architecture.md](docs/architecture.md) to identify which pipeline stage(s) the feature lives in.
2. Check `docs/designs/` and `docs/sleipnir/` for prior design context on the feature area.
3. Write or extend tests in `tests/test_<module>.py` first (TDD).
4. Implement, keeping changes inside the relevant pipeline stage where possible.
5. Run `pytest && ruff check src tests && mypy src`. All three must pass.
6. If the feature adds a CLI flag: update `cli.py`, `tests/test_cli.py`, the `README.md` Usage section, and the relevant page under `website/`.

## Testing notes

- All tests are in `tests/`; run the full suite before declaring done.
- Tests for history, diff, and symbol resolution use real `git` against fixture repos in `tests/fixtures/`.
- LLM calls are mocked — see `tests/_helpers.py` and `tests/conftest.py`.
- Tree-sitter grammars are loaded eagerly in `symbols.py`. Symbol-scoping tests must not depend on network access.

## Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Required for the default `groq` provider. |
| `WHY_LLM_PROVIDER` | `groq` (default) or `openai-compatible`. |
| `WHY_LLM_BASE_URL` | OpenAI-compatible endpoint (required when provider is `openai-compatible`). |
| `WHY_LLM_API_KEY` | Optional API key for local servers; most ignore it. |
| `WHY_LLM_MAX_CTX` | Prompt token budget. Defaults to `4096` for `openai-compatible`; disabled for `groq`. Set to `0` to disable auto-shrink. |
| `WHY_LLM_NUM_CTX` | Ollama KV-cache size. Auto-couples to `WHY_LLM_MAX_CTX` when unset; ignored for `groq`. |
| `GITHUB_TOKEN` | Optional; falls back to `gh auth token`. Needed for private repos or heavy PR-metadata use. |

## Docs

- User-facing manual: https://www.why.ai (built from `website/` via `.github/workflows/pages.yml`; custom domain pinned in `website/CNAME`). WIP — grows per release.
- Architecture map for agents: [docs/architecture.md](docs/architecture.md).
- Per-feature design docs: `docs/designs/` (chronological) and `docs/sleipnir/` (issue-tagged).
- Master idea doc: `docs/design/why-idea-04-18-2026-1.0.0.md`.

## Hard rules for agents

These override the agent's defaults:

- **Do not bypass safety checks.** No `--no-verify`, no skipped pre-commit hooks, no force-push to main.
- **Do not add backwards-compatibility shims** for code that has only been on main for one release. This project is pre-1.0; rewrite, don't shim.
- **Do not introduce new top-level dependencies** without a design note in `docs/sleipnir/` or `docs/designs/`. The dependency surface is intentionally small: `click`, `groq`, `openai`, `rich`, `tree-sitter`.
- **Citations are load-bearing.** Never relax `citations.py` checks to make a test pass. If a test fails because of citation validation, the LLM output fixture is wrong — fix the fixture, not the validator.
- **`synthesize_why` is the single integration seam.** Do not call pipeline stage functions directly from `cli.main()`; route through `synth.synthesize_why`.
- **Do not write per-function docstrings** unless the function's contract is genuinely non-obvious. Module-level docstrings are the documentation unit.
