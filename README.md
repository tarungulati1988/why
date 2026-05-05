# why

> git blame tells you who. `why` tells you why.

> **Docs:** Manual at https://www.why.ai (WIP). Architecture map: [`docs/architecture.md`](docs/architecture.md). Working with AI agents in this repo? See [`AGENTS.md`](AGENTS.md).

**Status:** Active development — release pipeline configured; first publish triggered on tag `v0.0.1`.

`why` is a CLI that explains why code is the way it is by mining git history and PR metadata, then synthesizing it with an LLM.

See the full design: [`docs/design/why-idea-04-18-2026-1.0.0.md`](docs/design/why-idea-04-18-2026-1.0.0.md).

## Install

### Homebrew (macOS — recommended)

```sh
brew tap tarungulati1988/why
brew install why
```

> **Note:** The `tarungulati1988/why` tap must be set up first — see [homebrew-why](https://github.com/tarungulati1988/homebrew-why). This becomes available after the first tagged release.

### pip / uv

```sh
pip install git-why
# or
uv tool install git-why
```

The PyPI package is `git-why`; the CLI command installed is `why`.

### From source

```sh
git clone https://github.com/tarungulati1988/why.git
cd why
pip install -e ".[dev]"
```

See [Development](#development) for the full local setup.

### After installing — set your API key

```sh
export GROQ_API_KEY=your_key_here
```

## Running on a local LLM

`why` ships with a generic OpenAI-compatible provider that works with Ollama, llama.cpp's `server`, LM Studio, vLLM, and HuggingFace TGI. Useful when you want to keep your code off cloud providers, or just want to try `why` without an API key.

### Quickstart with Ollama (8GB RAM)

```sh
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a small model
ollama pull qwen2.5:3b
# 3. Start the server (Ollama runs on :11434 by default)
ollama serve
```

```sh
export WHY_LLM_PROVIDER=openai-compatible
export WHY_LLM_BASE_URL=http://localhost:11434/v1
why --model qwen2.5:3b src/why/llm.py
```

That's it — `WHY_LLM_MAX_CTX` defaults to `4096` for `openai-compatible`, and `WHY_LLM_NUM_CTX` auto-couples to it, so Ollama's `num_ctx=2048` ceiling is raised to `4096` automatically.

### Recommended local models

| Model         | Params | Download | Fits 8GB RAM | Notes                                  |
|---------------|--------|----------|--------------|----------------------------------------|
| `qwen2.5:3b`  | 3.1B   | ~1.9 GB  | yes          | Strongest instruction-following at 3B. |
| `phi3:mini`   | 3.8B   | ~2.3 GB  | yes          | Good structured-output behavior.       |
| `llama3.2:3b` | 3.2B   | ~2.0 GB  | yes          | Solid generalist.                      |
| `qwen2.5:7b`  | 7.6B   | ~4.7 GB  | tight        | Better quality if you have headroom.   |

### Environment variables

| Variable           | Default                                | Purpose                                                       |
|--------------------|----------------------------------------|---------------------------------------------------------------|
| `WHY_LLM_PROVIDER` | `groq`                                 | Set to `openai-compatible` to use a local model.              |
| `WHY_LLM_BASE_URL` | (required for `openai-compatible`)     | OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1`. |
| `WHY_LLM_API_KEY`  | `not-needed`                           | Most local servers ignore this; some require any non-empty.   |
| `WHY_LLM_MAX_CTX`  | `4096` (local), disabled (groq)        | Prompt token budget; oldest commits dropped to fit.           |
| `WHY_LLM_NUM_CTX`  | unset (auto-couples to `WHY_LLM_MAX_CTX`) | Ollama-only: raises model context window above its 2048 default. |

See the [Usage](#usage) section below for the full semantics of each variable.

### Quality caveat

Local 3B models produce useful drafts but not Groq-70B-quality narratives. Two guardrails kick in automatically when provider is `openai-compatible`:

- **Auto-shrink** drops the oldest commits and truncates long diffs so the prompt fits the context window. A single warning describes what was dropped.
- **Strict citations** are auto-enabled — if the model invents a SHA or PR number, `why` fails loudly with a friendly error rather than emitting hallucinated references. Use `--no-strict-citations` to allow it.

## Usage

### Prerequisites

Set `GROQ_API_KEY` before running (see [Install](#install) above).

By default `why` uses Groq as the LLM provider. The active provider is controlled by `WHY_LLM_PROVIDER` (default: `groq`). Two providers are currently supported:

- **`groq`** (default) — Groq cloud API; requires `GROQ_API_KEY`.
- **`openai-compatible`** — any OpenAI-compatible local server (Ollama, llama.cpp, LM Studio, vLLM, TGI, …); requires `WHY_LLM_BASE_URL`. Example with Ollama:

  ```sh
  WHY_LLM_PROVIDER=openai-compatible \
  WHY_LLM_BASE_URL=http://localhost:11434/v1 \
  why --model qwen2.5:3b ...
  ```

#### Auto-shrink for small context windows

Local 3B-class models often have small context windows (Ollama's default `num_ctx` is 2048). `why` auto-shrinks the prompt to fit:

- **`WHY_LLM_MAX_CTX`** — token budget for prompt assembly. Set to a positive integer to enable; set to `0` to disable.
- When unset, defaults to **`4096` for `openai-compatible`** providers and is **disabled for `groq`**.
- When shrinking fires, `why` drops the oldest commits first and truncates each remaining diff to 80 lines, then prints a single warning to stderr describing what was dropped.
- `--max-commits` is honored *before* shrinking (user cap wins).

#### Ollama context window (`num_ctx`)

`WHY_LLM_MAX_CTX` controls how much context `why` assembles client-side. `WHY_LLM_NUM_CTX` tells Ollama how large a KV cache to allocate server-side. They are kept in sync automatically — usually you only need to set `WHY_LLM_MAX_CTX`.

Ollama defaults to `num_ctx=2048` regardless of the model's actual capability. Other OpenAI-compatible servers (vLLM, llama.cpp, LM Studio, TGI) ignore unknown options and are unaffected.

- **`WHY_LLM_NUM_CTX`** — when set on `openai-compatible`, passed to the server as `extra_body={"options": {"num_ctx": N}}`. Must be a positive integer; `0`, negative, or non-integer values raise an error at startup.
- **Auto-couple to `WHY_LLM_MAX_CTX`** — when `WHY_LLM_NUM_CTX` is unset and a `WHY_LLM_MAX_CTX` is in effect (whether user-set or the auto-default `4096`), `why` uses that value as `num_ctx`. The auto-shrink budget and the server's allocated context stay matched without manual bookkeeping. (So with stock settings on openai-compatible, num_ctx is sent as 4096.)
- **Disable** — set `WHY_LLM_NUM_CTX` explicitly to override the auto-couple, or set `WHY_LLM_MAX_CTX=0` to disable shrinking entirely (which also disables auto-couple).
- **No effect on `groq`** — silently ignored.

### GitHub token (optional but recommended)

`why` fetches PR metadata from the GitHub API. It discovers a token automatically:

1. `GITHUB_TOKEN` env var — set this if you prefer explicit control
2. `gh` CLI — if you have [GitHub CLI](https://cli.github.com) installed and have run `gh auth login`, `why` picks up the token automatically
3. Unauthenticated — works for public repos but is rate-limited to 60 requests/hour

For private repos or heavy use, set a token explicitly:

```sh
export GITHUB_TOKEN=your_personal_access_token
```

A classic PAT with `repo` (read) scope is sufficient.

### Analyse a file

```bash
why src/auth/middleware.py
```

### Narrow to a line

```bash
why src/auth/middleware.py:42
```

### Narrow to a symbol

```bash
why src/auth/middleware.py handle_session_timeout
```

### Override the model

```bash
why --model llama-3.1-8b-instant src/auth/middleware.py
```

### Get a 3-sentence summary

```bash
why --brief src/auth/middleware.py        # concise mode: one paragraph
```

### Verify claim grounding (two-pass)

```bash
why --verify src/auth/middleware.py       # second LLM call checks intent claims
```

### Disable color / pipe-friendly output

```bash
why --no-color src/auth/middleware.py     # raw markdown, no ANSI
why src/auth/middleware.py | grep "added" # piping auto-disables rich
```

### Get help

```bash
why --help          # full reference with argument descriptions
why --version       # print version and exit
```

## Supported languages

Symbol-scoped analysis (`why <file> <symbol>`) uses tree-sitter and supports:

| Language | Extensions |
|----------|-----------|
| Python   | `.py`     |
| Go       | `.go`     |

File and line targets (`why <file>` and `why <file>:<line>`) work with any language — tree-sitter is only needed for symbol lookup.

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the full check suite:

```bash
pytest              # all tests
ruff check .        # lint
mypy src            # strict type check
```

Run a single test file while iterating:

```bash
pytest tests/test_git.py -v
```

### Working with AI coding agents

This repo includes [`AGENTS.md`](AGENTS.md) (cross-tool guide for Cursor, Aider, etc.) and [`CLAUDE.md`](CLAUDE.md) (Claude Code specifics). Both link to [`docs/architecture.md`](docs/architecture.md), which is the canonical pipeline map.

## License

MIT — see [LICENSE](LICENSE).
