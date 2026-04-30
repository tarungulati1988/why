# why

> git blame tells you who. `why` tells you why.

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
- **Ollama `num_ctx`:** Ollama's default `num_ctx` is 2048. Bump it (Modelfile `PARAMETER num_ctx 4096` or set per-request) to match `WHY_LLM_MAX_CTX`'s default of 4096 — otherwise the prompt still overflows.

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

## License

MIT — see [LICENSE](LICENSE).
