# why

> git blame tells you who. `why` tells you why.

**Status:** M1 in progress — CLI and synthesis pipeline wired.

`why` is a CLI that explains why code is the way it is by mining git history and PR metadata, then synthesizing it with an LLM.

See the full design: [`docs/design/why-idea-04-18-2026-1.0.0.md`](docs/design/why-idea-04-18-2026-1.0.0.md).

## Install

Not yet published to PyPI. For now, install from source — see Development below.

## Usage

### Prerequisites

Set your Groq API key (default LLM backend):

```bash
export GROQ_API_KEY=your_key_here
```

By default `why` uses Groq as the LLM provider. The active provider is controlled by `WHY_LLM_PROVIDER` (default: `groq`). Currently only `groq` is supported — other providers are planned.

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
