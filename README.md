# why

> git blame tells you who. `why` tells you why.

**Status:** early scaffolding — no functionality yet.

`why` is a CLI that explains why code is the way it is by mining git history and PR metadata, then synthesizing it with an LLM.

See the full design: [`docs/design/why-idea-04-18-2026-1.0.0.md`](docs/design/why-idea-04-18-2026-1.0.0.md).

## Install

Not yet published to PyPI. For now, install from source — see Development below.

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests
mypy src
```

## License

MIT — see [LICENSE](LICENSE).
