# Design — Issue #96: Auto-enable strict citation validation for local providers

**Issue:** [#96](https://github.com/tarungulati1988/why/issues/96)
**Branch:** `sleipnir/issue-96-strict-citations-local`
**Depends on:** #93 (provider attr), #94/#95 (provider plumbing) — all merged
**Scope:** small (~1-2h)

## Problem

3B-class local models hallucinate SHAs and PR numbers more often than 70B Groq. The pipeline already has `validate_citations(..., strict=True)` that raises on hallucinations, but it's opt-in and never wired into the CLI. Default is `strict=False` (warn-only), which produces output that *looks* grounded but isn't.

We want strict citations auto-on whenever provider is `openai-compatible`, with an explicit opt-out.

## Decisions

### 1. New CLI flag `--no-strict-citations`

Boolean opt-out. Default unset. Resolution:

```python
strict = (llm.provider == "openai-compatible") and not no_strict_citations
```

groq stays warn-only regardless of flag (matches current behavior; flag is a no-op there). The flag is opt-out (rather than positive `--strict-citations`) because the auto-on behavior is the new default for local; users only need to flip it off.

### 2. Typed exception `CitationError`

`validate_citations` currently raises `ValueError("citation validation failed: N issues")` in strict mode. Replace with a typed `CitationError(ValueError)` that carries `issues: list[ValidationIssue]` so the CLI can render a friendly message naming the offending value.

```python
class CitationError(ValueError):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        # Use first issue for primary message; full list available via .issues
        first = issues[0]
        super().__init__(f"unknown {first.kind.removeprefix('unknown_')}: {first.value}")
```

`ValueError` subclass keeps backwards compatibility for any callers (tests) currently doing `except ValueError`.

### 3. Friendly CLI error

Wrap `CitationError` in `cli.py`:

```python
except CitationError as exc:
    click.echo(
        f"⚠ Local model hallucinated citation: {exc}. "
        f"Try --no-strict-citations to allow, or switch to a larger model.",
        err=True,
    )
    sys.exit(1)
```

The `except (LLMError, GitError, SymbolNotFoundError)` block stays — `CitationError` gets its own clause *before* it (more specific).

### 4. No change to `synthesize_why` signature

`strict: bool = False` already exists. CLI just passes the resolved value. Keeps the synth API stable.

## File layout

```
src/why/citations.py    Add CitationError class; raise it instead of ValueError
src/why/cli.py          Add --no-strict-citations flag; resolve strict; catch CitationError
tests/test_citations.py Extend: CitationError raised, carries issues
tests/test_cli.py       New tests: auto-on for openai-compatible, off for groq, opt-out works,
                        friendly error printed
```

## Acceptance

- `provider=openai-compatible`, no flag → `strict=True` passed to `synthesize_why`.
- `provider=openai-compatible`, `--no-strict-citations` → `strict=False`.
- `provider=groq` → `strict=False` regardless of flag.
- `CitationError` raised in strict mode, carries `issues` list.
- CLI prints friendly message + exit 1 on `CitationError`.

## Strides

```
Stride 1: CitationError typed exception | test: tests/test_citations.py | impl: src/why/citations.py | depends: []
Stride 2: --no-strict-citations flag + auto-on resolution | test: tests/test_cli.py | impl: src/why/cli.py | depends: [1]
Stride 3: Friendly error wrapping | test: tests/test_cli.py | impl: src/why/cli.py | depends: [1, 2]
```

Wave 1: Stride 1. Wave 2: Strides 2+3 (same file, dispatched sequentially).

## Out of scope

- Retry-with-feedback when citation hallucinated (issue's "interesting follow-up").
- Citation grounding for `--verify` two-pass.
- Per-issue partial recovery (just fail loudly).
