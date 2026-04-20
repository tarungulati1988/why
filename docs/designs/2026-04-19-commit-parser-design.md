# Commit dataclass + porcelain parser — design

**Ticket:** [#3](https://github.com/tarungulati1988/why/issues/3)
**Milestone:** M1 — Core (git-only why)
**Depends on:** #2 (merged — `src/why/git.py` wrapper)
**Branch:** `feat/commit-parser`

---

## Problem

`why` synthesises explanations from git history. Everything downstream (ranking, LLM
prompting, display) consumes a structured `Commit` object, not raw git output. We need
a typed representation and a deterministic parser that turns `git log` porcelain text
into `list[Commit]`.

This ticket is **parser-only**. The `git log` invocation itself is a future ticket.

## Goals

- `Commit` dataclass with the fields listed in the ticket acceptance criteria.
- `parse_porcelain(output: str) -> list[Commit]` that tolerates real-world messiness:
  multi-line bodies, special chars in subjects, empty bodies, merge commits (no shortstat),
  commits with only insertions or only deletions.
- Fails loudly (`ValueError`) on structurally malformed input — no silent skipping.
- mypy-strict clean, no new runtime deps.

## Non-goals

- Running `git log` (separate ticket).
- Caching / persistence of parsed commits.
- Per-file stats (`--numstat`) — only aggregate additions/deletions.
- Byte-level handling of non-UTF-8 author names (M1 accepts stdlib `str` only).

## Design

### `Commit`

Frozen dataclass in `src/why/commit.py`:

```python
@dataclass(frozen=True)
class Commit:
    sha: str
    short_sha: str          # derived: sha[:7]
    author_name: str
    author_email: str
    date: datetime          # parsed from %aI (ISO 8601 with offset)
    subject: str
    body: str
    parents: tuple[str, ...]
    additions: int = 0
    deletions: int = 0

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1
```

Notes:
- `parents` is a tuple (hashable, frozen-dataclass-friendly).
- `short_sha` is derived in `__post_init__` via `object.__setattr__` (frozen dataclass
  escape hatch). We don't require the caller to pass it.

### Format strings

```python
PORCELAIN_FORMAT = "%H%x00%an%x00%ae%x00%aI%x00%s%x00%b%x00%P\x1e"
SHORTSTAT_FLAG = "--shortstat"
```

Field separator `\x00` (NULL), commit separator `\x1e` (ASCII Record Separator). Two
distinct control chars means we stay deterministic even if a commit body contains
tabs, newlines, quotes, or other shell metacharacters.

### `parse_porcelain`

Pipeline:

1. Split `output` on `\x1e`. The trailing empty chunk (from the final separator) is
   dropped.
2. For each chunk:
   - `lstrip("\n")` — shortstat from the *previous* commit leaves a blank line before
     the next record starts.
   - Detect an optional trailing shortstat line via regex `r"^\s*\d+ files? changed"`.
     If present, peel it off.
   - Split the remaining block on `\x00`. Expect exactly 7 fields → else `ValueError`.
   - Parse `date` via `datetime.fromisoformat` (Python 3.11 handles timezone offsets
     natively).
   - Parse `parents` as `tuple(seventh_field.split())`.
   - If a shortstat line was peeled, extract insertions/deletions via one regex with
     two optional groups (a commit can have only inserts OR only deletes).
   - Construct `Commit`.

### Why single-pass (option 1)

The ticket constants `PORCELAIN_FORMAT` and `SHORTSTAT_FLAG` are designed to be used
together in a single `git log` invocation, so shortstat is already interleaved with
metadata in the same stdout blob. A single parser walks it once.

The alternative is **option 2**: run two git commands (`git log --format=...` and
`git log --shortstat --format=%H`), keep `parse_porcelain` metadata-only, add a
sibling `parse_shortstat`, zip results by SHA. Simpler individual parsers, twice the
git invocations.

**When we'd refactor to option 2:** if stats ever need to come from a *different source*
than the commit metadata. Concrete example — if we want per-file stats via
`git diff --numstat <parent> <sha>` for renames/binary files, or stats pulled from a
cached stats database that doesn't run git at all. At that point, metadata and stats
diverge and the single-pass design becomes an obstacle. This rationale will be
captured as a comment in `commit.py`.

### Logging

Use `logging.getLogger(__name__)`. `DEBUG` level only — parsing is a hot path and
should be silent under normal use. Three log sites:

- Start: `logger.debug("parse_porcelain: %d chunks", len(chunks))`
- Malformed chunk (before raising): `logger.debug("malformed chunk at index %d: %r", i, chunk)`
- End: `logger.debug("parse_porcelain: parsed %d commits (%d merges)", n, m)`

No `print`. Consumers flip logging on via `logging.basicConfig(level=logging.DEBUG)`
or a future CLI `--verbose` flag.

### Error handling

- Wrong field count → `ValueError` with chunk index and the received count.
- Unparseable date → let `datetime.fromisoformat` raise `ValueError` naturally.
- Unparseable shortstat → `ValueError` (shouldn't happen with real git output; if it
  does, we want to see it).

## Tests — `tests/test_commit.py`

Unit (fixture-based):
- Normal commit with body spanning multiple lines.
- Merge commit (>1 parent, no shortstat line).
- Commit with empty body.
- Commit with special chars in subject (newlines, tabs, `|`, quotes).
- Commit with only additions (`" 1 file changed, 5 insertions(+)"`).
- Commit with only deletions (`" 1 file changed, 3 deletions(-)"`).
- Malformed input (wrong field count) → `pytest.raises(ValueError)`.
- `short_sha == sha[:7]`.
- `is_merge` true iff `len(parents) > 1`.

Integration (real git, reusing the `git_repo` fixture pattern from `test_git.py`):
- Build a tmp repo with a normal commit + a merge commit.
- Run `git log --format=<PORCELAIN_FORMAT> --shortstat`.
- Feed stdout through `parse_porcelain`.
- Assert count, SHAs round-trip, merge detection, additions/deletions match the diff.

## Rollout

Single PR, no feature flag. The module is unused by production code until the next
ticket (`git log` runner) wires it in.

## Risks

- **Non-UTF-8 author names.** Mitigation: `str`-only input in M1; upgrade to bytes-aware
  parsing only if/when a user reports a `UnicodeDecodeError` from `run_git`.
- **Git version drift in shortstat wording.** The regex `r"\d+ files? changed"` is
  stable across modern git (2.x+). If CI ever runs against an ancient git, the
  integration test will flag it.
