# Design: Diff Extraction for Target Scope (Issue #11)

**Date:** 2026-04-24
**Milestone:** M1 - Core (git-only why)
**Depends on:** #2 (run_git wrapper — already shipped)

---

## Problem

The LLM pipeline needs the concrete text of what changed in a commit, not just the commit message. For symbol-scoped queries (e.g. "why did this function change?") we want a diff scoped to a specific file and optionally a line range, so we don't flood the context window with unrelated changes.

---

## Interface

```python
# src/why/diff.py

def get_commit_diff(
    sha: str,
    file: Path,
    line_range: tuple[int, int] | None,
    repo: Path,
    max_chars: int = 2000,
) -> str:
```

- Returns a raw diff string (unified diff format).
- Truncates output > `max_chars` with `"\n... [truncated]"` appended.
- Raises `GitError` (or subclass) on git failures that aren't gracefully handled.

---

## Approach: Two git commands, one per mode

### Whole-file mode (`line_range=None`)

```
git show --format= <sha> -- <file>
```

- `--format=` suppresses the commit header; stdout is pure unified diff.
- Returns the full file patch for that commit.

### Line-range mode (`line_range=(start, end)`)

```
git log -L <start>,<end>:<file> <sha>^!
```

- `git log -L` tracks the specified line range through history, showing exactly what changed in those lines for this commit.
- `<sha>^!` is shorthand for `<sha>^..<sha>` — limits output to this single commit vs its parent.
- Semantically correct: git tracks the range even if lines moved due to earlier insertions/deletions.

### Root commit fallback

`git log -L ... <sha>^!` fails when `sha` has no parent (root commit). In that case, fall back to `git show --format= <sha> -- <file>` (whole-file mode), ignoring `line_range`. Root commits always show all lines as additions, so the LLM still gets meaningful context.

**Detection:** Check if `git rev-parse <sha>^` exits non-zero, OR catch the `GitError` from the failing `git log -L` call and re-run as whole-file. The catch-and-retry approach avoids an extra subprocess for the common case.

---

## Truncation

After the git command succeeds:

```python
if len(output) > max_chars:
    output = output[:max_chars] + "\n... [truncated]"
```

Simple character-count truncation. Mid-hunk truncation is acceptable — callers are LLMs, not diff processors.

---

## Module structure

```
src/why/diff.py        # new module
tests/test_diff.py     # new test file
```

No changes to existing modules. `diff.py` imports only `run_git` from `why.git`.

---

## Tests

**Unit (mocked `run_git`):**
- Whole-file mode builds correct `git show` args
- Line-range mode builds correct `git log -L` args
- Truncation at exactly `max_chars` characters
- Truncation appends `"\n... [truncated]"`
- Output shorter than `max_chars` is returned verbatim

**Integration (real git repo via `make_git_runner` from `conftest`):**
- Known commit + known file → diff string contains expected `+`/`-` lines
- Line-range on a known commit → diff contains only the changed lines in range
- Root commit fallback → returns a non-empty diff without error

---

## What this is NOT

- No caching — callers decide whether to cache.
- No multi-file diffs — `file` is always a single `Path`.
- No binary file handling — git returns a binary notice string; that's fine.
