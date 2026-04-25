"""Synthesis pipeline orchestration for the why CLI."""

from __future__ import annotations

import logging
from pathlib import Path

from why.diff import get_commit_diff
from why.git import GitError
from why.history import get_file_history, get_line_history
from why.llm import LLMClient
from why.prompts import WHY_SYSTEM_PROMPT, CommitWithPR, build_why_prompt
from why.scoring import select_key_commits
from why.symbols import SymbolNotFoundError, find_symbol_range
from why.target import Target

_log = logging.getLogger(__name__)

# Lines of context to include above and below a bare line target.
_LINE_WINDOW = 20


def _extract_current_code(target: Target, line_range: tuple[int, int] | None = None) -> str:
    """Return the relevant source text for *target*.

    Priority order:
    1. Symbol — use the pre-resolved line_range if provided; fall back to full
       file when line_range is None (symbol not found).
    2. Line — return a ±_LINE_WINDOW window around the given line, clamped
       to file bounds.
    3. File-only — return the entire file text.
    """
    if target.symbol is not None:
        if line_range is None:
            return target.file.read_text()
        start, end = line_range
        lines = target.file.read_text().splitlines(keepends=True)
        return "".join(lines[start - 1 : end])

    if target.line is not None:
        lines = target.file.read_text().splitlines(keepends=True)
        # Convert 1-based line number to 0-based index, then build window
        idx = target.line - 1
        lo = max(0, idx - _LINE_WINDOW)
        hi = min(len(lines), idx + _LINE_WINDOW + 1)
        return "".join(lines[lo:hi])

    return target.file.read_text()


def _resolve_line_range(target: Target) -> tuple[int, int] | None:
    """Return the (start, end) line range (1-based, inclusive) for *target*.

    Priority order:
    1. Symbol — delegate to find_symbol_range; return None on SymbolNotFoundError.
    2. Line — return (line, line).
    3. File-only — return None (no narrowing possible).
    """
    if target.symbol is not None:
        try:
            return find_symbol_range(target.file, target.symbol)
        except SymbolNotFoundError:
            return None

    if target.line is not None:
        return (target.line, target.line)

    return None


def synthesize_why(
    target: Target,
    repo: Path,
    llm: LLMClient,
    prs: dict[str, str] | None = None,
) -> str:
    """Orchestrate the full why pipeline and return the LLM's explanation.

    Steps:
      1. Normalise prs to an empty dict when callers pass None.
      2. Fetch the relevant commit history (line-scoped when target.line is set,
         file-scoped otherwise).
      3. Short-circuit with a sentinel string when there are no commits.
      4. Skip scoring for very sparse histories (< 3 commits) — there is nothing
         meaningful to rank.
      5. Fetch per-commit diffs, narrowed to the target's line range if available.
      6. Extract the current source text for the target region.
      7. Pair each key commit with its PR body and diff.
      8. Build the prompt and call the LLM.
    """
    prs = prs or {}

    # Resolve the line range once here to avoid duplicate tree-sitter parses.
    line_range = _resolve_line_range(target)

    # Route to the correct history function using the already-resolved line_range.
    if target.symbol is not None:
        if line_range is not None:
            history = get_line_history(target.file, repo, line=line_range[0])
        else:
            history = get_file_history(target.file, repo)
    elif target.line is not None:
        history = get_line_history(target.file, repo, line=target.line)
    else:
        history = get_file_history(target.file, repo)

    if not history:
        return "file has no git history"

    # Sparse path: too few commits to score meaningfully — use all of them.
    key_commits = history if len(history) < 3 else select_key_commits(history, prs)

    # Sort oldest-first so the LLM sees chronological order regardless of path.
    key_commits = sorted(key_commits, key=lambda c: c.date)

    # Fetch per-commit diffs; swallow GitError on individual commits rather than
    # aborting the whole call.
    diffs: dict[str, str] = {}
    for c in key_commits:
        try:
            diffs[c.sha] = get_commit_diff(c.sha, target.file, line_range, repo)
        except GitError as exc:
            _log.warning("could not fetch diff for %s: %s; using empty diff", c.sha[:12], exc)
            diffs[c.sha] = ""

    current_code = _extract_current_code(target, line_range)

    commits_with_prs = [
        CommitWithPR(commit=c, pr_body=prs.get(c.sha), diff=diffs[c.sha])
        for c in key_commits
    ]

    messages = build_why_prompt(target, current_code, commits_with_prs)
    return llm.complete(WHY_SYSTEM_PROMPT, messages)
