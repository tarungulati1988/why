"""Synthesis pipeline orchestration for the why CLI."""

from __future__ import annotations

import logging
import subprocess
import urllib.parse
from pathlib import Path

from why.citations import validate_citations
from why.timeline import validate_and_repair_timeline
from why.diff import get_commit_diff
from why.git import GitError
from why.history import get_file_history, get_line_history
from why.llm import LLMClient
from why.prompts import CommitWithPR, build_system_prompt, build_why_prompt
from why.scoring import select_key_commits
from why.symbols import find_symbol_range
from why.target import Target

_log = logging.getLogger(__name__)

# Lines of context to include above and below a bare line target.
_LINE_WINDOW = 20


def _get_repo_url(repo: Path) -> str | None:
    """Return the origin remote URL for *repo*, or None if unavailable.

    Converts SSH URLs of the form ``git@github.com:owner/repo.git`` to the
    canonical HTTPS form ``https://github.com/owner/repo``.  Trailing ``.git``
    is stripped from both SSH and HTTPS URLs.

    Returns None on any subprocess failure (no remote, not a git repo, etc.).
    """
    try:
        # Fix 1: add timeout to prevent hanging on slow/unresponsive git remotes
        proc = subprocess.run(
            ["git", "-C", str(repo), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None

        url = proc.stdout.strip()

        # Convert SSH shorthand to HTTPS: git@github.com:owner/repo.git
        if url.startswith("git@"):
            # Fix 3: guard against malformed SSH URLs that have no ":" separator
            without_prefix = url[len("git@"):]
            host, sep, path = without_prefix.partition(":")
            if not sep:
                return None  # malformed SSH URL — no path separator
            url = f"https://{host}/{path}"

        # Strip trailing ".git" suffix
        if url.endswith(".git"):
            url = url[:-4]

        # Fix 2: validate URL scheme and reject control characters to prevent prompt injection
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("https", "http"):
            return None
        if any(c in url for c in ("\n", "\r", "\x00")):
            return None

        return url
    except Exception:
        # Swallow all errors (subprocess not found, permissions, TimeoutExpired, etc.)
        return None


def _extract_current_code(target: Target, line_range: tuple[int, int] | None = None) -> str:
    """Return the relevant source text for *target*.

    Priority order:
    1. Symbol — use the pre-resolved line_range (must not be None; callers
       are responsible for resolving via _resolve_line_range first).
    2. Line — return a ±_LINE_WINDOW window around the given line, clamped
       to file bounds.
    3. File-only — return the entire file text.
    """
    if target.symbol is not None:
        # line_range is always resolved before this call for symbol targets;
        # SymbolNotFoundError is raised by _resolve_line_range if the symbol is missing.
        assert line_range is not None, (
            "_extract_current_code: symbol target requires a resolved line_range"
        )
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
    1. Symbol — delegate to find_symbol_range; raises SymbolNotFoundError if missing.
    2. Line — return (line, line).
    3. File-only — return None (no narrowing possible).
    """
    if target.symbol is not None:
        # Let SymbolNotFoundError propagate so the CLI can surface it to the user.
        return find_symbol_range(target.file, target.symbol)

    if target.line is not None:
        return (target.line, target.line)

    return None


def synthesize_why(
    target: Target,
    repo: Path,
    llm: LLMClient,
    prs: dict[str, str] | None = None,
    strict: bool = False,
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
    repo_url = _get_repo_url(repo)
    result = llm.complete(build_system_prompt(repo_url), messages)

    # Post-process: validate every SHA mentioned in the LLM output against the
    # set of SHAs we actually provided in context.  PR numbers aren't available
    # from the prs dict (it maps sha → body), so pass an empty set for now.
    known_shas = {c.sha for c in key_commits}
    issues = validate_citations(result, known_shas, known_prs=set(), strict=strict)
    for issue in issues:
        # Strip non-printable characters from issue.value before logging to
        # prevent control codes or ANSI escapes from polluting log output.
        safe_value = "".join(c for c in issue.value if c.isprintable())
        _log.warning("citation issue %s: %s", issue.kind, safe_value)

    # Post-process: ensure the ## 📊 Timeline section is present and valid.
    # Repairs or appends a deterministic timeline when the LLM omits or
    # hallucinates SHAs in that section.
    result = validate_and_repair_timeline(result, commits_with_prs, repo_url)

    return result
