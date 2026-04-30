"""Synthesis pipeline orchestration for the why CLI."""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.parse
from dataclasses import replace
from pathlib import Path

import click

from why.cache import PRCache
from why.citations import validate_citations
from why.diff import get_commit_diff
from why.git import GitError
from why.github import GitHubAuthError, GitHubClient
from why.history import get_file_history, get_line_history
from why.llm import LLMClient, Message
from why.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    CommitWithPR,
    PRMetadata,
    build_grounding_prompt,
    build_system_prompt,
    build_why_prompt,
)
from why.scoring import select_key_commits
from why.symbols import find_symbol_range
from why.target import Target
from why.timeline import validate_and_repair_timeline

_log = logging.getLogger(__name__)

# Lines of context to include above and below a bare line target.
_LINE_WINDOW = 20

_COST_PER_1K_TOKENS = 0.0008  # llama-3.3-70b Groq approximate rate
_DEEP_COST_WARN_THRESHOLD = 0.50

_MAX_DIFF_LINES = 80
_CTX_HEADROOM = 0.85
_DEFAULT_CTX_OPENAI_COMPAT = 4096


def _estimate_tokens(text: str) -> int:
    """Cheap token estimate: chars/4. Good enough — the budget reserves 15% headroom."""
    return len(text) // 4


def _shrink_for_budget(
    commits: list[CommitWithPR],
    current_code: str,
    system_prompt: str,
    target_tokens: int,
) -> tuple[list[CommitWithPR], int, int]:
    """Shrink commits to fit `target_tokens` budget for context-constrained models.

    Strategy:
      1. Truncate any diff longer than 80 lines to first 80 + sentinel.
      2. Drop oldest commits (commits[0] first) until total estimated
         tokens fit `target_tokens * 0.85` (15% headroom for the model's reply).

    Returns: (shrunk_commits, dropped_count, truncated_count).

    `commits` is expected to be sorted oldest-first (synthesize_why sorts this way).
    """
    truncated = 0
    new_commits: list[CommitWithPR] = []
    for c in commits:
        lines = c.diff.splitlines()
        if len(lines) > _MAX_DIFF_LINES:
            dropped_lines = len(lines) - _MAX_DIFF_LINES
            new_diff = (
                "\n".join(lines[:_MAX_DIFF_LINES])
                + f"\n... [truncated {dropped_lines} lines]"
            )
            c = replace(c, diff=new_diff)
            truncated += 1
        new_commits.append(c)

    headroom = int(target_tokens * _CTX_HEADROOM)
    fixed = _estimate_tokens(system_prompt) + _estimate_tokens(current_code)
    budget = headroom - fixed

    def _commit_cost(c: CommitWithPR) -> int:
        return (
            _estimate_tokens(c.diff)
            + _estimate_tokens(c.commit.subject)
            + _estimate_tokens(c.pr_body or "")
            + _estimate_tokens(c.pr_title or "")
        )

    dropped = 0
    total = sum(_commit_cost(c) for c in new_commits)
    while new_commits and total > budget:
        total -= _commit_cost(new_commits[0])
        new_commits.pop(0)
        dropped += 1

    return new_commits, dropped, truncated


def _resolve_max_ctx(provider: str) -> int | None:
    """Resolve effective WHY_LLM_MAX_CTX target.

    Resolution rules:
      - WHY_LLM_MAX_CTX set to a positive integer → return that int.
      - WHY_LLM_MAX_CTX set to "0" → return None (explicit disable).
      - WHY_LLM_MAX_CTX unset:
          - provider == "openai-compatible" → return _DEFAULT_CTX_OPENAI_COMPAT (default).
          - Otherwise → return None.
      - WHY_LLM_MAX_CTX set to a negative integer or non-integer string → return None
        and log a single warning via the existing `_log` logger; do NOT raise.
    """
    raw = os.environ.get("WHY_LLM_MAX_CTX")

    if raw is None:
        if provider == "openai-compatible":
            return _DEFAULT_CTX_OPENAI_COMPAT
        return None

    try:
        value = int(raw)
    except ValueError:
        _log.warning("WHY_LLM_MAX_CTX=%r is not a valid integer; ignoring", raw)
        return None

    if value == 0:
        return None  # explicit disable

    if value < 0:
        _log.warning("WHY_LLM_MAX_CTX=%d is negative; ignoring", value)
        return None

    return value


def _estimate_prompt_cost(system: str, messages: list[Message]) -> float:
    total_chars = len(system) + sum(len(m.content) for m in messages)
    tokens = total_chars / 4
    return (tokens / 1000) * _COST_PER_1K_TOKENS


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
    prs: dict[str, PRMetadata] | None = None,
    gh: GitHubClient | None = None,
    pr_cache: PRCache | None = None,
    strict: bool = False,
    two_pass: bool = False,
    brief: bool = False,
    deep: bool = False,
    max_commits: int | None = None,
) -> str:
    """Orchestrate the full why pipeline and return the LLM's explanation.

    Steps:
      1. Normalise prs (dict[str, PRMetadata]) to an empty dict when callers pass None.
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

    if deep:
        capped = history[:max_commits] if max_commits is not None else history
        key_commits = capped
    else:
        key_commits = history if len(history) < 3 else select_key_commits(history, prs)
        if max_commits is not None:
            key_commits = key_commits[:max_commits]

    # Sort oldest-first so the LLM sees chronological order regardless of path.
    key_commits = sorted(key_commits, key=lambda c: c.date)

    # Fetch PR metadata via GitHub client when available.
    # Explicit prs dict takes precedence (backward-compatible for tests and direct callers).
    resolved_prs: dict[str, PRMetadata] = dict(prs) if prs else {}
    if gh is not None and not resolved_prs:
        for c in key_commits:
            # Cache-first: check cache before hitting API
            cached = pr_cache.get(c.sha) if pr_cache is not None else None
            if cached is not None:
                if cached:
                    resolved_prs[c.sha] = cached[0]
                continue
            # Cache miss — fetch from GitHub
            try:
                fetched = gh.get_prs_for_commit(c.sha)
            except GitHubAuthError:
                click.echo(
                    "⚠  GitHub PR data unavailable — proceeding without PR context",
                    err=True,
                )
                break
            if pr_cache is not None:
                try:
                    pr_cache.set(c.sha, fetched)
                except OSError as exc:
                    _log.warning("could not write PR cache for %s: %s", c.sha[:12], exc)
            if fetched:
                resolved_prs[c.sha] = fetched[0]

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

    commits_with_prs = []
    for c in key_commits:
        meta = resolved_prs.get(c.sha)
        commits_with_prs.append(
            CommitWithPR(
                commit=c,
                pr_body=meta.body if meta else None,
                pr_number=meta.number if meta else None,
                pr_title=meta.title if meta else None,
                diff=diffs[c.sha],
            )
        )

    repo_url = _get_repo_url(repo)
    system_prompt = build_system_prompt(repo_url)

    target_ctx = _resolve_max_ctx(llm.provider)
    if target_ctx is not None:
        user_set = os.getenv("WHY_LLM_MAX_CTX") is not None
        commits_with_prs, dropped, truncated = _shrink_for_budget(
            commits_with_prs, current_code, system_prompt, target_ctx
        )
        if dropped or truncated:
            msg = (
                f"⚠ Auto-shrunk to fit context budget: dropped {dropped} commit(s), "
                f"truncated {truncated} diff(s)."
            )
            if not user_set:
                msg += " Set WHY_LLM_MAX_CTX=0 to disable."
            click.echo(msg, err=True)

    messages = build_why_prompt(target, current_code, commits_with_prs, brief=brief)

    if deep:
        estimated_cost = _estimate_prompt_cost(system_prompt, messages)
        if estimated_cost > _DEEP_COST_WARN_THRESHOLD:
            click.echo(
                f"Warning: --deep estimated cost ${estimated_cost:.2f} exceeds "
                f"${_DEEP_COST_WARN_THRESHOLD:.2f} threshold "
                f"(rate assumes {llm.model}; your actual cost may differ).",
                err=True,
            )

    result = llm.complete(system_prompt, messages)

    # Post-process: validate every SHA mentioned in the LLM output against the
    # set of SHAs we actually provided in context.  PR numbers come from PRMetadata,
    # scoped to commits_with_prs so we only accept citations the LLM was shown.
    known_shas = {c.sha for c in key_commits}
    known_prs = {cwpr.pr_number for cwpr in commits_with_prs if cwpr.pr_number is not None}
    issues = validate_citations(result, known_shas, known_prs=known_prs, strict=strict)
    for issue in issues:
        # Strip non-printable characters from issue.value before logging to
        # prevent control codes or ANSI escapes from polluting log output.
        safe_value = "".join(c for c in issue.value if c.isprintable())
        _log.warning("citation issue %s: %s", issue.kind, safe_value)

    # Post-process: ensure the ## 📊 Timeline section is present and valid.
    # Repairs or appends a deterministic timeline when the LLM omits or
    # hallucinates SHAs in that section.
    result = validate_and_repair_timeline(result, commits_with_prs, repo_url)

    if two_pass:
        grounding_messages = build_grounding_prompt(result, commits_with_prs)
        grounding_section = llm.complete(GROUNDING_SYSTEM_PROMPT, grounding_messages)
        if "## 🔍 Grounding Check" in grounding_section:
            result = result + "\n\n" + grounding_section
        else:
            _log.warning("grounding pass returned unexpected content; skipping grounding section")

    return result
