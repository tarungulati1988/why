"""File commit history retrieval for the why CLI.

This module provides ``get_file_history`` and ``get_line_history``: thin
orchestration layers that build git invocations and delegate parsing to
``parse_porcelain``.

Design note — why relative paths are not normalised here:
  ``target.py`` always resolves paths to absolute before they reach this
  layer, so adding ``file.relative_to(repo)`` normalization here would be
  solving a problem already handled upstream.  If a programmatic API ever
  bypasses ``Target`` resolution and passes a relative path directly, that
  is the right time to add normalization in this function — not before.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .commit import PORCELAIN_FORMAT, SHORTSTAT_FLAG, Commit, parse_porcelain
from .git import GitError, run_git


def get_file_history(
    file: Path,
    repo: Path,
    since: datetime | None = None,
) -> list[Commit]:
    """Return the git commit history for ``file`` inside ``repo``.

    Uses ``--follow`` so renames are traversed.  Results are newest-first,
    matching the default ``git log`` ordering.

    Args:
        file: Absolute path to the file whose history is requested.
        repo: Root of the git repository (used as the cwd for git).
        since: When provided, only commits after this datetime are returned.
    """
    args = [
        "log",
        "--follow",
        SHORTSTAT_FLAG,
        f"--format={PORCELAIN_FORMAT}",
    ]

    # --since is optional; omitting it returns the full history
    if since is not None:
        args.append(f"--since={since.isoformat()}")

    # '--' tells git that what follows is a pathspec, not a branch/revision —
    # required when the filename might look like a ref.
    args.extend(["--", str(file)])

    output = run_git(args, cwd=repo)
    return parse_porcelain(output)


def get_line_history(file: Path, repo: Path, *, line: int) -> list[Commit]:
    """Return the git commit history for a specific line of ``file``.

    Uses ``git log -L<line>,<line>:<rel_file>`` to walk commits that touched
    the given line.  The ``-L`` flag requires a repo-relative path, so
    ``file.relative_to(repo)`` is computed here.

    No shortstat is requested (``--no-patch``) — additions/deletions remain
    0 on the returned commits.  ``--follow`` is not used because ``-L``
    handles continuity through edits, not renames.

    Args:
        file: Absolute path to the file whose line history is requested.
        repo: Root of the git repository (used as the cwd for git).
        line: 1-based line number to trace (keyword-only).

    Returns:
        List of :class:`Commit` objects newest-first, or ``[]`` if ``line``
        is out of bounds for the file.
    """
    # Reject non-positive line numbers before touching the filesystem.
    if line < 1:
        raise ValueError(f"line must be >= 1, got {line}")

    # Resolve symlinks and ensure the file is inside the repo root,
    # preventing path-traversal attacks (e.g. file=../../etc/passwd).
    resolved = file.resolve()
    repo_resolved = repo.resolve()
    if not resolved.is_relative_to(repo_resolved):
        raise ValueError(f"file {file} escapes repository root {repo}")
    # -L requires the path relative to the repo root, not an absolute path.
    rel = resolved.relative_to(repo_resolved)
    args = [
        "log",
        f"-L{line},{line}:{rel}",
        f"--format={PORCELAIN_FORMAT}",
        "--no-patch",  # suppresses -L diff output; requires git >= 2.30
    ]
    try:
        output = run_git(args, cwd=repo)
    except GitError as e:
        # git emits "has only N lines" when the requested line exceeds the
        # file length — treat this as "no history at that line".
        # The English phrase is guaranteed because run_git sets LC_ALL=C.
        if "has only" in str(e):
            return []
        raise
    return parse_porcelain(output)
