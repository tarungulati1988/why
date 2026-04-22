"""File commit history retrieval for the why CLI.

This module provides ``get_file_history``: a thin orchestration layer that
builds a ``git log --follow`` invocation and delegates parsing to
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
from .git import run_git


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
