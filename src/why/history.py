"""File and line-level commit history retrieval.

Stage: git/history — sits between the target (what the user pointed at) and
       commit parsing; delegates to parse_porcelain for all output parsing.

Inputs:
    file — absolute path to the file of interest (resolved by target.py).
    repo — repository root used as cwd for git subprocesses.

Outputs:
    list[Commit] — newest-first, matching git log default ordering.
                   get_line_history returns [] when the requested line is out
                   of bounds.  find_introduction returns None when the file
                   has no git history.

Invariants:
    - Paths reaching this layer are assumed to be absolute and already
      confined to the repo root (enforced by target.py upstream). The one
      exception is get_line_history, which adds an explicit path-confinement
      guard because -L requires a repo-relative path and the resolution must
      happen here.
    - get_file_history uses --follow to traverse renames.
    - get_line_history uses git log -L for semantic line tracking (no --follow;
      -L handles continuity through edits but not renames).
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


def find_introduction(file: Path, repo: Path) -> Commit | None:
    """Return the oldest commit that first introduced ``file`` into the repo.

    Uses ``--diff-filter=A --follow`` to find the commit that added the file,
    traversing renames so the original introduction is found even if the file
    was later renamed.

    Args:
        file: Absolute path to the file whose introduction commit is requested.
        repo: Root of the git repository (used as the cwd for git).

    Returns:
        The oldest :class:`Commit` that added ``file``, or ``None`` if the
        file has no git history (e.g., untracked or never committed).
    """
    # NOTE: enhancement opportunity — path-confinement guard
    # If callers ever bypass Target resolution and pass untrusted paths directly,
    # add a guard here similar to get_line_history:
    #   resolved = file.resolve()
    #   repo_resolved = repo.resolve()
    #   if not resolved.is_relative_to(repo_resolved):
    #       raise ValueError(f"file {file} escapes repository root {repo}")
    # At that point also pass str(resolved) instead of str(file).
    args = [
        "log",
        "--diff-filter=A",
        "--follow",
        f"--format={PORCELAIN_FORMAT}",
        "--",
        str(file),
    ]
    output = run_git(args, cwd=repo)
    if not output.strip():
        return None
    commits = parse_porcelain(output)
    # parse_porcelain returns newest-first. --diff-filter=A --follow normally
    # yields exactly 1 commit, but a file deleted and re-added has multiple add
    # events. commits[-1] always returns the oldest (the true introduction).
    return commits[-1] if commits else None
