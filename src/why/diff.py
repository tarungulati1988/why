"""Utilities for retrieving a commit's diff from a git repository."""

from __future__ import annotations

from pathlib import Path

from why.git import GitError, run_git


def get_commit_diff(
    sha: str,
    file: Path,
    line_range: tuple[int, int] | None,
    repo: Path,
    max_chars: int = 2000,
) -> str:
    """Return the diff for *sha* scoped to *file*, optionally narrowed to *line_range*.

    Parameters
    ----------
    sha:
        The commit SHA to inspect.
    file:
        Path to the file (relative to *repo*) whose diff is wanted.
    line_range:
        Optional ``(start, end)`` tuple (1-based, inclusive). When supplied the
        function uses ``git log -L`` to track the exact line range; otherwise the
        full file diff from ``git show`` is returned.
    repo:
        Absolute path to the git repository root used as cwd for every subprocess.
    max_chars:
        Maximum number of characters to return. Output longer than this is
        truncated and a ``"\\n... [truncated]"`` suffix is appended.

    Returns
    -------
    str
        The (possibly truncated) diff text.
    """
    if line_range is not None:
        start, end = line_range
        # Use git log -L to track the line range semantically within a single
        # commit. `sha^!` is shorthand for `sha^..sha` (scopes to one commit).
        log_args = [
            "log",
            "-L",
            f"{start},{end}:{file}",  # range spec: <start>,<end>:<path>
            f"{sha}^!",
        ]
        try:
            output = run_git(log_args, cwd=repo)
        except GitError as exc:
            # Only the base GitError (non-zero exit) signals a root-commit
            # failure (no parent for sha^!). Subclasses like GitTimeoutError or
            # GitNotFoundError indicate a different problem and must propagate.
            if type(exc) is not GitError:
                raise
            output = _whole_file_diff(sha, file, repo)
    else:
        output = _whole_file_diff(sha, file, repo)

    # Truncate long output so callers receive a bounded string.
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... [truncated]"

    return output


def _whole_file_diff(sha: str, file: Path, repo: Path) -> str:
    """Run ``git show --format= <sha> -- <file>`` and return raw unified diff."""
    # --format= suppresses the commit header so only the patch is emitted.
    args = ["show", "--format=", sha, "--", str(file)]
    return run_git(args, cwd=repo)
