"""Per-commit diff retrieval, optionally scoped to a line range.

Stage: diff — called from synthesize_why for each key commit after scoring.

Inputs:
    sha        — commit SHA to inspect.
    file       — path to the file whose diff is wanted.
    line_range — optional (start, end) 1-based inclusive tuple; when set,
                 git log -L is used instead of git show for semantic tracking.
    repo       — repository root used as cwd for git subprocesses.

Outputs:
    str — unified diff text, truncated to max_chars (default 2000) with a
          "[truncated]" sentinel appended when clipped.

Invariants:
    For root commits (no parent for sha^!), git log -L fails with the base
    GitError; the code falls back to a whole-file diff via git show. Only
    the exact base GitError triggers this fallback — GitTimeoutError and
    GitNotFoundError are subclasses and propagate to the caller unchanged.
    Truncation via max_chars is always applied; callers cannot receive a diff
    larger than that value.

Notes:
    git log -L fails on root commits (no parent for sha^!); the error is caught
    and the code falls back to a whole-file diff via git show. Only the base
    GitError triggers the fallback — GitTimeoutError and GitNotFoundError propagate.
"""

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
