"""Thin, defensive wrapper around git subprocess calls."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GitError(Exception):
    """Base exception for git subprocess failures."""


class NotAGitRepoError(GitError):
    """Raised when a git command is run outside any git repository."""


class GitNotFoundError(GitError):
    """Raised when the git binary is not on PATH."""


class GitTimeoutError(GitError):
    """Raised when a git command exceeds its timeout."""


# Environment overrides applied to every git invocation to suppress interactive
# prompts, pagers, and editors that would hang a subprocess call.
_HARDENING_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "GIT_EDITOR": "true",
    "LC_ALL": "C",
}


def run_git(args: list[str], *, cwd: Path, timeout: float = 30.0) -> str:
    """Run ``git <args>`` in ``cwd`` and return stdout (unstripped).

    Stdout is returned verbatim (trailing newline preserved). Callers that
    want a stripped value must call ``.rstrip()`` themselves — this avoids
    surprising binary-safe consumers later.

    Raises GitNotFoundError, GitTimeoutError, NotAGitRepoError, or GitError.
    """
    # Layer hardening vars on top of the current environment so PATH etc. survive.
    env = {**os.environ, **_HARDENING_ENV}
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise GitNotFoundError("git binary not found on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise GitTimeoutError(
            f"git {' '.join(args)} timed out after {timeout}s"
        ) from e
    except OSError as e:
        # cwd not a directory, permissions, etc. — any launch-time OSError
        # that isn't specifically "git binary missing".
        raise GitError(f"could not launch git in {cwd}: {e}") from e

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Match on English stderr — LC_ALL=C in _HARDENING_ENV guarantees
        # git emits this exact phrase regardless of the user's locale.
        if "not a git repository" in stderr:
            raise NotAGitRepoError(f"{cwd} is not a git repository")
        raise GitError(f"git {' '.join(args)} failed: {stderr}")
    return result.stdout
