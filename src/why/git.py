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


# Environment overrides applied to every git invocation. Each one closes a
# specific way git can block a headless subprocess or drift under the user's
# locale. See `git help environment` for the canonical reference.
_HARDENING_ENV = {
    # Git feature: if set to 0, git refuses to prompt on the controlling
    # terminal for usernames/passwords (normally triggered by HTTPS remotes
    # without a credential helper). Instead of hanging forever waiting for
    # stdin, git exits with a "terminal prompts disabled" error we can surface.
    "GIT_TERMINAL_PROMPT": "0",
    # Git feature: overrides `core.pager` for this invocation. Git's default
    # pager is `less`, which can attach to a TTY and wait for the user to
    # press `q`. Forcing `cat` makes pagination a no-op pass-through.
    "GIT_PAGER": "cat",
    # Git feature: overrides `core.editor` for commands that would otherwise
    # open `$EDITOR` (commit, rebase -i, tag -a). `true` is a shell builtin
    # that exits 0 with no output, so any unexpected editor invocation
    # returns immediately instead of blocking on vim/nano.
    "GIT_EDITOR": "true",
    # POSIX locale override (not git-specific): forces all libc-localised
    # output — including git's own stderr messages — to the "C" locale
    # (ASCII English). Required because we substring-match on English
    # phrases like "not a git repository" to classify errors; a German or
    # Japanese user's default locale would otherwise silently break that
    # classification.
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
