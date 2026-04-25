"""Shared test utilities for the why test suite."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --update-goldens flag for regenerating golden files."""
    parser.addoption("--update-goldens", action="store_true", default=False)


@pytest.fixture
def update_goldens(request: pytest.FixtureRequest) -> bool:
    """Return True when --update-goldens was passed on the command line."""
    return bool(request.config.getoption("--update-goldens"))


def _tmp_path_is_clean(tmp_path: Path) -> bool:
    """Return True if tmp_path is NOT inside an existing git repo.

    Used by integration test fixtures to skip when pytest's tmp_path
    happens to land inside a real git working tree (e.g. the project repo
    itself), which would invalidate isolation assumptions.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    return result.returncode != 0


def make_git_runner(repo: Path) -> Callable[..., None]:
    """Return a git() callable pre-configured for repo with test env vars.

    The returned callable accepts any git subcommand args and an optional
    `date` keyword argument.  When `date` is supplied it is set as both
    GIT_AUTHOR_DATE and GIT_COMMITTER_DATE so commits get deterministic
    timestamps regardless of when the test runs.
    """
    base_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "GIT_TERMINAL_PROMPT": "0",
        # Match run_git's _HARDENING_ENV so fixture output is locale-stable
        # and never blocked by a pager.
        "LC_ALL": "C",
        "GIT_PAGER": "cat",
    }

    def git(*args: str, date: str | None = None) -> None:
        env = {**base_env}
        if date is not None:
            # Pin both author and committer timestamps for deterministic history.
            env["GIT_AUTHOR_DATE"] = date
            env["GIT_COMMITTER_DATE"] = date
        try:
            subprocess.run(
                ["git", *args],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            # Re-raise with stderr so fixture failures are diagnosable.
            raise RuntimeError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc

    return git
