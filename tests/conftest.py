"""Shared test utilities for the why test suite."""

from __future__ import annotations

import subprocess
from pathlib import Path


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
