"""SHA-keyed local cache for GitHub PR metadata.

Stage: sidecar — consulted and updated inside synthesize_why before each GitHub API call.

Inputs:
    repo_slug — "owner__repo" string derived from the remote URL (double-underscore,
                path-safe for filenames), passed to PRCache.__init__.
    sha       — full commit SHA used as the cache key.

Outputs:
    list[PRMetadata] — cached PR records returned to synthesize_why; avoids a GitHub
                       API round-trip for commits already seen within the TTL window.

Invariants:
    - Cache entries expire after 30 days (TTL enforced on read, not on write).
    - Writes are atomic: JSON is written to a temp file then renamed into place.
    - Corrupt or missing cache files are silently treated as a cache miss.

Notes:
    Cache lives at XDG_CACHE_HOME/why/<repo_slug>.json (defaults to ~/.cache/why/).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from why.prompts import PRMetadata

_TTL = timedelta(days=30)


def xdg_cache_dir() -> Path:
    """Return XDG_CACHE_HOME if set, else ~/.cache"""
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg)
    return Path.home() / ".cache"


class PRCache:
    def __init__(self, repo_slug: str) -> None:
        """
        repo_slug: "owner__repo" (double underscore, safe for filenames)
        Cache file: xdg_cache_dir() / "why" / f"{repo_slug}.json"
        """
        self.path: Path = xdg_cache_dir() / "why" / f"{repo_slug}.json"

    def get(self, sha: str) -> list[PRMetadata] | None:
        """Return cached PRs for sha, or None if not cached / expired / corrupt."""
        data = self._load()
        entry = data.get(sha)
        if entry is None:
            return None
        if self._is_expired(entry):
            return None
        try:
            return [
                PRMetadata(number=pr["number"], title=pr["title"], body=pr["body"])
                for pr in entry["prs"]
            ]
        except (KeyError, TypeError):
            return None

    def set(self, sha: str, prs: list[PRMetadata]) -> None:
        """Write prs to cache under sha key. Creates parent directories if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = self._load()
        data[sha] = {
            "cached_at": datetime.now(UTC).isoformat(),
            "prs": [
                {"number": pr.number, "title": pr.title, "body": pr.body}
                for pr in prs
            ],
        }
        self._save(data)

    def _load(self) -> dict[str, Any]:
        """Load cache JSON from disk. Returns {} on missing or corrupt file."""
        try:
            result: dict[str, Any] = json.loads(self.path.read_text())
            return result
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        """Write cache JSON to disk atomically (write tmp → rename)."""
        dir_ = self.path.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Return True if entry's cached_at is >30 days ago."""
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            return datetime.now(UTC) - cached_at > _TTL
        except (KeyError, ValueError, TypeError):
            return True
