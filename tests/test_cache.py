"""Tests for src/why/cache.py — SHA-keyed PR metadata cache.

Written BEFORE the implementation (TDD red phase).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from why.cache import PRCache, xdg_cache_dir
from why.prompts import PRMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prs() -> list[PRMetadata]:
    return [PRMetadata(number=42, title="Fix bug", body="body text")]


def _patch_xdg(monkeypatch, tmp_path: Path) -> None:
    """Override xdg_cache_dir so the cache writes under tmp_path."""
    monkeypatch.setattr("why.cache.xdg_cache_dir", lambda: tmp_path)


# ---------------------------------------------------------------------------
# xdg_cache_dir
# ---------------------------------------------------------------------------

class TestXdgCacheDir:
    def test_xdg_cache_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/test")
        assert xdg_cache_dir() == Path("/tmp/test")

    def test_xdg_cache_dir_default(self, monkeypatch):
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert xdg_cache_dir() == Path.home() / ".cache"


# ---------------------------------------------------------------------------
# PRCache.path
# ---------------------------------------------------------------------------

class TestPRCachePath:
    def test_cache_path_uses_repo_slug(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        assert cache.path == tmp_path / "why" / "owner__repo.json"


# ---------------------------------------------------------------------------
# PRCache.get — misses
# ---------------------------------------------------------------------------

class TestPRCacheMiss:
    def test_cache_miss_returns_none(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        assert cache.get("deadbeef") is None

    def test_cache_missing_file_returns_none(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        # Explicitly ensure the file does not exist.
        assert not cache.path.exists()
        assert cache.get("deadbeef") is None

    def test_cache_corrupt_file_returns_none(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        cache.path.write_text("not valid json{{{{")
        assert cache.get("deadbeef") is None


# ---------------------------------------------------------------------------
# PRCache.set / get — round-trips
# ---------------------------------------------------------------------------

class TestPRCacheRoundTrip:
    def test_cache_set_then_get(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        prs = _make_prs()
        cache.set("abc123", prs)
        result = cache.get("abc123")
        assert result == prs

    def test_cache_stores_title(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        prs = [PRMetadata(number=7, title="My feature PR", body="desc")]
        cache.set("sha1", prs)
        result = cache.get("sha1")
        assert result is not None
        assert result[0].title == "My feature PR"

    def test_cache_empty_pr_list(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        cache.set("sha_empty", [])
        result = cache.get("sha_empty")
        # Must return [] (not None) — cached empty list is a valid hit
        assert result == []

    def test_cache_creates_parent_dirs(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        # Parent dirs do not exist yet
        assert not cache.path.parent.exists()
        cache.set("sha_dirs", _make_prs())
        assert cache.path.parent.exists()
        assert cache.path.exists()


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

class TestPRCacheTTL:
    def _write_raw_entry(self, cache: PRCache, sha: str, cached_at: datetime) -> None:
        """Write a raw cache entry with a custom timestamp for TTL testing."""
        data = cache._load()
        data[sha] = {
            "cached_at": cached_at.isoformat(),
            "prs": [{"number": 1, "title": "t", "body": "b"}],
        }
        cache._save(data)

    def test_cache_hit_not_expired(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        one_day_ago = datetime.now(UTC) - timedelta(days=1)
        self._write_raw_entry(cache, "fresh_sha", one_day_ago)
        result = cache.get("fresh_sha")
        assert result is not None
        assert result[0].number == 1

    def test_cache_expired_returns_none(self, monkeypatch, tmp_path):
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        thirty_one_days_ago = datetime.now(UTC) - timedelta(days=31)
        self._write_raw_entry(cache, "stale_sha", thirty_one_days_ago)
        result = cache.get("stale_sha")
        assert result is None

    def test_cache_naive_datetime_returns_none(self, monkeypatch, tmp_path):
        """A cache entry with a naive (no tzinfo) datetime string must return None, not crash."""
        _patch_xdg(monkeypatch, tmp_path)
        cache = PRCache("owner__repo")
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        # Write a naive ISO datetime string (no +00:00 suffix) directly into the cache file.
        data = {
            "naive_sha": {
                "cached_at": "2025-01-01T00:00:00",  # naive — no timezone info
                "prs": [{"number": 1, "title": "t", "body": "b"}],
            }
        }
        cache._save(data)
        # Must return None (treat as expired/invalid) without raising TypeError
        assert cache.get("naive_sha") is None
