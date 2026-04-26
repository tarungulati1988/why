"""Tests for src/why/github.py — GitHubClient and detect_github_token.

Written BEFORE the implementation (TDD red phase).
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from why.github import GitHubAuthError, GitHubClient, detect_github_token
from why.prompts import PRMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_urlopen_mock(payload: list[dict]) -> MagicMock:
    """Return a mock that urlopen returns a file-like object with JSON payload."""
    response_body = json.dumps(payload).encode()
    mock_response = MagicMock()
    mock_response.read.return_value = response_body
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = MagicMock(return_value=False)
    mock_urlopen = MagicMock(return_value=mock_response)
    return mock_urlopen


# ---------------------------------------------------------------------------
# get_prs_for_commit — happy path
# ---------------------------------------------------------------------------

class TestGetPrsForCommit:
    def _client(self) -> GitHubClient:
        return GitHubClient("https://github.com/owner/repo")

    def test_get_prs_for_commit_empty(self):
        mock_urlopen = _make_urlopen_mock([])
        with patch("urllib.request.urlopen", mock_urlopen):
            result = self._client().get_prs_for_commit("abc123")
        assert result == []

    def test_get_prs_for_commit_single_pr(self):
        payload = [{"number": 42, "title": "Fix bug", "body": "body text"}]
        mock_urlopen = _make_urlopen_mock(payload)
        with patch("urllib.request.urlopen", mock_urlopen):
            result = self._client().get_prs_for_commit("abc123")
        assert result == [PRMetadata(42, "Fix bug", "body text")]

    def test_get_prs_for_commit_multiple_prs(self):
        payload = [
            {"number": 1, "title": "First", "body": "body one"},
            {"number": 2, "title": "Second", "body": "body two"},
        ]
        mock_urlopen = _make_urlopen_mock(payload)
        with patch("urllib.request.urlopen", mock_urlopen):
            result = self._client().get_prs_for_commit("abc123")
        assert result == [
            PRMetadata(1, "First", "body one"),
            PRMetadata(2, "Second", "body two"),
        ]

    def test_get_prs_for_commit_null_body(self):
        payload = [{"number": 7, "title": "No body PR", "body": None}]
        mock_urlopen = _make_urlopen_mock(payload)
        with patch("urllib.request.urlopen", mock_urlopen):
            result = self._client().get_prs_for_commit("abc123")
        assert result == [PRMetadata(7, "No body PR", "")]


# ---------------------------------------------------------------------------
# get_prs_for_commit — error handling
# ---------------------------------------------------------------------------

class TestGetPrsForCommitErrors:
    def _client(self) -> GitHubClient:
        return GitHubClient("https://github.com/owner/repo")

    def test_get_prs_for_commit_404_returns_empty(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                url="https://api.github.com/...", code=404,
                msg="Not Found", hdrs=None, fp=None
            )
            result = self._client().get_prs_for_commit("abc123")
        assert result == []

    def test_get_prs_for_commit_401_raises_auth_error(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                url="https://api.github.com/...", code=401,
                msg="Unauthorized", hdrs=None, fp=None
            )
            with pytest.raises(GitHubAuthError):
                self._client().get_prs_for_commit("abc123")

    def test_get_prs_for_commit_403_raises_auth_error(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = HTTPError(
                url="https://api.github.com/...", code=403,
                msg="Forbidden", hdrs=None, fp=None
            )
            with pytest.raises(GitHubAuthError):
                self._client().get_prs_for_commit("abc123")

    def test_get_prs_for_commit_network_error_returns_empty(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("Connection refused")
            result = self._client().get_prs_for_commit("abc123")
        assert result == []

    def test_get_prs_for_commit_timeout_returns_empty(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("timed out")
            result = self._client().get_prs_for_commit("abc123")
        assert result == []


# ---------------------------------------------------------------------------
# detect_github_token
# ---------------------------------------------------------------------------

class TestDetectGithubToken:
    def test_detect_github_token_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "abc123")
        assert detect_github_token() == "abc123"

    def test_detect_github_token_none_when_missing(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            result = detect_github_token()
        assert result is None


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------

class TestGithubClientInit:
    def test_github_client_parses_owner_repo(self):
        client = GitHubClient("https://github.com/owner/repo")
        assert client._owner == "owner"
        assert client._repo == "repo"

    def test_github_client_strips_dot_git(self):
        client = GitHubClient("https://github.com/owner/repo.git")
        assert client._repo == "repo"

    def test_github_client_stores_token(self):
        client = GitHubClient("https://github.com/owner/repo", token="mytoken")
        assert client._token == "mytoken"

    def test_github_client_no_token_is_none(self):
        client = GitHubClient("https://github.com/owner/repo")
        assert client._token is None
