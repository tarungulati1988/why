"""GitHub REST API client for PR metadata lookup.

Stage: sidecar — called from synthesize_why once per key commit when a GitHub
       remote is detected and no cached entry exists.

Inputs:
    repo_url — canonical https://github.com/owner/repo URL.
    sha      — full commit SHA to look up associated PRs for.
    token    — optional GitHub personal access token (GITHUB_TOKEN env var or
               gh auth token); omitted for unauthenticated requests.

Outputs:
    list[PRMetadata] — PRs merged via this commit; empty list on any API failure
                       (404, timeout, network error). GitHubAuthError is re-raised
                       on 401/403 so the caller can surface auth issues to the user.

Invariants:
    Only 401/403 responses re-raise (as GitHubAuthError); all other failures
    (network errors, 404, timeouts, unexpected status codes) return [].
    The token is set once at construction via the token parameter and stored
    as self._token; it does not change between calls.

Notes:
    Token discovery order: GITHUB_TOKEN env var → `gh auth token` subprocess → None.
    Unauthenticated requests are subject to GitHub's 60 req/hr rate limit per IP.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request

from why.prompts import PRMetadata

_API_BASE = "https://api.github.com"
_TIMEOUT = 10


class GitHubError(Exception):
    """Base exception for GitHub API failures."""


class GitHubAuthError(GitHubError):
    """Raised when GitHub returns 401 or 403."""


class GitHubClient:
    def __init__(self, repo_url: str, token: str | None = None) -> None:
        """
        repo_url: normalized https://github.com/owner/repo URL
        token: GitHub personal access token or None for unauthenticated
        """
        parsed = urllib.parse.urlparse(repo_url)
        # path is like /owner/repo or /owner/repo.git
        parts = parsed.path.strip("/").split("/")
        self._owner = parts[0]
        self._repo = parts[1].removesuffix(".git") if len(parts) > 1 else parts[0]
        self._token = token

    def get_prs_for_commit(self, sha: str) -> list[PRMetadata]:
        """
        Return PRs associated with sha. Returns [] on any API failure (404, timeout, etc).
        Raises GitHubAuthError on 401/403.
        """
        try:
            return self._fetch_prs(sha)
        except GitHubAuthError:
            raise
        except Exception:
            return []

    def _fetch_prs(self, sha: str) -> list[PRMetadata]:
        """Raw API call to /repos/{owner}/{repo}/commits/{sha}/pulls"""
        url = f"{_API_BASE}/repos/{self._owner}/{self._repo}/commits/{sha}/pulls"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")

        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise GitHubAuthError(f"GitHub auth failed: {exc.code}") from exc
            raise

        return [
            PRMetadata(
                number=pr["number"],
                title=pr["title"],
                body=pr["body"] or "",
            )
            for pr in data
        ]


def detect_github_token() -> str | None:
    """
    Discover GitHub token:
    1. GITHUB_TOKEN env var
    2. subprocess: gh auth token (silent on failure)
    3. None (unauthenticated)
    """
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    try:
        result = subprocess.run(
            ["gh", "auth", "token"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            token = result.stdout.strip()
            if token and "\n" not in token and "\r" not in token and " " not in token:
                return token
    except Exception:
        pass

    return None
