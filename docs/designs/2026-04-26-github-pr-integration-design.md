# Design: M2 GitHub PR Integration (#24–#27)

**Date:** 2026-04-26  
**Issues:** #24 (fetch PR for SHA), #25 (PR cache), #26 (prompt injection), #27 (synthesizer wiring)  
**Milestone:** M2 — GitHub PR integration

---

## Problem

`why` mines git history to explain code decisions, but commit messages are often thin ("fix bug", "update logic"). PR bodies carry the richer context: the *why*, the trade-offs considered, the issue link. Today `synthesize_why` accepts a `prs: dict[str, PRMetadata]` parameter but the CLI always passes `None` — the plumbing exists, the data source does not.

---

## Goals

- Implement `GitHubClient.get_prs_for_commit(sha)` using the GitHub REST API
- Cache results SHA-keyed in `~/.cache/why/` (XDG-aware, 30-day TTL)
- Add `title` to `PRMetadata` and inject `PR #N: Title` + body into the prompt
- Wire PR fetching into `synthesize_why` after key-commit selection (fetch ≤5 commits)
- Degrade gracefully: missing token, API error, non-GitHub remote → proceed without PRs

## Non-goals

- Scoring boost from PR membership at selection time (left for M3 — requires prefetching all history commits)
- GitLab / Bitbucket MR support
- GitHub App authentication (token-based only for now)

---

## Solution

### 1. `PRMetadata` schema change (prompts.py)

Add `title` field. All callers that construct `PRMetadata` must supply it.

```python
class PRMetadata(NamedTuple):
    number: int
    title: str
    body: str
```

### 2. New module: `src/why/github.py`

```
GitHubClient
  __init__(repo_url: str, token: str | None)
  get_prs_for_commit(sha: str, cache: PRCache | None) -> list[PRMetadata]
  _fetch_prs(sha: str) -> list[PRMetadata]          # raw API call
  _parse_remote(repo_url: str) -> tuple[str, str]   # owner, repo

def detect_github_token() -> str | None
  # 1. GITHUB_TOKEN env var
  # 2. gh auth token (subprocess, silent on failure)
  # 3. None

def get_github_remote(repo: Path) -> str | None
  # git remote get-url origin → filter for github.com URLs
```

**API endpoint:** `GET /repos/{owner}/{repo}/commits/{sha}/pulls`  
**Transport:** `urllib.request` (stdlib, no new dep)  
**Auth header:** `Authorization: Bearer <token>` when token available  
**Accept header:** `application/vnd.github+json`  
**Rate limits:** Authenticated = 5000 req/hr, unauthenticated = 60 req/hr

**Error handling:**
- `401/403` → log warning "GitHub auth failed, proceeding without PRs", return `[]`
- `404` → return `[]` (commit not in repo, valid state)
- `422` → return `[]` (SHA not a valid commit)
- Network/timeout → log warning, return `[]`
- Non-GitHub remote → skip entirely (no `GitHubClient` constructed)

### 3. New module: `src/why/cache.py`

```
PRCache
  __init__(repo_slug: str)   # repo_slug = "owner__repo"
  path: Path                 # XDG_CACHE_HOME/why/<repo_slug>.json
  get(sha: str) -> list[PRMetadata] | None
  set(sha: str, prs: list[PRMetadata]) -> None
  _is_expired(entry: dict) -> bool   # >30 days since "cached_at"

def xdg_cache_dir() -> Path
  # os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
```

**Cache file format:**
```json
{
  "abc1234...": {
    "cached_at": "2026-04-26T12:00:00Z",
    "prs": [{"number": 42, "title": "Fix the thing", "body": "..."}]
  }
}
```

**`--no-cache` flag:** Added to CLI; when set, `PRCache` is not passed to `get_prs_for_commit`, forcing a fresh fetch.

### 4. Prompt injection update (prompts.py, issue #26)

In `build_why_prompt`, the commit block for a PR-backed commit changes from:

```
**PR Body:**
```text
<body or N/A>
```
```

to:

```
**PR [#42](…/pull/42): Fix the thing**
```text
<body truncated to 1000 chars>
```
```

When no PR: the section is omitted entirely (cleaner than "N/A").

Truncation: `body[:1000] + " …"` if `len(body) > 1000`.

### 5. Synthesizer wiring (synth.py + cli.py, issue #27)

**Fetch timing: after key-commit selection.**

```
git history → select_key_commits() → [up to 5 commits]
                                          ↓
                               GitHubClient.get_prs_for_commit(sha)  ← NEW
                               (for each key commit, with cache)
                                          ↓
                               prs: dict[str, PRMetadata]
                                          ↓
                               build_why_prompt(..., commits_with_prs)
```

**`synthesize_why` change:** The function already accepts `prs` and passes it through. No signature change needed — the CLI layer populates it.

**CLI change (`cli.py`):**
```python
# After target/repo parsing, before synthesize_why call:
github_remote = get_github_remote(repo)
prs: dict[str, PRMetadata] = {}
if github_remote:
    token = detect_github_token()
    client = GitHubClient(github_remote, token)
    cache = None if no_cache else PRCache(repo_slug)
    # key_commits derived after synthesis — need to expose them
    # OR: fetch inside synthesize_why after key commit selection
```

**Decision:** Fetch inside `synthesize_why`, not in CLI. The CLI doesn't have access to key commits (they're selected inside `synthesize_why`). Pass `GitHubClient | None` into `synthesize_why` instead of a pre-built `prs` dict.

Updated signature:
```python
def synthesize_why(
    target: Target,
    repo: Path,
    llm: LLMClient,
    gh: GitHubClient | None = None,      # NEW (replaces prs param)
    pr_cache: PRCache | None = None,     # NEW
    prs: dict[str, PRMetadata] | None = None,  # kept for direct callers / tests
    ...
) -> str:
```

Inside `synthesize_why`, after `select_key_commits`:
```python
if gh is not None:
    for commit in key_commits:
        fetched = gh.get_prs_for_commit(commit.sha, pr_cache)
        if fetched:
            resolved_prs[commit.sha] = fetched[0]  # take first PR (squash-merge norm)
```

`prs` parameter takes precedence if supplied (backward-compatible for tests).

**Warning output:** When `gh` is provided but token is missing or API fails:
```
⚠  GitHub PR data unavailable — proceeding without PR context
```

---

## File Changes

| File | Change |
|------|--------|
| `src/why/github.py` | **New** — `GitHubClient`, `detect_github_token`, `get_github_remote` |
| `src/why/cache.py` | **New** — `PRCache`, `xdg_cache_dir` |
| `src/why/prompts.py` | Add `title` to `PRMetadata`; update commit block rendering |
| `src/why/synth.py` | Add `gh` + `pr_cache` params; fetch PRs after key-commit selection |
| `src/why/cli.py` | Detect GitHub remote; init `GitHubClient`; pass `--no-cache` flag |
| `tests/test_github.py` | **New** — mocked API: 0/1/2 PRs, 401, 404, timeout |
| `tests/test_cache.py` | **New** — cache hit/miss/expired |
| `tests/test_prompts.py` | Update golden prompts for title field, body truncation |
| `tests/test_synth.py` | Update for new `gh` param; mock `GitHubClient` |

---

## Test Plan

- `test_github.py`: Mock `urllib.request.urlopen`; assert correct URL construction, header injection, response parsing for 0/1/2 PRs; assert `[]` on 401/404/timeout
- `test_cache.py`: Cache miss → calls API; cache hit → skips API; expired entry → re-fetches; `--no-cache` → always fetches
- `test_prompts.py`: Golden prompt with PR present (title + truncated body); golden prompt without PR (section omitted)
- `test_synth.py`: `gh=MockClient` → `prs` dict populated after selection; `gh=None` → no PR fetch; API error → `prs` empty, synthesis continues

---

## Rollout

All four issues ship together in one PR on branch `sleipnir/issue-24-27-github-pr-integration`. No feature flag needed — graceful degradation handles the unauthenticated case.
