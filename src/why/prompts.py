"""Prompt-building layer for the why LLM synthesis call."""

from __future__ import annotations

from dataclasses import dataclass

from why.commit import Commit
from why.llm import Message
from why.target import Target

# Inject the sparse-history notice when key_commits is strictly below this count
# (0-2 triggers the notice; 3+ suppresses it).
SPARSE_COMMIT_THRESHOLD: int = 3

# ---------------------------------------------------------------------------
# System prompt — built via build_system_prompt() for repo-URL parameterisation
# ---------------------------------------------------------------------------

# Template with two URL-dependent placeholders for commit and PR link examples.
# The surrounding prompt narrative is fixed; only these two lines vary.
_SYSTEM_PROMPT_TEMPLATE: str = """You are a principal engineer on this team, and a teammate just walked over to your desk \
and asked: "why is this code written the way it is?" Your job is to answer that question — \
thoroughly, honestly, and in your own voice.

Write as a colleague, not as a documentation generator. Use first-person plural: "we added", \
"the team decided", "you'll notice". Be authoritative and direct. Over-explaining is fine; \
under-explaining is not. Your reader is a new engineer who is sharp and experienced — \
don't dumb it down, but do connect the dots they can't see from the code alone.

---

## The story you are telling

Think of the code's history like the evolution of a city over 50 years. Don't just list changes — \
narrate how things got to where they are today:

- Open by describing what this code was solving originally, or what the landscape looked like before \
  the first notable change. If there's no early history to draw on, say so plainly and reason from \
  the earliest evidence you do have.
- Walk through each major change in chronological order (oldest to newest). For each one, explain \
  what changed and, more importantly, why the team made that call at the time. Write in flowing prose \
  paragraphs. "The first major overhaul came when..." is the right register.
- Close with a paragraph that ties it all together: "Today it looks like this because..."

Use Markdown headers to break up major phases if the history warrants it. Do not use bullet \
lists as a substitute for prose — use them only for genuinely enumerable things (e.g., a list \
of edge cases a guard handles).

---

## Citing your sources inline

Every factual claim about intent or causation must be grounded in a commit, PR, or observable \
code structure. Cite inline — do not collect citations at the bottom.

When you reference a commit, render its short SHA as a GitHub link:
{commit_link_example}

When you reference a PR, link to it inline:
{pr_link_example}

If a commit message explicitly states the reason, say so naturally: "the commit message explains \
that..." or "as the PR description put it, ...". If you're reading intent from a diff pattern \
rather than stated text, say so: "reading the diff, it looks like..." or "the way the guard was \
written suggests...".

---

## Expressing uncertainty

Do not assign a confidence score. Instead, express uncertainty in plain language as you go:

- "We don't have much history here, but reading the code it looks like..."
- "The commit message doesn't say explicitly, but the diff shows..."
- "It's not clear from the history why this was done this way — one reading is..."

If there is genuinely no commit history, say so upfront, then reason carefully from the code \
structure — naming conventions, constants, separation of concerns, defensive guards, invariants. \
Describe what those imply in natural prose, not as tagged labels.

---

## Hard constraints (non-negotiable)

1. Every claim must be grounded in a commit, PR, or observable code structure. Do not invent \
   intent or context that isn't in the diff or PR body.

2. Chronological order: walk changes oldest to newest. Do not attribute intent backwards in time.

3. Scope lock: only explain the provided code region. Do not wander into unrelated file changes.

4. If there is a Sparse History Notice in the user message, read the code structure carefully and \
   surface design decisions implied by naming conventions, separation of concerns, constants, \
   defensive patterns, and invariants. Describe these in natural prose as part of the narrative — \
   not as a separate tagged section.

---

## If history is truly insufficient

If the inputs don't give you enough to say anything meaningful, say so plainly in one or two \
sentences and explain what's missing. Do not produce a skeletal or placeholder response.\""""


def build_system_prompt(repo_url: str | None = None) -> str:
    """Build the system prompt, optionally embedding a real repo URL.

    When repo_url is provided (e.g. "https://github.com/org/repo"), the commit
    and PR link examples in the "Citing your sources inline" section use real URLs.
    When repo_url is None, generic `<repo-url>` placeholders are used instead.
    """
    if repo_url is not None and "github.com" in repo_url:
        # Only use real GitHub link format for github.com repos; other hosts
        # (GitLab, Bitbucket, etc.) use different URL structures so fall back
        # to the generic placeholder for them.
        base = repo_url.rstrip("/")
        commit_link_example = f"[`abc1234`]({base}/commit/<full_sha>)"
        pr_link_example = f"[PR #42]({base}/pull/42)"
    else:
        # Generic placeholders — no hardcoded owner/repo
        commit_link_example = "[`abc1234`](<repo-url>/commit/<full_sha>)"
        pr_link_example = "[PR #42](<repo-url>/pull/42)"

    return _SYSTEM_PROMPT_TEMPLATE.format(
        commit_link_example=commit_link_example,
        pr_link_example=pr_link_example,
    )


# Backward-compatible module-level constant; callers that import WHY_SYSTEM_PROMPT
# directly continue to work — it uses the generic placeholder form (no repo URL).
WHY_SYSTEM_PROMPT: str = build_system_prompt()  # no-URL fallback


# ---------------------------------------------------------------------------
# CommitWithPR — pairs a Commit with an optional PR body and patch text
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommitWithPR:
    """A commit paired with its associated PR description and diff patch."""

    commit: Commit
    pr_body: str | None = None
    diff: str = ""  # patch text from get_commit_diff(); empty = not yet fetched


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_target(target: Target) -> str:
    """Return the ## Target section as a string.

    Examples:
      File: `src/foo.py`
      File: `src/foo.py`, Line: 42
      File: `src/foo.py`, Line: 42, Symbol: `bar`
      File: `src/foo.py`, Symbol: `bar`
    """
    # Sanitize file path newlines to prevent section injection
    safe_file = str(target.file).replace("\n", " ").replace("\r", "")
    line = f"File: `{safe_file}`"

    if target.line is not None:
        line += f", Line: {target.line}"

    if target.symbol is not None:
        # Sanitize symbol to prevent injection
        safe_symbol = str(target.symbol).replace("\n", " ").replace("\r", "")
        line += f", Symbol: `{safe_symbol}`"

    return f"## Target\n\n{line}"


def _render_commit(cwpr: CommitWithPR) -> str:
    """Render a single CommitWithPR as a Markdown commit section.

    Format:
      ### `<short_sha>` — "<subject>" · <YYYY-MM-DD> · <author_name>

      **Diff:**

      ```diff
      <diff or "(no diff available)">
      ```

      **PR Body:**

      <fenced pr_body or "N/A">

      ---
    """
    commit = cwpr.commit

    date_str = commit.date.strftime("%Y-%m-%d")

    # Strip newlines from subject and author to prevent injecting new Markdown sections
    safe_subject = commit.subject.replace("\n", " ").replace("\r", "")
    # author_email intentionally omitted — PII not needed for archaeology context
    safe_author = commit.author_name.replace("\n", " ").replace("\r", "")

    # Use the explicit diff field; fall back to placeholder when not yet fetched
    diff_text = cwpr.diff if cwpr.diff else "(no diff available)"

    # Escape triple backticks in diff content to prevent premature fence closure
    safe_diff = diff_text.replace("```", "\\`\\`\\`")

    # Escape triple backticks in pr_body to prevent premature text-fence closure.
    # Both None and "" fall through to "N/A".
    safe_pr_body = cwpr.pr_body.replace("```", "\\`\\`\\`") if cwpr.pr_body else None
    pr_section = f"```text\n{safe_pr_body}\n```" if safe_pr_body else "N/A"

    return (
        f'### `{commit.short_sha}` — "{safe_subject}" · {date_str} · {safe_author}\n'
        f"\n"
        f"**Diff:**\n"
        f"\n"
        f"```diff\n"
        f"{safe_diff}\n"
        f"```\n"
        f"\n"
        f"**PR Body:**\n"
        f"\n"
        f"{pr_section}\n"
        f"\n"
        f"---"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_why_prompt(
    target: Target,
    current_code: str,
    key_commits: list[CommitWithPR],
) -> list[Message]:
    """Build the user message payload for the why LLM synthesis call.

    Returns a single-element list containing a Message(role='user') whose
    content is a Markdown document with three sections: Target (the code
    region under analysis), Current Code (the current snapshot), and Commits
    (each key commit with diff and PR body, oldest to newest).
    """
    target_section = _render_target(target)

    safe_current_code = current_code.replace("```", "\\`\\`\\`")
    code_section = f"## Current Code\n\n```python\n{safe_current_code}\n```"

    # Build commit sub-sections unconditionally; join is empty string when no commits
    commits_body = "\n\n".join(_render_commit(cwpr) for cwpr in key_commits)
    commits_section = f"## Commits\n\n{commits_body}" if commits_body else "## Commits"

    # Sparse-history notice: instruct LLM to inspect code structure when commits < threshold.
    # 0 commits gets distinct wording (no mention of "0 commit(s)"); 1-2 gets count-aware wording.
    sparse_sections = []
    if len(key_commits) < SPARSE_COMMIT_THRESHOLD:
        n = len(key_commits)
        if n == 0:
            sparse_notice = (
                "## Sparse History Notice\n\n"
                "No git history is available for this region. "
                "Analyse the code structure itself and surface design decisions implied by "
                "the code. Look for: naming conventions, separation of concerns, "
                "module-level constants, defensive patterns, and invariants. "
                "Describe these in natural prose as part of the narrative."
            )
        else:
            sparse_notice = (
                f"## Sparse History Notice\n\n"
                f"The git history for this region is sparse ({n} commit(s)). "
                f"In addition to explaining what the commits show, analyse the code "
                f"structure itself and surface any design decisions implied by the code "
                f"but not explained by the commit messages. Look for: naming conventions, "
                f"separation of concerns, module-level constants, defensive patterns, and "
                f"invariants. Distinguish commit-backed reasoning from structural inference."
            )
        sparse_sections = [sparse_notice]

    # Join all sections with horizontal-rule separators.
    # Sparse notice is placed before commits so that "## Commits" remains the final
    # section when key_commits is empty, preserving the invariant that there is no
    # trailing "\n\n" after the commits header in the empty case.
    all_sections = [target_section, code_section, *sparse_sections, commits_section]
    content = "\n\n---\n\n".join(all_sections)

    return [Message(role="user", content=content)]
