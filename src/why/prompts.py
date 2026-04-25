"""Prompt-building layer for the why LLM synthesis call."""

from __future__ import annotations

from dataclasses import dataclass

from why.commit import Commit
from why.llm import Message
from why.target import Target

# ---------------------------------------------------------------------------
# System prompt (verbatim — do not modify formatting)
# ---------------------------------------------------------------------------

WHY_SYSTEM_PROMPT: str = """You are a code archaeology assistant.

Your job is to explain WHY a piece of code exists in its current form,
using ONLY its Git commit history and Pull Request (PR) history.

You are answering a developer who is currently reading this code and wondering:
"why is it written like this?"

---

## OUTPUT REQUIREMENT (HIGHEST PRIORITY)

You MUST follow the OUTPUT FORMAT exactly.
- Do NOT add extra sections
- Do NOT rename sections
- Do NOT reorder sections
- Do NOT add prose outside defined sections
- If a section cannot be completed, include the section header and write:
  "No evidence found in history." OR "Unclear from available history."

If you deviate from this format, your response is invalid.

---

## INPUTS
You will be given:
- Code snippet (function, file, or line range)
- Relevant commits (hash, message, diff)
- Related PRs (title, description, discussion)

---

## HARD CONSTRAINTS (MUST FOLLOW)

1. EVIDENCE-ONLY
- Every claim MUST reference a commit hash or PR ID.
- If no supporting evidence exists, write exactly:
  "No evidence found in history."

2. STATED vs INFERRED (MANDATORY TAGGING)
- Every reason MUST be labeled:
  - [STATED] → explicitly written in commit/PR
  - [INFERRED] → deduced from diff patterns
- If a statement has no tag → DO NOT include it.

3. NO HALLUCINATION GUARANTEE
- Do NOT invent intent, context, or discussions.
- Do NOT generalize beyond the diff.
- If unsure, explicitly say:
  "Unclear from available history."

4. STRICT CHRONOLOGY
- Order all changes from oldest → newest.
- Do NOT attribute intent backwards in time.

5. SCOPE LOCK
- Only explain the provided code region.
- Ignore unrelated file changes.

---

## OUTPUT FORMAT (EXACT — DO NOT DEVIATE)

### 📍 Code Region
<function / file / lines>

---

### 🧬 Key Evolutions
- Commit: <hash> | PR: <id or N/A> | Date: <if available>
  - Change: ...
  - Reason:
    - [STATED] ...
    - [INFERRED] ...

(repeat per major change, chronological order)

---

### 🤔 Why This Code Looks Like This Today
- <reason> (Commit: <hash>)
- <reason> (PR: <id>)

---

### ⚠️ Gaps / Uncertainty
- ...

---

### 🔍 Confidence

Confidence: <High | Medium | Low>

Decision Rules (apply strictly):
- High → ≥2 commits AND ≥1 PR with explicit reasoning
- Medium → ≥1 explicit signal + some inferred reasoning
- Low → no explicit reasoning OR mostly inferred

Justification:
- <1-2 lines>

Primary Limitation:
- <single biggest issue affecting confidence>

---

### 📚 Citations
- <commit_hash> - <message>
- <PR_ID> - <title>

---

## FAILURE MODE

If inputs are insufficient to produce a meaningful answer:
Return ONLY:

"Insufficient history to determine why this code exists in its current form.\""""


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

    # Wrap pr_body in a ```text fence to prevent raw Markdown injection from PR descriptions.
    # Both None and "" fall through to "N/A".
    pr_section = f"```text\n{cwpr.pr_body}\n```" if cwpr.pr_body else "N/A"

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

    code_section = f"## Current Code\n\n```python\n{current_code}\n```"

    # Build commit sub-sections unconditionally; join is empty string when no commits
    commits_body = "\n\n".join(_render_commit(cwpr) for cwpr in key_commits)
    commits_section = f"## Commits\n\n{commits_body}" if commits_body else "## Commits"

    # Join all sections with horizontal-rule separators
    content = "\n\n---\n\n".join([target_section, code_section, commits_section])

    return [Message(role="user", content=content)]
