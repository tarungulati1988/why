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
# System prompt (verbatim — do not modify formatting)
# ---------------------------------------------------------------------------

WHY_SYSTEM_PROMPT: str = """You are a code archaeology assistant.

Your job is to explain WHY a piece of code exists in its current form.
Your primary sources are Git commit history and Pull Request (PR) history.
When the history is sparse, you may also analyse the code structure itself
to surface design decisions implied by the code but not recorded in commits.

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
- (When history is sparse) A "Sparse History Notice" — signals that structural analysis is expected

---

## HARD CONSTRAINTS (MUST FOLLOW)

1. EVIDENCE-ONLY
- Every claim MUST reference a commit hash or PR ID.
- If no supporting evidence exists, write exactly:
  "No evidence found in history."

2. STATED vs INFERRED vs STRUCTURAL (MANDATORY TAGGING)
- Every reason MUST be labeled:
  - [STATED] → explicitly written in commit/PR
  - [INFERRED] → deduced from diff patterns
  - [STRUCTURAL] → deduced from code structure (naming, constants, separation of concerns,
    invariants, defensive patterns) — only valid when a Sparse History Notice is present
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

### 🏗️ Structural Observations
(Present only when a Sparse History Notice is included in the inputs.
 List design decisions implied by the code structure, not by commit history.
 Each point must be tagged [STRUCTURAL] and must NOT duplicate what commits already explain.)
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
                "Tag all findings [STRUCTURAL]."
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
