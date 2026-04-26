"""Timeline validation and deterministic rendering for why LLM output.

This module provides two public functions:

- ``render_deterministic_timeline``: builds the ASCII timeline block directly
  from ``key_commits`` without calling the LLM, ensuring correctness.

- ``validate_and_repair_timeline``: checks that every SHA in an LLM-generated
  timeline block is known; replaces the block with a deterministic one if not.
"""

from __future__ import annotations

import re

from why.prompts import CommitWithPR


def render_deterministic_timeline(key_commits: list[CommitWithPR]) -> str:
    """Generate the ## 📊 Timeline section directly from key_commits.

    Iterates key_commits in the order provided (callers pass a sorted list).
    Each row is: ``YYYY-MM-DD  <short_sha>  <subject>`` with an optional
    ``  [PR #N]`` suffix when the commit's pr_number field is set.

    Returns a plain "No commit history available." message when the list is empty.
    """
    if not key_commits:
        return "## 📊 Timeline\n\nNo commit history available."

    rows: list[str] = []
    for cwpr in key_commits:
        date_str = cwpr.commit.date.strftime("%Y-%m-%d")
        short_sha = cwpr.commit.short_sha
        # Sanitize subject: replace newlines with spaces, strip carriage returns
        safe_subject = cwpr.commit.subject.replace("\n", " ").replace("\r", "")

        row = f"{date_str}  {short_sha}  {safe_subject}"

        # Append PR annotation using the explicit pr_number field (source of truth)
        if cwpr.pr_number is not None:
            row += f"  [PR #{cwpr.pr_number}]"

        rows.append(row)

    body = "\n".join(rows)
    return f"## 📊 Timeline\n\n```text\n{body}\n```"


def validate_and_repair_timeline(
    response: str,
    key_commits: list[CommitWithPR],
    repo_url: str | None = None,
) -> str:
    """Validate the ## 📊 Timeline section in an LLM response and repair if needed.

    Steps:
    1. If no ## 📊 Timeline section is found, append a deterministic one and return.
    2. If present, extract all 7-char hex tokens from the timeline block.
    3. If any token is not in the set of known short SHAs, replace the entire
       timeline section with a deterministic render.
    4. If all tokens are known (or no tokens found), return response unchanged.

    The ``repo_url`` parameter is accepted for forward compatibility but is not
    currently used in rendering.
    """
    # Check whether a timeline section exists in the response
    section_match = re.search(
        r"(## 📊 Timeline.*?)(?=\n## |\Z)", response, re.DOTALL
    )

    if section_match is None:
        # No timeline section — append a deterministic one
        return response + "\n\n" + render_deterministic_timeline(key_commits)

    # Extract the matched timeline block text
    timeline_block = section_match.group(1)

    # Find all 7-char lowercase hex tokens within the block
    extracted_shas = re.findall(r"\b[0-9a-f]{7}\b", timeline_block)

    if not extracted_shas:
        # No hex tokens — nothing to validate, return unchanged
        return response

    # Build the set of known short SHAs from the provided commits
    known_shas = {cwpr.commit.short_sha for cwpr in key_commits}

    # Check if every extracted SHA is in the known set
    all_valid = all(sha in known_shas for sha in extracted_shas)

    if all_valid:
        return response

    # At least one hallucinated SHA — replace the entire timeline section.
    # Use a callable replacement (lambda) so Python does NOT interpret backreference
    # sequences (e.g. \1, \g<0>) in the replacement string — commit subjects that
    # contain such patterns would otherwise corrupt the output or raise re.error.
    replacement = render_deterministic_timeline(key_commits)
    return re.sub(
        r"## 📊 Timeline.*?(?=\n## |\Z)",
        lambda _: replacement,
        response,
        flags=re.DOTALL,
    )
