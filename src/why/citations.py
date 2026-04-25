"""Citation validator for synthesize_why output.

Post-processing pass that checks every SHA and PR number mentioned in the LLM
output against the set of known SHAs and PRs derived from the input context.
Any reference that cannot be traced back to the input is flagged as a potential
hallucination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class ValidationIssue:
    kind: Literal["unknown_sha", "unknown_pr"]
    value: str       # the offending SHA or PR number string
    output_line: str # the full line from the LLM output that contained it


def validate_citations(
    output: str,
    known_shas: set[str],
    known_prs: set[int],
    strict: bool = False,
) -> list[ValidationIssue]:
    """Post-processing pass: verify every SHA and PR number mentioned in output
    appears in the input context. Flag hallucinations.

    - Extracts all short SHAs (7+ hex chars) from output, checks against known_shas
      using prefix matching (bidirectional startswith).
    - Extracts all PR references (#NNN) from output, checks against known_prs.
    - If strict=True and issues found, raises ValueError.
    - If strict=False, returns list of issues (may be empty).
    """
    issues: list[ValidationIssue] = []

    for line in output.splitlines():
        # Only validate SHA references when the caller supplied known SHAs;
        # skipping prevents every hex token in LLM output from being a false positive.
        if known_shas:
            for m in re.finditer(r"\b[a-f0-9]{7,40}\b", line):
                sha = m.group()
                # Bidirectional prefix check: short ref or longer ref against stored SHA.
                if not any(s.startswith(sha) or sha.startswith(s) for s in known_shas):
                    safe_line = "".join(c for c in line if c.isprintable())
                    issues.append(ValidationIssue("unknown_sha", sha, safe_line))

        # Only validate PR references when the caller supplied known PRs;
        # skipping prevents every #NNN in LLM output from being a false positive.
        if known_prs:
            for m in re.finditer(r"#(\d+)", line):
                pr = int(m.group(1))
                if pr not in known_prs:
                    safe_line = "".join(c for c in line if c.isprintable())
                    issues.append(ValidationIssue("unknown_pr", str(pr), safe_line))

    if strict and issues:
        raise ValueError(f"citation validation failed: {len(issues)} issues")

    return issues
