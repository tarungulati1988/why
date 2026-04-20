"""Commit dataclass and porcelain log parser for why.

This module owns two things:
1. ``Commit`` — an immutable, typed representation of a single git commit.
2. ``parse_porcelain`` — converts raw ``git log`` output (using the constants
   below) into ``list[Commit]``.

Design note — why single-pass (option 1):
  ``PORCELAIN_FORMAT`` and ``SHORTSTAT_FLAG`` are designed to be passed together
  in a single ``git log`` invocation.  Because shortstat is already interleaved
  with commit metadata in the same stdout blob, one parser can walk it in a
  single pass — no second subprocess, no SHA-keyed merge step.

  The alternative (option 2) is to run two git commands:
    - ``git log --format=PORCELAIN_FORMAT`` for metadata only
    - ``git log --shortstat --format=%H``   for stat lines
  … keep each parser simple and focused, then zip the two result lists by SHA.

  Concrete refactor trigger:
    If we ever want per-file stats via ``git diff --numstat``, or stats from a
    cached DB that doesn't call git, split metadata and stats parsing and zip
    by SHA.  At that point the two data sources diverge and the single-pass
    design becomes an obstacle.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format constants — these two constants are used together in one git call:
#   git log --format=<PORCELAIN_FORMAT> <SHORTSTAT_FLAG>
#
# Field separator: NULL (\x00) — safe inside commit messages (git strips it
# from log output anyway, but in practice subjects/bodies never contain NUL).
# Record separator: ASCII Record Separator (\x1e) — terminates each commit so
# shortstat lines that follow don't bleed into the next chunk.
# ---------------------------------------------------------------------------

PORCELAIN_FORMAT = "%H%x00%an%x00%ae%x00%aI%x00%s%x00%b%x00%P\x1e"
SHORTSTAT_FLAG = "--shortstat"

# Regex to identify a shortstat summary line, e.g.:
#   " 3 files changed, 10 insertions(+), 4 deletions(-)"
#   " 1 file changed, 5 insertions(+)"
_SHORTSTAT_LINE_RE = re.compile(r"^\s*\d+ files? changed", re.MULTILINE)

# Two optional capture groups — a commit may have only insertions OR only
# deletions OR both.
_STAT_RE = re.compile(r"(\d+) insertions?\(\+\)|(\d+) deletions?\(-\)")


# ---------------------------------------------------------------------------
# Commit dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Commit:
    """Immutable representation of a single git commit."""

    sha: str
    author_name: str
    author_email: str
    date: datetime
    subject: str
    body: str
    parents: tuple[str, ...]
    additions: int = 0
    deletions: int = 0

    @property
    def short_sha(self) -> str:
        """First 7 hex characters of the commit SHA."""
        return self.sha[:7]

    @property
    def is_merge(self) -> bool:
        """True iff this commit has more than one parent (i.e. it's a merge)."""
        return len(self.parents) > 1


# ---------------------------------------------------------------------------
# Parser helpers (module-level so mypy sees their signatures)
# ---------------------------------------------------------------------------


def _extract_stats(shortstat_text: str) -> tuple[int, int]:
    """Return (additions, deletions) extracted from a shortstat line."""
    additions = 0
    deletions = 0
    for ins_str, del_str in _STAT_RE.findall(shortstat_text):
        if ins_str:
            additions = int(ins_str)
        if del_str:
            deletions = int(del_str)
    return additions, deletions


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_porcelain(output: str) -> list[Commit]:
    """Parse raw ``git log`` output into a list of :class:`Commit` objects.

    ``output`` must be the stdout of::

        git log --format=<PORCELAIN_FORMAT> --shortstat

    Real git output structure when split on \\x1e (record separator):
      chunk 0: <commit0_NUL_fields>
      chunk 1: \\n<commit1_NUL_fields>
      chunk 2: \\n\\n<shortstat_for_commit1>\\n<commit2_NUL_fields>
      ...
      chunk N: \\n\\n<shortstat_for_commit_N-1>\\n<commitN_NUL_fields>
      chunk N+1: \\n\\n<shortstat_for_commitN>\\n   (tail, no more commit fields)

    In other words the shortstat from commit K appears as a *prefix* in the
    chunk that also contains commit K+1's NUL-delimited fields.  The tail
    chunk (after the last commit) is shortstat-only or empty.

    Pipeline:
    1. Split on \\x1e; lstrip("\\n") each chunk.
    2. For each chunk:
       a. Peel an optional leading shortstat block (regex on "N files? changed").
          This shortstat belongs to the commit parsed in the *previous* iteration.
       b. If any NUL-delimited fields remain, parse them as the current commit.
    3. At the end, patch the *last* built commit with the trailing shortstat
       (from the final tail chunk, if it exists).

    Raises :class:`ValueError` if any commit chunk is structurally malformed.
    """
    if not output:
        return []

    # Step 1: split on record separator and normalise leading newlines.
    raw_chunks = output.split("\x1e")
    chunks = [c.lstrip("\n") for c in raw_chunks]

    logger.debug("parse_porcelain: %d chunks", len(chunks))

    # pending holds the last-parsed Commit so we can patch its stats when the
    # next chunk reveals the shortstat for it.
    pending: Commit | None = None
    commits: list[Commit] = []

    for i, chunk in enumerate(chunks):
        # Step 2a: peel an optional leading shortstat block.
        # Only search the region before the first NUL — the shortstat always
        # appears as a leading prefix before the NUL-delimited commit fields.
        # Searching the whole chunk risks matching a body line like
        # " 3 files changed today" as a false shortstat.
        first_null = chunk.find("\x00")
        search_region = chunk if first_null == -1 else chunk[:first_null]
        shortstat_match = _SHORTSTAT_LINE_RE.search(search_region)
        shortstat_text = ""
        if shortstat_match:
            # Shortstat for the *previous* commit.  Take everything up to the
            # end of the shortstat line(s), then let the remainder be the
            # current commit's fields.
            #
            # The shortstat block ends at the first line that does NOT look
            # like stats — i.e., the first line containing a NUL byte or the
            # start of the next commit's sha.  In practice git emits exactly
            # one shortstat line, so we find the newline after it.
            stat_start = shortstat_match.start()
            # Find where the shortstat block ends: the newline terminating the
            # summary line.  Everything after that is the next commit's fields.
            stat_end = chunk.find("\n", stat_start)
            if stat_end == -1:
                # Shortstat is the entire remainder of the chunk (tail chunk).
                shortstat_text = chunk[stat_start:]
                chunk = chunk[:stat_start]
            else:
                shortstat_text = chunk[stat_start : stat_end + 1]
                chunk = chunk[:stat_start] + chunk[stat_end + 1 :]

            # Patch the previous commit's stats using dataclasses.replace —
            # no mutable dict needed; frozen dataclass is copied with new values.
            if pending is not None:
                adds, dels = _extract_stats(shortstat_text)
                pending = replace(pending, additions=adds, deletions=dels)

        # Step 2b: strip residual whitespace and check for commit fields.
        chunk = chunk.strip()
        if not chunk:
            # Empty or pure-shortstat chunk — no new commit to parse.
            continue

        # Flush the previous pending commit before starting a new one.
        if pending is not None:
            commits.append(pending)

        # Split on NUL — must yield exactly 7 fields.
        fields = chunk.split("\x00")
        if len(fields) != 7:
            logger.debug("malformed chunk at index %d: %r", i, chunk)
            raise ValueError(
                f"malformed commit chunk {i}: expected 7 fields, got {len(fields)}"
            )

        sha, author_name, author_email, date_str, subject, body, parents_str = fields

        # Parse date (Python 3.11 handles ISO 8601 timezone offsets natively).
        date = datetime.fromisoformat(date_str)

        # Parents are space-separated SHAs; root commits have an empty field.
        parents: tuple[str, ...] = tuple(parents_str.split()) if parents_str.strip() else ()

        # Build the Commit with zero stats; shortstat (if any) arrives in the
        # next chunk and will be applied via dataclasses.replace.
        pending = Commit(
            sha=sha,
            author_name=author_name,
            author_email=author_email,
            date=date,
            subject=subject,
            body=body,
            parents=parents,
            additions=0,
            deletions=0,
        )

    # Flush the final pending commit (its stats were already patched if the
    # tail chunk contained a shortstat).
    if pending is not None:
        commits.append(pending)

    n = len(commits)
    m = sum(1 for c in commits if c.is_merge)
    logger.debug("parse_porcelain: parsed %d commits (%d merges)", n, m)

    return commits
