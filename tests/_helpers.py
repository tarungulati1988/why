"""Shared test helpers for building Commit and CommitWithPR fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

from why.commit import Commit
from why.prompts import CommitWithPR

_FIXED_DATE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def make_commit(
    sha: str,
    subject: str = "some commit message",
    date: datetime | None = None,
    author_name: str = "Test Author",
    author_email: str = "test@example.com",
    body: str = "",
    additions: int = 0,
    deletions: int = 0,
) -> Commit:
    """Return a minimal Commit suitable for testing."""
    return Commit(
        sha=sha,
        author_name=author_name,
        author_email=author_email,
        date=date if date is not None else _FIXED_DATE,
        subject=subject,
        body=body,
        parents=(),
        additions=additions,
        deletions=deletions,
    )


def make_cwpr(
    sha: str,
    diff: str = "",
    subject: str = "some commit message",
    pr_title: str | None = None,
    pr_body: str | None = None,
    date: datetime | None = None,
) -> CommitWithPR:
    """Return a CommitWithPR suitable for testing."""
    return CommitWithPR(
        commit=make_commit(sha, subject=subject, date=date),
        diff=diff,
        pr_title=pr_title,
        pr_body=pr_body,
    )
