"""Tests for src/why/synth.py — internal helpers _extract_current_code and _resolve_line_range,
and the public synthesize_why orchestration function."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from why.commit import Commit
from why.llm import LLMClient
from why.symbols import SymbolNotFoundError
from why.synth import _extract_current_code, _resolve_line_range, synthesize_why
from why.target import Target

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_py_file(tmp_path: Path, name: str, content: str) -> Path:
    """Write a real Python file to disk so tree-sitter can parse it."""
    f = tmp_path / name
    f.write_text(content)
    return f


def _make_large_py_file(tmp_path: Path) -> Path:
    """Create a Python file with >40 lines to test the line-window slicing."""
    lines = [f"# line {i}\n" for i in range(1, 60)]
    # Insert a real function so tree-sitter can find it
    func_lines = [
        "def target_func():\n",
        "    return 42\n",
    ]
    # function at lines 20-21 within the 59-line file
    lines[19:19] = func_lines  # insert before line 20 (0-indexed 19)
    content = "".join(lines)
    return _make_py_file(tmp_path, "large.py", content)


# ---------------------------------------------------------------------------
# _extract_current_code
# ---------------------------------------------------------------------------

class TestExtractCurrentCodeSymbol:
    def test_symbol_returns_only_function_lines(self, tmp_path: Path) -> None:
        content = (
            "x = 1\n"
            "y = 2\n"
            "def my_func():\n"
            "    return 99\n"
            "z = 3\n"
        )
        f = _make_py_file(tmp_path, "sample.py", content)
        target = Target(file=f, symbol="my_func")
        # Resolve the line range as synthesize_why does, then pass it in.
        line_range = _resolve_line_range(target)

        result = _extract_current_code(target, line_range)

        assert "def my_func():" in result
        assert "return 99" in result
        # Must NOT contain lines outside the function
        assert "x = 1" not in result
        assert "z = 3" not in result

    def test_symbol_requires_line_range(self, tmp_path: Path) -> None:
        """Passing line_range=None for a symbol target triggers the assertion guard."""
        f = _make_py_file(tmp_path, "s.py", "def foo():\n    pass\n")
        target = Target(file=f, symbol="foo")
        with pytest.raises(AssertionError):
            _extract_current_code(target, None)


class TestExtractCurrentCodeLine:
    def test_line_target_returns_window_smaller_than_full_file(self, tmp_path: Path) -> None:
        f = _make_large_py_file(tmp_path)
        full_text = f.read_text()
        # Pick a line somewhere in the middle
        target = Target(file=f, line=30)

        result = _extract_current_code(target)

        assert len(result) < len(full_text)

    def test_line_at_start_clamps_without_negative_indices(self, tmp_path: Path) -> None:
        content = "".join(f"line{i}\n" for i in range(1, 50))
        f = _make_py_file(tmp_path, "clamped.py", content)
        target = Target(file=f, line=1)

        # Must not raise and must contain line 1
        result = _extract_current_code(target)

        assert "line1" in result
        # Window is clamped — result should be non-empty
        assert result.strip()


class TestExtractCurrentCodeFileOnly:
    def test_no_line_no_symbol_returns_full_file(self, tmp_path: Path) -> None:
        content = "a = 1\nb = 2\nc = 3\n"
        f = _make_py_file(tmp_path, "full.py", content)
        target = Target(file=f)

        result = _extract_current_code(target)

        assert result == content


# ---------------------------------------------------------------------------
# _resolve_line_range
# ---------------------------------------------------------------------------

class TestResolveLineRangeSymbol:
    def test_symbol_returns_start_end_tuple(self, tmp_path: Path) -> None:
        content = (
            "x = 1\n"
            "def my_func():\n"
            "    return 42\n"
            "y = 2\n"
        )
        f = _make_py_file(tmp_path, "sample.py", content)
        target = Target(file=f, symbol="my_func")

        result = _resolve_line_range(target)

        # my_func starts at line 2 and ends at line 3
        assert result == (2, 3)

    def test_symbol_not_found_raises(self, tmp_path: Path) -> None:
        """_resolve_line_range must raise SymbolNotFoundError for missing symbols.

        The error propagates to the CLI so the user sees an explicit message
        instead of a silent fallback to file-scoped history.
        """
        content = "def exists():\n    pass\n"
        f = _make_py_file(tmp_path, "sample.py", content)
        target = Target(file=f, symbol="ghost")

        with pytest.raises(SymbolNotFoundError):
            _resolve_line_range(target)


class TestResolveLineRangeLine:
    def test_line_target_returns_line_line_tuple(self, tmp_path: Path) -> None:
        content = "a = 1\nb = 2\nc = 3\n"
        f = _make_py_file(tmp_path, "sample.py", content)
        target = Target(file=f, line=2)

        result = _resolve_line_range(target)

        assert result == (2, 2)


class TestResolveLineRangeFileOnly:
    def test_no_line_no_symbol_returns_none(self, tmp_path: Path) -> None:
        content = "a = 1\n"
        f = _make_py_file(tmp_path, "sample.py", content)
        target = Target(file=f)

        result = _resolve_line_range(target)

        assert result is None


# ---------------------------------------------------------------------------
# synthesize_why
# ---------------------------------------------------------------------------

# Fixture helpers

def _make_commit(sha: str) -> Commit:
    """Return a minimal Commit with the given SHA, suitable for testing."""
    return Commit(
        sha=sha,
        author_name="Alice",
        author_email="alice@example.com",
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),  # noqa: UP017 — datetime.UTC absent in 3.11.0b1
        subject=f"commit {sha[:7]}",
        body="",
        parents=(),
    )


# Patch targets — all patched on the synth module so the function uses them
_PATCH_FILE_HISTORY = "why.synth.get_file_history"
_PATCH_LINE_HISTORY = "why.synth.get_line_history"
_PATCH_SELECT = "why.synth.select_key_commits"
_PATCH_DIFF = "why.synth.get_commit_diff"
_PATCH_BUILD_PROMPT = "why.synth.build_why_prompt"
_PATCH_EXTRACT_CODE = "why.synth._extract_current_code"
_PATCH_RESOLVE_RANGE = "why.synth._resolve_line_range"
_PATCH_GET_REPO_URL = "why.synth._get_repo_url"


class TestSynthesizeWhyHappyPath:
    """History ≥3: select_key_commits is called; llm.complete receives WHY_SYSTEM_PROMPT."""

    def test_calls_select_key_commits_and_returns_llm_output(self, tmp_path: Path) -> None:
        commits = [_make_commit(f"aaa{i}" * 10) for i in range(3)]
        key = commits[:2]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        fake_messages = [MagicMock()]

        llm = MagicMock()
        llm.complete.return_value = "the answer"

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT, return_value=key) as mock_select,
            patch(_PATCH_DIFF, return_value="diff text"),
            patch(_PATCH_EXTRACT_CODE, return_value="code snippet"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=fake_messages),
            patch(_PATCH_GET_REPO_URL, return_value=None),  # no remote in tmp_path
        ):
            result = synthesize_why(target, tmp_path, llm)

        # select_key_commits must have been called with the full history
        mock_select.assert_called_once_with(commits, {})
        # llm.complete must receive build_system_prompt(None) as system arg
        from why.prompts import build_system_prompt
        llm.complete.assert_called_once_with(build_system_prompt(None), fake_messages)
        # synthesize_why appends a ## 📊 Timeline block after the LLM response
        assert result.startswith("the answer")


class TestSynthesizeWhySparsePath:
    """History <3: select_key_commits is NOT called; all commits passed directly."""

    def test_skips_scoring_for_two_commits(self, tmp_path: Path) -> None:
        commits = [_make_commit("bbb1" * 10), _make_commit("bbb2" * 10)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        fake_messages = [MagicMock()]

        llm = MagicMock()
        llm.complete.return_value = "sparse answer"

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT) as mock_select,
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=fake_messages) as mock_prompt,
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, tmp_path, llm)

        # select_key_commits must NOT have been called
        mock_select.assert_not_called()
        # Both commits must have been forwarded to build_why_prompt
        call_args = mock_prompt.call_args
        commits_with_prs = call_args.args[2]
        assert len(commits_with_prs) == 2
        # synthesize_why appends a ## 📊 Timeline block after the LLM response
        assert result.startswith("sparse answer")


class TestSynthesizeWhyNoHistory:
    """Empty history returns the sentinel string without touching the LLM."""

    def test_returns_no_history_message(self, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()

        with (
            patch(_PATCH_FILE_HISTORY, return_value=[]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, tmp_path, llm)

        assert result == "file has no git history"
        llm.complete.assert_not_called()


class TestSynthesizeWhyLineTargetRouting:
    """When target.line is set, get_line_history is called instead of get_file_history."""

    def test_uses_get_line_history(self, tmp_path: Path) -> None:
        commits = [_make_commit("ccc1" * 10), _make_commit("ccc2" * 10)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f, line=5)
        llm = MagicMock()
        llm.complete.return_value = "line answer"

        with (
            patch(_PATCH_FILE_HISTORY) as mock_file_hist,
            patch(_PATCH_LINE_HISTORY, return_value=commits) as mock_line_hist,
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=(5, 5)),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm)

        mock_line_hist.assert_called_once_with(f, tmp_path, line=5)
        mock_file_hist.assert_not_called()


class TestSynthesizeWhyFileOnlyRouting:
    """When no line or symbol, get_file_history is called."""

    def test_uses_get_file_history(self, tmp_path: Path) -> None:
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        llm.complete.return_value = "file answer"

        with (
            patch(_PATCH_FILE_HISTORY, return_value=[]) as mock_file_hist,
            patch(_PATCH_LINE_HISTORY) as mock_line_hist,
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm)

        mock_file_hist.assert_called_once_with(f, tmp_path)
        mock_line_hist.assert_not_called()


class TestSynthesizeWhyPRBodyWiring:
    """prs dict is forwarded to select_key_commits AND populates CommitWithPR.pr_body."""

    def test_pr_body_populated_for_matching_shas(self, tmp_path: Path) -> None:
        sha_a = "aaa0" * 10
        sha_b = "bbb0" * 10
        sha_c = "ccc0" * 10
        commits = [_make_commit(sha_a), _make_commit(sha_b), _make_commit(sha_c)]
        # Only sha_a has a PR body; sha_c does not
        prs = {sha_a: "PR body for A"}
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        llm.complete.return_value = "pr answer"

        captured_commits_with_prs = []

        def fake_build_prompt(t, code, cwprs, **kwargs):
            captured_commits_with_prs.extend(cwprs)
            return [MagicMock()]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT, return_value=commits) as mock_select,
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, side_effect=fake_build_prompt),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm, prs=prs)

        # prs must be forwarded to select_key_commits
        mock_select.assert_called_once_with(commits, prs)
        # CommitWithPR for sha_a must have pr_body set; sha_b and sha_c must be None
        pr_bodies = {cwpr.commit.sha: cwpr.pr_body for cwpr in captured_commits_with_prs}
        assert pr_bodies[sha_a] == "PR body for A"
        assert pr_bodies[sha_b] is None
        assert pr_bodies[sha_c] is None


# ---------------------------------------------------------------------------
# Fix #1: symbol target routes to get_line_history, not get_file_history
# ---------------------------------------------------------------------------


class TestSynthesizeWhySymbolRouting:
    """When target.symbol is set and resolves to a line_range, get_line_history is called."""

    def test_symbol_target_uses_get_line_history(self, tmp_path: Path) -> None:
        commits = [_make_commit("sym1" * 10), _make_commit("sym2" * 10)]
        content = "x = 1\ndef my_func():\n    return 42\n"
        f = _make_py_file(tmp_path, "foo.py", content)
        target = Target(file=f, symbol="my_func")
        llm = MagicMock()
        llm.complete.return_value = "symbol answer"

        with (
            patch(_PATCH_FILE_HISTORY) as mock_file_hist,
            patch(_PATCH_LINE_HISTORY, return_value=commits) as mock_line_hist,
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=(2, 3)),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm)

        mock_line_hist.assert_called_once_with(f, tmp_path, line=2)
        mock_file_hist.assert_not_called()


# ---------------------------------------------------------------------------
# Fix #2: GitError per-commit is caught; synthesize_why does not raise
# ---------------------------------------------------------------------------


class TestSynthesizeWhyGitErrorHandling:
    """When get_commit_diff raises GitError for one commit, synthesize_why continues."""

    def test_git_error_on_one_commit_does_not_raise(self, tmp_path: Path) -> None:
        from why.git import GitError

        sha_good = "good" * 10
        sha_bad = "bad0" * 10
        commits = [_make_commit(sha_good), _make_commit(sha_bad)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        llm.complete.return_value = "answer"

        def diff_side_effect(sha, *args, **kwargs):
            if sha == sha_bad:
                raise GitError("boom")
            return "diff text"

        captured: list = []

        def fake_build_prompt(t, code, cwprs, **kwargs):
            captured.extend(cwprs)
            return [MagicMock()]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, side_effect=diff_side_effect),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, side_effect=fake_build_prompt),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, tmp_path, llm)

        # synthesize_why appends a ## 📊 Timeline block after the LLM response
        assert result.startswith("answer")
        bad_cwpr = next(cwpr for cwpr in captured if cwpr.commit.sha == sha_bad)
        assert bad_cwpr.diff == ""


# ---------------------------------------------------------------------------
# Fix #3: sparse-history path returns commits in oldest-first order
# ---------------------------------------------------------------------------


class TestSynthesizeWhySparseHistoryOrdering:
    """With 2-commit history (newest-first), build_why_prompt receives oldest-first."""

    def test_sparse_commits_sorted_oldest_first(self, tmp_path: Path) -> None:
        newer = Commit(
            sha="new0" * 10,
            author_name="Alice",
            author_email="alice@example.com",
            date=datetime(2024, 6, 1, tzinfo=timezone.utc),  # noqa: UP017
            subject="newer commit",
            body="",
            parents=(),
        )
        older = Commit(
            sha="old0" * 10,
            author_name="Alice",
            author_email="alice@example.com",
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),  # noqa: UP017
            subject="older commit",
            body="",
            parents=(),
        )
        commits_newest_first = [newer, older]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        llm.complete.return_value = "answer"

        captured_order: list = []

        def fake_build_prompt(t, code, cwprs, **kwargs):
            captured_order.extend(cwprs)
            return [MagicMock()]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits_newest_first),
            patch(_PATCH_SELECT) as mock_select,
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, side_effect=fake_build_prompt),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm)

        mock_select.assert_not_called()
        assert captured_order[0].commit.sha == older.sha
        assert captured_order[1].commit.sha == newer.sha


class TestSynthesizeWhySequencing:
    """Call order: history → select → diff → prompt → llm."""

    def test_call_order_is_correct(self, tmp_path: Path) -> None:
        commits = [_make_commit(f"seq{i}" * 10) for i in range(3)]
        key = commits[:2]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        llm.complete.return_value = "ordered answer"
        fake_messages = [MagicMock()]

        call_log: list[str] = []

        def track(name: str):
            """Return a side_effect function that appends name to call_log."""
            def _side_effect(*args, **kwargs):
                call_log.append(name)
                # Return values needed by the pipeline
                returns = {
                    "file_history": commits,
                    "select": key,
                    "diff": "d",
                    "extract_code": "code",
                    "resolve_range": None,
                    "build_prompt": fake_messages,
                    "get_repo_url": None,
                }
                return returns[name]
            return _side_effect

        with (
            patch(_PATCH_FILE_HISTORY, side_effect=track("file_history")),
            patch(_PATCH_SELECT, side_effect=track("select")),
            patch(_PATCH_DIFF, side_effect=track("diff")),
            patch(_PATCH_EXTRACT_CODE, side_effect=track("extract_code")),
            patch(_PATCH_RESOLVE_RANGE, side_effect=track("resolve_range")),
            patch(_PATCH_BUILD_PROMPT, side_effect=track("build_prompt")),
            patch(_PATCH_GET_REPO_URL, side_effect=track("get_repo_url")),
        ):
            synthesize_why(target, tmp_path, llm)

        # history must come before select, select before diff, diff before prompt,
        # prompt before llm.complete
        assert call_log.index("file_history") < call_log.index("select")
        assert call_log.index("select") < call_log.index("diff")
        assert call_log.index("diff") < call_log.index("build_prompt")
        llm.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Fix #5: triple backticks in pr_body are escaped in _render_commit output
# ---------------------------------------------------------------------------


class TestRenderCommitPRBodyEscaping:
    """Triple backticks in pr_body must be escaped to prevent fence breakout."""

    def test_pr_body_triple_backticks_are_escaped(self) -> None:
        from datetime import UTC, datetime

        from why.commit import Commit
        from why.prompts import CommitWithPR, build_why_prompt
        from why.target import Target

        commit = Commit(
            sha="abc1234def5678901234567890",
            author_name="Alice",
            author_email="alice@example.com",
            date=datetime(2024, 1, 1, tzinfo=UTC),
            subject="some commit",
            body="",
            parents=(),
        )
        cwpr = CommitWithPR(commit=commit, pr_body="look at this ```code block```")
        target = Target(file=Path("src/foo.py"))

        result = build_why_prompt(target, "x = 1", [cwpr])
        content = result[0].content

        assert "\\`\\`\\`" in content
        assert 'look at this ```code block```' not in content


# ---------------------------------------------------------------------------
# Fix #6: triple backticks in current_code are escaped in build_why_prompt
# ---------------------------------------------------------------------------


class TestBuildWhyPromptCurrentCodeEscaping:
    """Triple backticks in current_code must be escaped in the output."""

    def test_current_code_triple_backticks_are_escaped(self) -> None:
        from why.prompts import build_why_prompt
        from why.target import Target

        target = Target(file=Path("src/foo.py"))
        current_code = 'def foo():\n    """\n    ```python\n    example\n    ```\n    """\n    pass'

        result = build_why_prompt(target, current_code, [])
        content = result[0].content

        assert "\\`\\`\\`" in content
        assert 'def foo():' in content


# ---------------------------------------------------------------------------
# Citation validation integration tests
# ---------------------------------------------------------------------------

_PATCH_VALIDATE_CITATIONS = "why.synth.validate_citations"
_PATCH_LOG = "why.synth._log"


class TestSynthesizeWhyCitationLogging:
    """When LLM output contains an unknown SHA, validate_citations issues are logged."""

    def test_synthesize_why_logs_citation_issues(self, tmp_path: Path) -> None:
        from why.citations import ValidationIssue

        # A known SHA for the single commit in history
        known_sha = "aaabbb1234567890abcdef1234567890abcd1234"
        # An unknown 7-char hex SHA the LLM hallucinated
        unknown_sha = "dead123"

        commits = [_make_commit(known_sha)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        # LLM output mentions an unknown SHA
        llm.complete.return_value = f"This was changed in {unknown_sha} for reasons."

        fake_issue = ValidationIssue(
            kind="unknown_sha",
            value=unknown_sha,
            output_line=f"This was changed in {unknown_sha} for reasons.",
        )

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_VALIDATE_CITATIONS, return_value=[fake_issue]) as mock_validate,
            patch(_PATCH_LOG) as mock_log,
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, tmp_path, llm)

        # validate_citations must be called with the LLM output and the known SHA set
        mock_validate.assert_called_once()
        call_args = mock_validate.call_args
        assert call_args.args[0] == llm.complete.return_value
        assert known_sha in call_args.args[1]  # known_shas contains the commit SHA
        assert call_args.kwargs.get("known_prs") == set() or call_args.args[2] == set()

        # _log.warning must have been called for the citation issue;
        # synth.py strips non-printable chars from value before logging.
        safe_value = "".join(c for c in fake_issue.value if c.isprintable())
        mock_log.warning.assert_called_with(
            "citation issue %s: %s", fake_issue.kind, safe_value
        )

        # Result starts with the raw LLM output — issues are logged, not filtered.
        # synthesize_why appends a ## 📊 Timeline block after the LLM response.
        assert result.startswith(llm.complete.return_value)

    def test_synthesize_why_no_issues_no_warnings(self, tmp_path: Path) -> None:
        """When LLM output contains only known SHAs, no citation warnings are logged."""
        known_sha = "aaabbb1234567890abcdef1234567890abcd1234"

        commits = [_make_commit(known_sha)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        # LLM output only mentions the known SHA (short form)
        llm.complete.return_value = f"Changed in {known_sha[:7]} to fix a bug."

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_VALIDATE_CITATIONS, return_value=[]) as mock_validate,
            patch(_PATCH_LOG) as mock_log,
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm)

        # validate_citations called but returned no issues — no citation warnings
        mock_validate.assert_called_once()
        # Ensure _log.warning was NOT called with the citation issue pattern
        for call in mock_log.warning.call_args_list:
            assert call.args[0] != "citation issue %s: %s", (
                "Expected no citation warnings, but one was logged"
            )


# ---------------------------------------------------------------------------
# Timeline validation integration tests
# ---------------------------------------------------------------------------


class TestSynthesizeWhyTimeline:
    """validate_and_repair_timeline is called after LLM response; timeline always present."""

    def test_synthesize_why_includes_timeline_section(self, tmp_path: Path) -> None:
        """LLM response already has a valid ## 📊 Timeline block — returned unchanged."""
        sha = "abc1234def5678901234567890"
        commit = _make_commit(sha)
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)

        # Build a response that already contains a valid timeline with the real short SHA.
        short_sha = sha[:7]
        llm_response = (
            "Here is why this changed.\n\n"
            f"## 📊 Timeline\n\n```text\n2024-01-01  {short_sha}  commit abc1234\n```"
        )

        llm = MagicMock()
        llm.complete.return_value = llm_response

        with (
            patch(_PATCH_FILE_HISTORY, return_value=[commit]),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, repo=tmp_path, llm=llm, prs={})

        assert "## 📊 Timeline" in result

    def test_synthesize_why_appends_timeline_when_missing(self, tmp_path: Path) -> None:
        """LLM response has NO ## 📊 Timeline — validate_and_repair_timeline appends one."""
        sha = "abc1234def5678901234567890"
        commit = _make_commit(sha)
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)

        # Response with no timeline section at all
        llm_response = "Here is why this changed — no timeline included."

        llm = MagicMock()
        llm.complete.return_value = llm_response

        with (
            patch(_PATCH_FILE_HISTORY, return_value=[commit]),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, repo=tmp_path, llm=llm, prs={})

        assert "## 📊 Timeline" in result

    def test_synthesize_why_repairs_hallucinated_timeline(self, tmp_path: Path) -> None:
        """LLM response has a timeline with a hallucinated SHA — replaced with real SHA."""
        sha = "abc1234def5678901234567890"
        commit = _make_commit(sha)
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)

        # Timeline contains a hallucinated 7-char SHA not matching the real commit
        hallucinated_sha = "deadbee"
        llm_response = (
            "Here is why this changed.\n\n"
            f"## 📊 Timeline\n\n```text\n2024-01-01  {hallucinated_sha}  fake commit\n```"
        )

        llm = MagicMock()
        llm.complete.return_value = llm_response

        with (
            patch(_PATCH_FILE_HISTORY, return_value=[commit]),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, repo=tmp_path, llm=llm, prs={})

        # The real short SHA should appear in the repaired timeline
        real_short_sha = sha[:7]
        assert real_short_sha in result
        # The hallucinated SHA should NOT appear in the result
        assert hallucinated_sha not in result


class TestSynthesizeWhyStrictMode:
    """When strict=True and LLM output contains an unknown SHA, ValueError is raised."""

    def test_synthesize_why_strict_raises_on_hallucinated_sha(self, tmp_path: Path) -> None:
        known_sha = "cccddd1234567890abcdef1234567890abcd1234"

        commits = [_make_commit(known_sha)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        llm.complete.return_value = "Blame some SHA for this."

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            # Mock validate_citations to raise — tests that synthesize_why propagates it.
            patch(
                _PATCH_VALIDATE_CITATIONS,
                side_effect=ValueError("citation validation failed: 1 issues"),
            ) as mock_validate,
            patch(_PATCH_GET_REPO_URL, return_value=None),
            pytest.raises(ValueError, match="citation validation failed"),
        ):
            synthesize_why(target, tmp_path, llm, strict=True)

        # Confirm strict=True was forwarded to validate_citations as a keyword argument.
        mock_validate.assert_called_once()
        assert mock_validate.call_args.kwargs.get("strict") is True


# ---------------------------------------------------------------------------
# Two-pass synthesis tests
# ---------------------------------------------------------------------------

_PATCH_BUILD_GROUNDING_PROMPT = "why.synth.build_grounding_prompt"


class TestSynthesizeWhyTwoPass:
    """When two_pass=True, synthesize_why makes a second llm.complete call for grounding."""

    def _make_setup(self, tmp_path: Path):
        sha = "abc1234def5678901234567890"
        commits = [_make_commit(sha)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        return commits, f, target

    def test_synthesize_why_two_pass_makes_second_llm_call(self, tmp_path: Path) -> None:
        commits, _f, target = self._make_setup(tmp_path)
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.side_effect = [
            "first pass output",
            "## 🔍 Grounding Check\n\n| Claim | Verdict |\n|---|---|\n| foo | supported |",
        ]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
            patch(_PATCH_BUILD_GROUNDING_PROMPT, return_value=[MagicMock()]),
        ):
            synthesize_why(target, tmp_path, mock_llm, two_pass=True)

        assert mock_llm.complete.call_count == 2

    def test_synthesize_why_two_pass_appends_grounding_section(self, tmp_path: Path) -> None:
        commits, _f, target = self._make_setup(tmp_path)
        grounding_output = (
            "## 🔍 Grounding Check\n\n| Claim | Verdict |\n|---|---|\n| foo | supported |"
        )
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.side_effect = ["first pass output", grounding_output]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
            patch(_PATCH_BUILD_GROUNDING_PROMPT, return_value=[MagicMock()]),
        ):
            result = synthesize_why(target, tmp_path, mock_llm, two_pass=True)

        assert "## 🔍 Grounding Check" in result

    def test_synthesize_why_single_pass_unchanged(self, tmp_path: Path) -> None:
        commits, _f, target = self._make_setup(tmp_path)
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.return_value = "single pass output"

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            result = synthesize_why(target, tmp_path, mock_llm)

        assert mock_llm.complete.call_count == 1
        assert "## 🔍 Grounding Check" not in result

    def test_synthesize_why_two_pass_grounding_uses_first_pass_as_input(
        self, tmp_path: Path
    ) -> None:
        commits, _f, target = self._make_setup(tmp_path)
        first_pass_text = "first pass output with timeline"
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.complete.side_effect = [
            first_pass_text,
            "## 🔍 Grounding Check\n\nok",
        ]
        fake_grounding_messages = [MagicMock()]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
            patch(
                _PATCH_BUILD_GROUNDING_PROMPT, return_value=fake_grounding_messages
            ) as mock_build_grounding,
        ):
            synthesize_why(target, tmp_path, mock_llm, two_pass=True)

        mock_build_grounding.assert_called_once()
        call_args = mock_build_grounding.call_args
        assert call_args.args[0].startswith(first_pass_text)
        assert isinstance(call_args.args[1], list)
        assert len(call_args.args[1]) > 0


# ---------------------------------------------------------------------------
# brief flag pass-through tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# deep flag tests
# ---------------------------------------------------------------------------


class TestSynthesizeWhyDeepFlag:
    """deep=True bypasses select_key_commits and uses full history directly."""

    def _make_setup(self, tmp_path: Path, n: int = 5):
        commits = [_make_commit(f"deep{i:04d}" * 10) for i in range(n)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        return commits, f, target

    def test_deep_true_skips_select_key_commits(self, tmp_path: Path) -> None:
        """When deep=True, select_key_commits is NOT called and all commits are used."""
        commits, _f, target = self._make_setup(tmp_path, n=5)
        llm = MagicMock()
        llm.complete.return_value = "deep answer"
        captured: list = []

        def fake_build_prompt(t, code, cwprs, **kwargs):
            captured.extend(cwprs)
            return [MagicMock()]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT) as mock_select,
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, side_effect=fake_build_prompt),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm, deep=True)

        mock_select.assert_not_called()
        assert len(captured) == 5

    def test_deep_true_with_max_commits_caps_newest_first(self, tmp_path: Path) -> None:
        """When deep=True and max_commits=2, only the first 2 commits (newest-first) are used."""

        newer = Commit(
            sha="new0" * 10,
            author_name="Alice",
            author_email="alice@example.com",
            date=datetime(2024, 6, 1, tzinfo=UTC),
            subject="newer commit",
            body="",
            parents=(),
        )
        middle = Commit(
            sha="mid0" * 10,
            author_name="Alice",
            author_email="alice@example.com",
            date=datetime(2024, 3, 1, tzinfo=UTC),
            subject="middle commit",
            body="",
            parents=(),
        )
        older = Commit(
            sha="old0" * 10,
            author_name="Alice",
            author_email="alice@example.com",
            date=datetime(2024, 1, 1, tzinfo=UTC),
            subject="older commit",
            body="",
            parents=(),
        )
        # git returns newest-first
        commits_newest_first = [newer, middle, older]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        llm = MagicMock()
        llm.complete.return_value = "capped answer"
        captured: list = []

        def fake_build_prompt(t, code, cwprs, **kwargs):
            captured.extend(cwprs)
            return [MagicMock()]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits_newest_first),
            patch(_PATCH_SELECT) as mock_select,
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, side_effect=fake_build_prompt),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm, deep=True, max_commits=2)

        mock_select.assert_not_called()
        # Only the 2 newest commits (capped before sort), then re-sorted oldest-first
        assert len(captured) == 2
        # After oldest-first sort: middle then newer
        assert captured[0].commit.sha == middle.sha
        assert captured[1].commit.sha == newer.sha

    def test_deep_false_calls_select_key_commits(self, tmp_path: Path) -> None:
        """When deep=False (default), select_key_commits is called as before."""
        commits, _f, target = self._make_setup(tmp_path, n=5)
        key = commits[:2]
        llm = MagicMock()
        llm.complete.return_value = "normal answer"

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT, return_value=key) as mock_select,
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm, deep=False)

        mock_select.assert_called_once_with(commits, {})

    def test_deep_true_with_small_history_uses_all_commits(self, tmp_path: Path) -> None:
        """When deep=True and history has <3 commits, all commits are still used."""
        commits, _f, target = self._make_setup(tmp_path, n=2)
        llm = MagicMock()
        llm.complete.return_value = "small deep answer"
        captured: list = []

        def fake_build_prompt(t, code, cwprs, **kwargs):
            captured.extend(cwprs)
            return [MagicMock()]

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT) as mock_select,
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, side_effect=fake_build_prompt),
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm, deep=True)

        mock_select.assert_not_called()
        assert len(captured) == 2


# ---------------------------------------------------------------------------
# deep flag cost warning tests
# ---------------------------------------------------------------------------


class TestSynthesizeWhyDeepCostWarning:
    """deep=True emits a warning to stderr when estimated cost exceeds $0.50."""

    _PATCH_CLICK = "why.synth.click"

    def _make_setup(self, tmp_path: Path):
        sha = "abc1234def5678901234567890"
        commits = [_make_commit(sha)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        return commits, f, target

    def _make_large_messages(self, n_chars: int = 3_000_000):
        """Return a list with one mock message whose .content is n_chars long."""
        msg = MagicMock()
        msg.content = "x" * n_chars
        return [msg]

    def test_deep_true_large_prompt_emits_warning(self, tmp_path: Path) -> None:
        """When deep=True and estimated cost > $0.50, click.echo is called with err=True."""
        commits, _f, target = self._make_setup(tmp_path)
        llm = MagicMock()
        llm.complete.return_value = "deep answer"

        # ~3M chars / 4 = 750k tokens; 750k/1000 * 0.0008 = $0.60 — exceeds threshold
        large_messages = self._make_large_messages(n_chars=3_000_000)

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=large_messages),
            patch(_PATCH_GET_REPO_URL, return_value=None),
            patch(self._PATCH_CLICK) as mock_click,
        ):
            synthesize_why(target, tmp_path, llm, deep=True)

        # click.echo must have been called with err=True and "Warning:" in message
        echo_calls = mock_click.echo.call_args_list
        assert any(
            call.kwargs.get("err") is True and "Warning:" in call.args[0]
            for call in echo_calls
        ), f"Expected a warning echo with err=True, got: {echo_calls}"

    def test_deep_true_small_prompt_no_warning(self, tmp_path: Path) -> None:
        """When deep=True but estimated cost <= $0.50, no warning is emitted."""
        commits, _f, target = self._make_setup(tmp_path)
        llm = MagicMock()
        llm.complete.return_value = "deep answer"

        # 10 chars — cost negligible, well below threshold
        small_messages = self._make_large_messages(n_chars=10)

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=small_messages),
            patch(_PATCH_GET_REPO_URL, return_value=None),
            patch(self._PATCH_CLICK) as mock_click,
        ):
            synthesize_why(target, tmp_path, llm, deep=True)

        mock_click.echo.assert_not_called()

    def test_deep_false_large_prompt_no_warning(self, tmp_path: Path) -> None:
        """When deep=False, no cost warning is emitted even for a large prompt."""
        commits, _f, target = self._make_setup(tmp_path)
        llm = MagicMock()
        llm.complete.return_value = "normal answer"

        large_messages = self._make_large_messages(n_chars=3_000_000)

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=large_messages),
            patch(_PATCH_GET_REPO_URL, return_value=None),
            patch(self._PATCH_CLICK) as mock_click,
        ):
            synthesize_why(target, tmp_path, llm, deep=False)

        mock_click.echo.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 3: _estimate_prompt_cost uses float division
# ---------------------------------------------------------------------------


class TestEstimatePromptCostFloatDivision:
    """_estimate_prompt_cost uses float division so small prompts return > 0."""

    def test_1999_char_system_returns_positive_cost(self) -> None:
        """A 1999-char system string must produce a non-zero cost estimate."""
        from why.synth import _estimate_prompt_cost

        system = "x" * 1999
        cost = _estimate_prompt_cost(system, [])
        # With float division: (1999 / 4) / 1000 * 0.0008 = 0.0003998
        # With integer division: (1999 // 4) / 1000 * 0.0008 = (499 / 1000) * 0.0008 = 0.0003992
        # Both are > 0, but we verify we get the float-division result
        expected = (1999 / 4) / 1000 * 0.0008
        assert cost == pytest.approx(expected)
        assert cost > 0.0


# ---------------------------------------------------------------------------
# Fix 4: Cost warning includes model name
# ---------------------------------------------------------------------------


class TestSynthesizeWhyDeepCostWarningIncludesModel:
    """When the cost warning fires, it includes the model name from llm.model."""

    _PATCH_CLICK = "why.synth.click"

    def _make_setup(self, tmp_path: Path):
        sha = "abc1234def5678901234567890"
        commits = [_make_commit(sha)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        return commits, f, target

    def _make_large_messages(self, n_chars: int = 3_000_000):
        msg = MagicMock()
        msg.content = "x" * n_chars
        return [msg]

    def test_cost_warning_includes_model_name(self, tmp_path: Path) -> None:
        """When the cost warning fires, the message contains the model name."""
        commits, _f, target = self._make_setup(tmp_path)
        llm = MagicMock()
        llm.model = "test-model"
        llm.complete.return_value = "deep answer"

        large_messages = self._make_large_messages(n_chars=3_000_000)

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=large_messages),
            patch(_PATCH_GET_REPO_URL, return_value=None),
            patch(self._PATCH_CLICK) as mock_click,
        ):
            synthesize_why(target, tmp_path, llm, deep=True)

        echo_calls = mock_click.echo.call_args_list
        warning_messages = [
            call.args[0]
            for call in echo_calls
            if call.kwargs.get("err") is True and "Warning:" in call.args[0]
        ]
        assert warning_messages, "Expected a cost warning to be emitted"
        assert any("test-model" in msg for msg in warning_messages), (
            f"Expected model name in warning, got: {warning_messages}"
        )


class TestSynthesizeWhyBriefFlag:
    """brief=True/False is forwarded to build_why_prompt as a keyword argument."""

    def _make_setup(self, tmp_path: Path):
        sha = "abc1234def5678901234567890"
        commits = [_make_commit(sha)]
        f = _make_py_file(tmp_path, "foo.py", "x = 1\n")
        target = Target(file=f)
        return commits, f, target

    def test_synthesize_why_brief_passes_brief_true_to_build_prompt(
        self, tmp_path: Path
    ) -> None:
        commits, _f, target = self._make_setup(tmp_path)
        llm = MagicMock()
        llm.complete.return_value = "answer"

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]) as mock_build_prompt,
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm, brief=True)

        call_kwargs = mock_build_prompt.call_args.kwargs
        assert call_kwargs.get("brief") is True

    def test_synthesize_why_no_brief_passes_brief_false_to_build_prompt(
        self, tmp_path: Path
    ) -> None:
        commits, _f, target = self._make_setup(tmp_path)
        llm = MagicMock()
        llm.complete.return_value = "answer"

        with (
            patch(_PATCH_FILE_HISTORY, return_value=commits),
            patch(_PATCH_SELECT),
            patch(_PATCH_DIFF, return_value=""),
            patch(_PATCH_EXTRACT_CODE, return_value="code"),
            patch(_PATCH_RESOLVE_RANGE, return_value=None),
            patch(_PATCH_BUILD_PROMPT, return_value=[MagicMock()]) as mock_build_prompt,
            patch(_PATCH_GET_REPO_URL, return_value=None),
        ):
            synthesize_why(target, tmp_path, llm)

        call_kwargs = mock_build_prompt.call_args.kwargs
        assert call_kwargs.get("brief") is False
