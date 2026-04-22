"""Tests for the Target resolver (src/why/target.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from why.target import Target, TargetError, parse_target


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a minimal repo layout with a real file for path resolution."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "foo.py").write_text("# placeholder\n")
    return tmp_path


class TestParseTargetLineNumber:
    def test_colon_with_integer_suffix_sets_line(self, tmp_repo: Path) -> None:
        t = parse_target("src/foo.py:45", repo=tmp_repo)
        assert t.line == 45
        assert t.symbol is None
        assert t.file == (tmp_repo / "src/foo.py").resolve()

    def test_colon_with_non_integer_suffix_raises(self, tmp_repo: Path) -> None:
        with pytest.raises(TargetError, match="must be an integer"):
            parse_target("src/foo.py:notanumber", repo=tmp_repo)

    def test_line_zero_raises(self, tmp_repo: Path) -> None:
        with pytest.raises(TargetError, match=">="):
            parse_target("src/foo.py:0", repo=tmp_repo)

    def test_negative_line_raises(self, tmp_repo: Path) -> None:
        with pytest.raises(TargetError, match=">="):
            parse_target("src/foo.py:-1", repo=tmp_repo)


class TestParseTargetSymbol:
    def test_extra_arg_sets_symbol(self, tmp_repo: Path) -> None:
        t = parse_target("src/foo.py", "my_func", repo=tmp_repo)
        assert t.symbol == "my_func"
        assert t.line is None
        assert t.file == (tmp_repo / "src/foo.py").resolve()


class TestParseTargetNoLineNoSymbol:
    def test_bare_path_returns_target_with_no_line_no_symbol(self, tmp_repo: Path) -> None:
        t = parse_target("src/foo.py", repo=tmp_repo)
        assert t.line is None
        assert t.symbol is None
        assert t.file == (tmp_repo / "src/foo.py").resolve()


class TestParseTargetAmbiguous:
    def test_colon_spec_with_extra_raises_target_error(self, tmp_repo: Path) -> None:
        with pytest.raises(TargetError, match="ambiguous"):
            parse_target("src/foo.py:45", "my_func", repo=tmp_repo)


class TestParseTargetMissingFile:
    def test_nonexistent_file_raises_target_error(self, tmp_repo: Path) -> None:
        with pytest.raises(TargetError, match="file not found"):
            parse_target("missing.py", repo=tmp_repo)


class TestParseTargetTraversal:
    def test_dotdot_traversal_raises(self, tmp_repo: Path) -> None:
        with pytest.raises(TargetError, match="escapes repository root"):
            parse_target("../../etc/passwd", repo=tmp_repo)
