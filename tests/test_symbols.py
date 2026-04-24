"""Tests for src/why/symbols.py."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from why.symbols import SymbolNotFoundError, find_symbol_range

FIXTURES = Path(__file__).parent / "fixtures" / "symbols"


class TestFindSymbolRangePython:
    def test_standalone_function(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.py", "standalone_function")
        assert start == 1
        assert end == 2

    def test_async_function(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.py", "async_function")
        assert start == 5
        assert end == 6

    def test_class(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.py", "MyClass")
        assert start == 9
        assert end == 14

    def test_method(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.py", "method_one")
        assert start == 10
        assert end == 11

    def test_symbol_not_found_raises(self) -> None:
        with pytest.raises(SymbolNotFoundError, match="not found"):
            find_symbol_range(FIXTURES / "sample.py", "nonexistent")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "foo.rb"
        f.write_text("puts 'hello'\n")
        with pytest.raises(SymbolNotFoundError, match="unsupported"):
            find_symbol_range(f, "foo")

    def test_duplicate_name_returns_first_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="why.symbols"):
            start, end = find_symbol_range(FIXTURES / "sample.py", "duplicate_name")
        assert start == 17
        assert end == 18
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.WARNING
        assert "matched" in caplog.records[0].message
        assert "returning first" in caplog.records[0].message


class TestFindSymbolRangeGo:
    def test_standalone_func(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.go", "StandaloneFunc")
        assert start == 4
        assert end == 6

    def test_struct(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.go", "Server")
        assert start == 9
        assert end == 11

    def test_method(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.go", "Start")
        assert start == 14
        assert end == 16

    def test_interface_type(self) -> None:
        # Returns range of the Handler interface block, not the method stub inside it.
        start, end = find_symbol_range(FIXTURES / "sample.go", "Handler")
        assert start == 19
        assert end == 22

    def test_interface_method_stub_not_matched(self) -> None:
        # ServeHTTP lives inside the interface as a method_spec — must NOT be found.
        with pytest.raises(SymbolNotFoundError, match="not found"):
            find_symbol_range(FIXTURES / "sample.go", "ServeHTTP")

    def test_type_alias(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.go", "Option")
        assert start == 25
        assert end == 25

    def test_functional_option_outer_func(self) -> None:
        start, end = find_symbol_range(FIXTURES / "sample.go", "WithTimeout")
        assert start == 28
        assert end == 32


class TestFindSymbolRangeEdgeCases:
    @pytest.mark.parametrize("suffix", [".py", ".go"])
    def test_empty_file_raises(self, tmp_path: Path, suffix: str) -> None:
        f = tmp_path / f"empty{suffix}"
        f.write_bytes(b"")
        with pytest.raises(SymbolNotFoundError, match="not found"):
            find_symbol_range(f, "anything")

    @pytest.mark.skipif(os.getuid() == 0, reason="root bypasses file permissions")
    def test_unreadable_file_raises_symbol_not_found(self, tmp_path: Path) -> None:
        f = tmp_path / "locked.py"
        f.write_text("def foo(): pass\n")
        f.chmod(0o000)
        try:
            with pytest.raises(SymbolNotFoundError, match="cannot read"):
                find_symbol_range(f, "foo")
        finally:
            f.chmod(0o644)
