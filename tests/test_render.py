"""Tests for why.render — render_output function.

Written BEFORE implementation (TDD). Covers:
- piped mode (isatty returns False) → click.echo called, Console.print NOT called
- color=False → click.echo called even when isatty returns True
- TTY + color=True → Console.print(Markdown(text)) called, click.echo NOT called
- ANSI escape sequences stripped before output in piped mode
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rich.markdown import Markdown

from why.render import render_output

TEXT = "## Hello\n\nThis is **markdown**."


def test_piped_mode_uses_click_echo() -> None:
    """When stdout is not a TTY, raw text is echoed via click.echo."""
    with (
        patch("why.render.sys") as mock_sys,
        patch("why.render.click") as mock_click,
        patch("why.render.Console") as mock_console_cls,
    ):
        mock_sys.stdout.isatty.return_value = False

        render_output(TEXT)

        mock_click.echo.assert_called_once_with(TEXT)
        mock_console_cls.return_value.print.assert_not_called()


def test_color_false_uses_click_echo_even_when_tty() -> None:
    """When color=False, raw text is echoed regardless of TTY status."""
    with (
        patch("why.render.sys") as mock_sys,
        patch("why.render.click") as mock_click,
        patch("why.render.Console") as mock_console_cls,
    ):
        mock_sys.stdout.isatty.return_value = True

        render_output(TEXT, color=False)

        mock_click.echo.assert_called_once_with(TEXT)
        mock_console_cls.return_value.print.assert_not_called()


def test_tty_with_color_uses_rich_console() -> None:
    """When stdout is a TTY and color=True, rich Console renders markdown."""
    with (
        patch("why.render.sys") as mock_sys,
        patch("why.render.click") as mock_click,
        patch("why.render.Console") as mock_console_cls,
        patch("why.render.Markdown") as mock_markdown_cls,
    ):
        mock_sys.stdout.isatty.return_value = True
        mock_console_instance = MagicMock()
        mock_console_cls.return_value = mock_console_instance
        mock_md = MagicMock()
        mock_markdown_cls.return_value = mock_md

        render_output(TEXT)

        mock_markdown_cls.assert_called_once_with(TEXT)
        mock_console_instance.print.assert_called_once_with(mock_md)
        mock_click.echo.assert_not_called()


def test_tty_color_path() -> None:
    """TTY + color=True → Console.print receives a Markdown instance."""
    with (
        patch("why.render.sys") as mock_sys,
        patch("why.render.Console") as mock_console_cls,
    ):
        mock_sys.stdout.isatty.return_value = True

        render_output(TEXT)

        mock_console_cls.return_value.print.assert_called_once()
        call_args = mock_console_cls.return_value.print.call_args
        assert isinstance(call_args.args[0], Markdown)


def test_ansi_stripped_in_piped_mode() -> None:
    """ANSI escape sequences in LLM output are stripped before echoing."""
    dirty = "hello \x1b[31mworld\x1b[0m"
    with (
        patch("why.render.sys") as mock_sys,
        patch("why.render.click") as mock_click,
    ):
        mock_sys.stdout.isatty.return_value = False
        render_output(dirty, color=True)
    mock_click.echo.assert_called_once_with("hello world")
