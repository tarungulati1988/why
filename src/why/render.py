"""Render markdown output to the terminal, or raw text when piped."""

from __future__ import annotations

import re
import sys

import click
from rich.console import Console
from rich.markdown import Markdown

# Matches common ANSI CSI sequences (e.g. \x1b[31m) and OSC sequences.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\x1b\][^\x07]*\x07")


def render_output(text: str, color: bool = True) -> None:
    """Render markdown text to the terminal, or raw if piped/no-color."""
    text = _ANSI_ESCAPE.sub("", text)
    if not sys.stdout.isatty() or not color:
        click.echo(text)
        return
    Console(force_terminal=True).print(Markdown(text))
