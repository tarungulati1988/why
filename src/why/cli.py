"""Entry point for the why CLI."""

import sys
from pathlib import Path

import click

from why import __version__
from why.git import GitError
from why.llm import LLMClient, LLMError
from why.synth import synthesize_why
from why.target import TargetError, parse_target


@click.command()
@click.version_option(__version__, prog_name="why")
@click.argument("target_spec", metavar="TARGET")
@click.argument("extra", required=False, metavar="SYMBOL")
@click.option(
    "--model",
    default="llama-3.3-70b-versatile",
    show_default=True,
    help="LLM model to use.",
)
def main(target_spec: str, extra: str | None, model: str) -> None:
    """Explain why code is the way it is via git history and LLM synthesis.

    TARGET is a file path, optionally narrowed to a line number (src/foo.py:42)
    or followed by a SYMBOL name (src/foo.py MyClass.method).
    """
    cwd = Path.cwd()

    try:
        target = parse_target(target_spec, extra, cwd)
    except TargetError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    try:
        llm = LLMClient(model)
        output = synthesize_why(target, cwd, llm)
    except (LLMError, GitError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(output)
