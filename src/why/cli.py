"""Entry point for the why CLI."""

import sys
from pathlib import Path

import click

from why import __version__
from why.git import GitError
from why.llm import LLMClient, LLMError
from why.render import render_output
from why.symbols import SymbolNotFoundError
from why.synth import synthesize_why
from why.target import TargetError, parse_target

# \b is a Click magic marker that disables paragraph re-wrapping for this block,
# preserving the indented formatting exactly as written.
_EPILOG = """\
\b
ARGUMENTS:
  TARGET  File path, optionally narrowed to a line or symbol:
            src/foo.py          whole-file analysis
            src/foo.py:42       line-scoped analysis
  SYMBOL  Symbol name to narrow analysis (optional):
            MyClass.method      method-scoped analysis
"""

@click.command(
    epilog=_EPILOG,
    no_args_is_help=True,  # bare invocation exits 2 (Click UsageError), not 0
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="why")
@click.argument("target_spec", metavar="TARGET")
@click.argument("extra", required=False, metavar="SYMBOL")
@click.option(
    "--model",
    default="llama-3.3-70b-versatile",
    show_default=True,
    help="LLM model to use.",
)
@click.option("--no-color", is_flag=True, default=False, help="Disable rich output formatting.")
def main(target_spec: str, extra: str | None, model: str, no_color: bool) -> None:
    """Explain why code is the way it is via git history and LLM synthesis."""
    cwd = Path.cwd()

    try:
        target = parse_target(target_spec, extra, cwd)
    except TargetError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    try:
        llm = LLMClient(model)
        output = synthesize_why(target, cwd, llm)
    except (LLMError, GitError, SymbolNotFoundError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    render_output(output, color=not no_color)
