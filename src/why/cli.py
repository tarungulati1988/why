"""Entry point for the why CLI."""

import click

from why import __version__


@click.command()
@click.version_option(__version__, prog_name="why")
def main() -> None:
    """Explain why code is the way it is via git history and LLM synthesis."""
    pass
