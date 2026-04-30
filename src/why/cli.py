"""Entry point for the why CLI."""

import sys
from pathlib import Path

import click

from why import __version__
from why.cache import PRCache
from why.citations import CitationError
from why.git import GitError
from why.github import GitHubClient, detect_github_token
from why.llm import LLMClient, LLMError
from why.render import render_output
from why.symbols import SymbolNotFoundError
from why.synth import _get_repo_url, synthesize_why
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
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help=(
        "Enable two-pass grounding check. A second LLM call evaluates each intent "
        "claim for evidence support and appends a Grounding Check section. "
        "Adds ~1 LLM call of latency and cost."
    ),
)
@click.option(
    "--brief",
    is_flag=True,
    default=False,
    help="Emit a 3-sentence summary instead of the full narrative.",
)
@click.option(
    "--deep",
    is_flag=True,
    default=False,
    help=(
        "Include every commit in history instead of the top-scored key commits. "
        "Larger prompt — use --max-commits to cap. Warns if estimated cost > $0.50."
    ),
)
@click.option(
    "--max-commits",
    default=None,
    type=int,
    help="Hard cap on the number of commits sent to the LLM (newest first).",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable local PR metadata cache and fetch fresh from GitHub.",
)
@click.option(
    "--no-strict-citations",
    is_flag=True,
    default=False,
    help=(
        "Allow citations to SHAs/PRs not in the LLM's input context. "
        "Strict mode is auto-enabled for local providers; this flag opts out."
    ),
)
def main(
    target_spec: str,
    extra: str | None,
    model: str,
    no_color: bool,
    verify: bool,
    brief: bool,
    deep: bool,
    max_commits: int | None,
    no_cache: bool,
    no_strict_citations: bool,
) -> None:
    """Explain why code is the way it is via git history and LLM synthesis."""
    cwd = Path.cwd()

    if max_commits is not None and max_commits < 1:
        raise click.BadParameter("must be >= 1", param_hint="'--max-commits'")

    if max_commits is not None and not deep:
        raise click.UsageError("--max-commits requires --deep")

    try:
        target = parse_target(target_spec, extra, cwd)
    except TargetError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # Detect GitHub remote and initialize PR client when available.
    repo_url = _get_repo_url(cwd)
    gh_client: GitHubClient | None = None
    pr_cache: PRCache | None = None
    if repo_url is not None and "github.com" in repo_url:
        token = detect_github_token()
        gh_client = GitHubClient(repo_url, token)
        if not no_cache:
            # Derive repo_slug from URL: "https://github.com/owner/repo" → "owner__repo"
            parts = repo_url.rstrip("/").split("/")
            if len(parts) >= 2:
                slug_owner = parts[-2]
                slug_repo = parts[-1].removesuffix(".git")
                # Guard against path traversal: reject slugs with ".." or path separators
                safe = all(".." not in s and "/" not in s for s in (slug_owner, slug_repo))
                if safe:
                    repo_slug = f"{slug_owner}__{slug_repo}"
                    pr_cache = PRCache(repo_slug)

    try:
        llm = LLMClient(model)
        strict = (llm.provider == "openai-compatible") and not no_strict_citations
        output = synthesize_why(
            target, cwd, llm,
            gh=gh_client,
            pr_cache=pr_cache,
            two_pass=verify,
            brief=brief,
            deep=deep,
            max_commits=max_commits,
            strict=strict,
        )
    except CitationError as exc:
        first = exc.issues[0]
        click.echo(
            f"⚠ Local model hallucinated citation: {first.value}. "
            f"Try --no-strict-citations to allow, or switch to a larger model.",
            err=True,
        )
        sys.exit(1)
    except (LLMError, GitError, SymbolNotFoundError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    render_output(output, color=not no_color)
