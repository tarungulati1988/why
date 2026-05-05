"""Target resolver — parses the CLI target spec into a typed, path-safe Target.

Stage: first — called from cli.main() before any git or LLM work begins.

Inputs:
    spec  — user-supplied string in the form "<path>[:<line>]".
    extra — optional positional symbol name from the CLI (mutually exclusive
            with the :<line> suffix in spec).
    repo  — repository root used to resolve relative paths (defaults to CWD).

Outputs:
    Target — frozen dataclass with an absolute file path, and optional
             line (int) and symbol (str) fields.

Invariants:
    - Resolved path is checked against repo root via relative_to(); specs that
      escape the repo (e.g. ../../etc/passwd) raise TargetError.
    - The :<line> suffix and the extra symbol argument are mutually exclusive;
      supplying both raises TargetError.
    - Line numbers must be integers >= 1; non-integer or zero/negative values
      raise TargetError.
    - No I/O beyond path resolution and existence check; no Click, no subprocess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    file: Path
    line: int | None = None
    symbol: str | None = None


class TargetError(Exception): ...


def parse_target(
    spec: str,
    extra: str | None = None,
    repo: Path | None = None,
) -> Target:
    """Resolve a user-supplied spec string into a :class:`Target`.

    ``spec`` follows the form ``<path>[:<line>]``.
    ``extra`` is an optional positional symbol name from the CLI.
    ``repo`` is the root used to resolve relative paths (defaults to CWD at call time).
    """
    repo = repo if repo is not None else Path.cwd()

    line: int | None = None
    symbol: str | None = None
    suffix: str | None = None

    if ":" in spec:
        path_part, suffix = spec.rsplit(":", 1)
    else:
        path_part = spec

    repo_root = repo.resolve()
    resolved = (repo_root / path_part).resolve()

    try:
        resolved.relative_to(repo_root)
    except ValueError as e:
        raise TargetError(f"path escapes repository root: {resolved}") from e

    if not resolved.exists():
        raise TargetError(f"file not found: {resolved}")

    if suffix is not None:
        if extra is not None:
            raise TargetError(
                f"ambiguous: both '{spec}' (contains ':') and extra='{extra}' supplied"
            )

        try:
            line = int(suffix)
        except ValueError as e:
            raise TargetError(
                f"line number must be an integer, got '{suffix}'"
            ) from e

        if line < 1:
            raise TargetError(f"line number must be >= 1, got {line}")

    elif extra is not None:
        symbol = extra

    return Target(file=resolved, line=line, symbol=symbol)
