"""Target resolver for the why CLI.

A Target is the thing the user points why at: a file, optionally narrowed
to a line number or a symbol name.  This module is a pure domain module —
no I/O beyond resolving the path, no Click, no subprocess.
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
    except ValueError:
        raise TargetError(f"path escapes repository root: {resolved}")

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
