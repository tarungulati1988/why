"""Symbol range resolver using tree-sitter AST parsing."""

from __future__ import annotations

import logging
from pathlib import Path

import tree_sitter_go as _tsgo
import tree_sitter_python as _tspython
from tree_sitter import Language, Node, Parser, Query

_log = logging.getLogger(__name__)

_PY_LANGUAGE = Language(_tspython.language())
_GO_LANGUAGE = Language(_tsgo.language())

EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".go": "go",
}

# Tree-sitter query for Python: captures named functions and classes.
# async functions use function_definition in this grammar version (no separate async node).
_PYTHON_QUERY_STRING = """
[
  (function_definition name: (identifier) @name) @definition
  (class_definition name: (identifier) @name) @definition
]
"""

# method_spec nodes (interface stubs) are intentionally excluded — only
# function_declaration and method_declaration match concrete Go definitions.
_GO_QUERY_STRING = """
[
  (function_declaration name: (identifier) @name) @definition
  (method_declaration name: (field_identifier) @name) @definition
  (type_declaration (type_spec name: (type_identifier) @name)) @definition
]
"""

# Pre-compiled Query objects and cached Parsers — constructed once at module load.
_PARSERS: dict[str, Parser] = {
    "python": Parser(_PY_LANGUAGE),
    "go": Parser(_GO_LANGUAGE),
}

_QUERIES: dict[str, Query] = {
    "python": Query(_PY_LANGUAGE, _PYTHON_QUERY_STRING),
    "go": Query(_GO_LANGUAGE, _GO_QUERY_STRING),
}

# tree-sitter 0.25+ replaced query.matches(node) with QueryCursor(query).matches(node).
# Both APIs return list[tuple[int, dict[str, list[Node]]]] so the call-site is identical.
try:
    from tree_sitter import QueryCursor as _QueryCursor

    def _exec_query(
        query: Query, node: Node
    ) -> list[tuple[int, dict[str, list[Node]]]]:
        return _QueryCursor(query).matches(node)

except ImportError:

    def _exec_query(
        query: Query, node: Node
    ) -> list[tuple[int, dict[str, list[Node]]]]:
        return query.matches(node)  # type: ignore[attr-defined, no-any-return]


class SymbolNotFoundError(Exception):
    """Raised when a symbol isn't found or the file extension is unsupported."""


def find_symbol_range(file: Path, symbol: str) -> tuple[int, int]:
    """Return the 1-indexed (line_start, line_end) range for ``symbol`` in ``file``.

    Uses tree-sitter to parse the file's AST and locate the named definition.
    Raises SymbolNotFoundError if the extension is unsupported or the symbol
    is not found.
    """
    lang_name = EXTENSION_TO_LANG.get(file.suffix)
    if lang_name is None:
        raise SymbolNotFoundError(f"unsupported language: {file.suffix}")

    parser = _PARSERS[lang_name]
    try:
        source = file.read_bytes()
    except OSError as exc:
        raise SymbolNotFoundError(f"cannot read {file}: {exc}") from exc
    tree = parser.parse(source)

    matches = _exec_query(_QUERIES[lang_name], tree.root_node)

    found = []
    for _pattern_index, captures in matches:
        name_nodes = captures.get("name", [])
        def_nodes = captures.get("definition", [])
        if not name_nodes or not def_nodes:
            continue
        if name_nodes[0].text == symbol.encode():
            found.append(def_nodes[0])

    if not found:
        raise SymbolNotFoundError(f"symbol '{symbol}' not found in {file}")

    if len(found) > 1:
        _log.warning(
            "symbol %r matched %d definitions in %s; returning first",
            symbol,
            len(found),
            file,
        )

    # tree-sitter uses 0-indexed rows; add 1 to convert to 1-indexed line numbers.
    definition = found[0]
    return (definition.start_point[0] + 1, definition.end_point[0] + 1)
