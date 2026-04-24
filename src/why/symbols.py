"""Symbol range resolver using tree-sitter AST parsing."""

from __future__ import annotations

import logging
from pathlib import Path

import tree_sitter_go as _tsgo
import tree_sitter_python as _tspython
from tree_sitter import Language, Parser

_log = logging.getLogger(__name__)

# Build language objects once at module level to avoid repeated construction overhead.
_PY_LANGUAGE = Language(_tspython.language())
_GO_LANGUAGE = Language(_tsgo.language())

_LANGUAGES: dict[str, Language] = {
    "python": _PY_LANGUAGE,
    "go": _GO_LANGUAGE,
}

# Maps file extensions to language names used in _LANGUAGES.
EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".go": "go",
}

# Tree-sitter query that captures named Python definitions (functions and classes).
# Note: async functions are represented as function_definition nodes in this grammar version.
# Each match yields a "name" capture (the identifier node) and a "definition" capture (the full node).
_PYTHON_QUERY = """
[
  (function_definition name: (identifier) @name) @definition
  (class_definition name: (identifier) @name) @definition
]
"""

# Tree-sitter query for Go top-level named definitions.
# type_declaration wraps one or more type_spec nodes; matching at that level gives us
# the full declaration range (including the braces for structs/interfaces) rather than
# just the inner type_spec.  method_spec nodes (interface stubs) are intentionally
# excluded — only function_declaration and method_declaration are captured at top level.
_GO_QUERY = """
[
  (function_declaration name: (identifier) @name) @definition
  (method_declaration name: (field_identifier) @name) @definition
  (type_declaration (type_spec name: (type_identifier) @name)) @definition
]
"""

_PARSERS: dict[str, Parser] = {
    "python": Parser(_PY_LANGUAGE),
    "go": Parser(_GO_LANGUAGE),
}

_QUERIES: dict[str, str] = {
    "python": _PYTHON_QUERY,
    "go": _GO_QUERY,
}


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

    language = _LANGUAGES[lang_name]
    parser = _PARSERS[lang_name]
    try:
        source = file.read_bytes()
    except OSError as exc:
        raise SymbolNotFoundError(f"cannot read {file}: {exc}") from exc
    tree = parser.parse(source)
    query_string = _QUERIES[lang_name]

    query = language.query(query_string)
    matches = query.matches(tree.root_node)

    # Collect all definition nodes whose name capture matches the requested symbol.
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
