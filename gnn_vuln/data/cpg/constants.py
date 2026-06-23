"""CPG constants: edge types, node types, dangerous APIs, enum mappings."""
from __future__ import annotations

import logging
import re

from gnn_vuln.data.node_embedder import NODE_TYPES, NODE_TYPE_TO_IDX

_log = logging.getLogger(__name__)
_unknown_enum_seen: set[str] = set()

# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------
EDGE_TYPES = [
    "ALIAS_OF",        # type alias
    "ARGUMENT",        # call → arguments
    "AST",             # abstract syntax tree
    "BINDS",           # type binding
    "CALL",            # function calls
    "CAPTURE",         # closure capture
    "CATCH_BODY",      # try → catch body
    "CDG",             # control dependence
    "CFG",             # control flow
    "CONDITION",       # control structure → condition expr
    "CONTAINS",        # containment
    "DOMINATE",        # dominance
    "EVAL_TYPE",       # type evaluation
    "FALSE_BODY",      # if → false branch
    "FINALLY_BODY",    # try → finally body
    "FOR_BODY",        # for → body
    "FOR_INIT",        # for → init
    "FOR_UPDATE",      # for → update
    "IMPORTS",         # import/include (Java/Python)
    "INHERITS_FROM",   # class inheritance
    "IS_CALL_FOR_IMPORT", # import call
    "PARAMETER_LINK",  # call arg → formal param
    "POST_DOMINATE",   # post-dominance
    "REACHING_DEF",    # data dependence (DDG)
    "RECEIVER",        # method call receiver
    "REF",             # identifier → declaration
    "SOURCE_FILE",     # source file link
    "TAGGED_BY",       # tag annotation
    "TRUE_BODY",       # if → true branch
    "TRY_BODY",        # try → body
]
EDGE_TYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPES)}

# ---------------------------------------------------------------------------
# Node feature constants
# ---------------------------------------------------------------------------
_NUM_NODE_TYPES = len(NODE_TYPES)
_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"

DANGEROUS_APIS = frozenset({
    "malloc", "free", "realloc", "calloc",
    "memcpy", "memmove", "memset", "memcmp",
    "strcpy", "strncpy", "strcat", "strncat", "strcmp", "strlen",
    "sprintf", "snprintf", "vsprintf", "vsnprintf",
    "gets", "fgets", "scanf", "sscanf",
    "read", "write", "recv", "send",
    "open", "fopen", "popen",
    "system", "execl", "execv", "execve",
    "alloca",
})

_TOKEN_RE = re.compile(r'\b\w+\b')

# ---------------------------------------------------------------------------
# Enum mappings for node attributes
# ---------------------------------------------------------------------------
_CST_TO_IDX = {
    "IF": 1, "ELSE": 2, "WHILE": 3, "FOR": 4,
    "SWITCH": 5, "BREAK": 6, "CONTINUE": 7, "GOTO": 8,
    "TRY": 9, "CATCH": 10, "FINALLY": 11,
    "DO": 12, "THROW": 13,  # Java/JS
}
_EVAL_TO_IDX  = {"BY_VALUE": 1, "BY_REFERENCE": 2, "BY_SHARING": 3}
_DISPATCH_IDX = {"STATIC_DISPATCH": 1, "DYNAMIC_DISPATCH": 2, "INLINED": 3}
_DISPATCH_VALS = set(_DISPATCH_IDX)

_INTEGER_TYPES = frozenset({
    "int", "long", "short", "char",
    "unsigned int", "unsigned long", "unsigned short", "unsigned char",
    "size_t", "ssize_t", "ptrdiff_t", "intptr_t", "uintptr_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
})
_MAX_ARG_IDX = 16.0


def _check_enum(field: str, value: str, known: set | dict) -> None:
    """Warn once when an unknown enum value is encountered."""
    known_set = set(known) if not isinstance(known, set) else known
    if value and value not in known_set:
        key = f"{field}:{value}"
        if key not in _unknown_enum_seen:
            _unknown_enum_seen.add(key)
            _log.warning(
                "Unknown enum value for %s: %r — not in known set %s. "
                "Add it to the mapping in cpg/constants.py.",
                field, value, sorted(known_set),
            )
