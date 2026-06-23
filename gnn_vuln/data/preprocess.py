"""
preprocess.py
~~~~~~~~~~~~~
Source-code normalisation utilities before graph extraction.

Pipeline (in order):
  1. strip_comments       — remove C/C++ block and line comments
  2. normalize_identifiers — rename user-defined variables/functions (opt-in)
  3. normalize_literals   — replace string/number literals with STR/NUM tokens (opt-in)
  4. remove_blank_lines   — collapse consecutive blank lines

Identifier normalisation uses tree-sitter for accurate C parsing so that
C keywords, types, and stdlib names are preserved while user-defined names
(variables, parameters, function declarations, called functions) are renamed
to VAR_0/VAR_1/... and FUNC_0/FUNC_1/... in order of first appearance.

If tree-sitter is unavailable the function silently returns the code unchanged
and logs a warning — the rest of the pipeline still runs.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# C reserved words and common stdlib names that must NOT be renamed
# ---------------------------------------------------------------------------

_C_RESERVED: frozenset[str] = frozenset({
    # C89/C99/C11 keywords
    "auto", "break", "case", "char", "const", "continue", "default", "do",
    "double", "else", "enum", "extern", "float", "for", "goto", "if",
    "inline", "int", "long", "register", "restrict", "return", "short",
    "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
    "unsigned", "void", "volatile", "while",
    # C11 additions
    "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic",
    "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
    # C99 bool/NULL
    "bool", "true", "false", "NULL", "nullptr",
    # Common POSIX / stdlib types
    "size_t", "ssize_t", "ptrdiff_t", "uintptr_t", "intptr_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "FILE", "DIR", "va_list",
    # Memory management
    "malloc", "calloc", "realloc", "free",
    # I/O
    "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf",
    "scanf", "fscanf", "sscanf",
    "fopen", "fclose", "fread", "fwrite", "fseek", "ftell", "rewind",
    "fgets", "fputs", "fgetc", "fputc", "getchar", "putchar", "gets", "puts",
    # String / memory
    "strlen", "strcpy", "strncpy", "strcmp", "strncmp", "strcat", "strncat",
    "strchr", "strrchr", "strstr", "strtok", "strtok_r",
    "memcpy", "memmove", "memset", "memcmp", "memchr",
    # Conversion
    "atoi", "atol", "atof", "strtol", "strtoul", "strtod",
    # Process
    "exit", "abort", "assert",
    # POSIX file I/O
    "open", "close", "read", "write", "lseek",
    # Memory mapping
    "mmap", "munmap", "mprotect",
    # Socket basics (common in CVE functions)
    "send", "recv", "connect", "bind", "listen", "accept",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strip_comments(code: str, lang: str = "c") -> str:
    """
    Remove comments while preserving line count.

    Block comments /* ... */ spanning N lines are replaced with N blank lines
    so every source line number stays the same after stripping.
    Line comments // ... are blanked in-place (newline kept).
    """
    if lang in ("c", "cpp", "java"):
        def _blank_block(m: re.Match) -> str:
            # keep one \n per original newline so line numbers are preserved
            newlines = m.group(0).count("\n")
            return "\n" * newlines

        code = re.sub(r"/\*.*?\*/", _blank_block, code, flags=re.DOTALL)
        code = re.sub(r"//[^\n]*", "", code)
    elif lang == "python":
        code = re.sub(r"#[^\n]*", "", code)
    return code


def normalize_identifiers(code: str) -> str:
    """
    Rename user-defined variable and function names to VAR_N / FUNC_N.

    Uses tree-sitter-c for accurate parsing so that C keywords, standard
    types, and common stdlib function names are never renamed.

    Mapping is per-function and consistent: the same name always gets the
    same token within one call. Names are numbered in order of first
    appearance:
        int vulnFunc(char *buf, int len) → int FUNC_0(char *VAR_0, int VAR_1)

    Returns the original code unchanged (with a logged warning) if
    tree-sitter is not importable.
    """
    try:
        import tree_sitter_c as tsc
        from tree_sitter import Language, Parser as TSParser
    except ImportError:
        import warnings
        warnings.warn(
            "tree-sitter-c not available — identifier normalisation skipped. "
            "Run `uv add tree-sitter tree-sitter-c` to enable it.",
            stacklevel=2,
        )
        return code

    try:
        lang_obj = Language(tsc.language())
        parser = TSParser(lang_obj)
        encoded = code.encode("utf-8", errors="replace")
        tree = parser.parse(encoded)
    except Exception:
        return code

    var_map: dict[str, str] = {}
    func_map: dict[str, str] = {}
    replacements: list[tuple[int, int, str]] = []

    def _is_call_target(node) -> bool:
        """True if this identifier is the name being called in a call_expression."""
        p = node.parent
        if p is None or p.type != "call_expression":
            return False
        first = p.children[0] if p.children else None
        if first is None:
            return False
        return first.start_byte == node.start_byte and first.end_byte == node.end_byte

    def _walk(node) -> None:
        if node.type == "identifier":
            name = encoded[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            if name not in _C_RESERVED:
                if _is_call_target(node):
                    if name not in func_map:
                        func_map[name] = f"FUNC_{len(func_map)}"
                    replacements.append((node.start_byte, node.end_byte, func_map[name]))
                else:
                    if name not in var_map:
                        var_map[name] = f"VAR_{len(var_map)}"
                    replacements.append((node.start_byte, node.end_byte, var_map[name]))
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)

    # Apply in reverse byte order so earlier offsets stay valid
    replacements.sort(key=lambda r: r[0], reverse=True)
    result = bytearray(encoded)
    for start, end, token in replacements:
        result[start:end] = token.encode("utf-8")

    return result.decode("utf-8", errors="replace")


def normalize_literals(code: str) -> str:
    """Replace string and numeric literals with STR / NUM placeholder tokens."""
    code = re.sub(r'"[^"]*"', '"STR"', code)
    code = re.sub(r"'[^']*'", "'STR'", code)
    code = re.sub(r"\b\d+(\.\d+)?\b", "NUM", code)
    return code


def remove_blank_lines(code: str) -> str:
    """Collapse consecutive blank lines to a single blank line."""
    return re.sub(r"\n{3,}", "\n\n", code)


def preprocess(
    code: str,
    lang: str = "c",
    normalize: bool = False,
    normalize_literals_flag: bool = False,
) -> str:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    code                  : raw function source code
    lang                  : 'c', 'cpp', 'java', or 'python'
    normalize             : rename user-defined identifiers (VAR_N / FUNC_N)
    normalize_literals_flag : replace string/number literals with STR / NUM
    """
    code = strip_comments(code, lang)
    if normalize:
        code = normalize_identifiers(code)
    if normalize_literals_flag:
        code = normalize_literals(code)
    # remove_blank_lines intentionally omitted — collapsing blank lines shifts
    # line numbers and breaks flaw_line / CPG node_line alignment
    return code
