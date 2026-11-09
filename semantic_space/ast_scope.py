"""AST traversal that keeps a test body separate from nested definitions."""

from __future__ import annotations

import ast
from collections.abc import Iterator


NESTED_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def lexical_body_nodes(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """Yield expressions/statements in a function body, without entering local definitions.

    Source presence in this iterator still does not mean a conditional branch ran.
    """
    stack = list(reversed(node.body))
    while stack:
        part = stack.pop()
        if isinstance(part, NESTED_DEFINITION):
            continue
        yield part
        stack.extend(reversed(list(ast.iter_child_nodes(part))))


def local_definitions(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Yield locally defined functions without descending into their bodies."""
    stack = list(reversed(node.body))
    while stack:
        part = stack.pop()
        if isinstance(part, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield part
            continue
        if isinstance(part, (ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(part))))
