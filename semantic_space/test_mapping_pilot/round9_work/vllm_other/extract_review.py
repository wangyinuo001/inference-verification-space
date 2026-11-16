"""Extract source-grounded review packets for the assigned vLLM tests.

This module only collects evidence. It never assigns verification properties.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent.parent
PROJECT = HERE.parents[5]
INDEX = PILOT / "full_delivery" / "all-entries.tsv"
SOURCE = PROJECT / "experiments/third_party/reference_repos/vllm-20261105"


def assigned(row: dict[str, str]) -> bool:
    path = row["path"]
    return (
        row["snapshot"] == "vllm"
        and not path.startswith(("tests/v1/", "tests/distributed/", "rust/"))
        and not path.endswith(".rs")
        and row["entry_kind"] not in ("source_test_file_unresolved", "rust_test_definition")
        and row["review_status"] not in ("property_reviewed", "outside_current_scope")
    )


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        name = call_name(node.value)
        return f"{name}.{node.attr}" if name else node.attr
    return ""


def source_segment(text: str, node: ast.AST) -> str:
    return " ".join((ast.get_source_segment(text, node) or "").strip().split())[:300]


def local_defs(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = node
    return result


def locate(tree: ast.AST, row: dict[str, str]):
    name = row["test_name"].split("::")[-1].split("[")[0]
    line = int(row["line"] or 0)
    hits = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if not hits:
        return None
    return min(hits, key=lambda node: abs(node.lineno - line))


def evidence(text: str, node: ast.AST, defs: dict, depth: int = 0, seen=()) -> dict:
    calls = []
    checks = []
    exceptions = []
    local_helpers = []
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            name = call_name(inner.func)
            if name:
                calls.append({"line": inner.lineno, "name": name, "source": source_segment(text, inner)})
            if name.endswith(("raises", "warns")):
                exceptions.append({"line": inner.lineno, "name": name, "source": source_segment(text, inner)})
            if name in defs and name not in seen and name != getattr(node, "name", ""):
                local_helpers.append(name)
        elif isinstance(inner, ast.Assert):
            checks.append({"line": inner.lineno, "kind": "assert", "source": source_segment(text, inner)})
        elif isinstance(inner, ast.Call) and call_name(inner.func).startswith(("self.assert", "torch.testing.")):
            checks.append({"line": inner.lineno, "kind": "assertion_call", "source": source_segment(text, inner)})
    for item in calls:
        name = item["name"]
        if name.startswith(("self.assert", "torch.testing.assert", "np.testing.assert", "pytest.assume")):
            checks.append({"line": item["line"], "kind": "assertion_call", "source": item["source"]})
    helpers = {}
    if depth < 2:
        for name in sorted(set(local_helpers)):
            helper = defs[name]
            helpers[name] = evidence(text, helper, defs, depth + 1, (*seen, getattr(node, "name", "")))
    return {
        "source_line": getattr(node, "lineno", None),
        "end_line": getattr(node, "end_lineno", None),
        "body": source_segment(text, node) if depth else None,
        "calls": calls,
        "checks": checks,
        "expected_exceptions": exceptions,
        "local_helpers": helpers,
    }


def run() -> None:
    rows = [row for row in csv.DictReader(INDEX.open(), delimiter="\t") if assigned(row)]
    grouped = {}
    for row in rows:
        grouped.setdefault(row["path"], []).append(row)
    stats = Counter()
    with (HERE / "source-packets.jsonl").open("w") as out:
        for path, file_rows in sorted(grouped.items()):
            source = SOURCE / path
            if not source.is_file():
                for row in file_rows:
                    out.write(json.dumps({"key": [row[k] for k in ("snapshot", "path", "test_name", "line")], "error": "source_missing"}, ensure_ascii=False) + "\n")
                stats["source_missing"] += len(file_rows)
                continue
            data = source.read_bytes()
            text = data.decode("utf-8", errors="replace")
            sha = hashlib.sha256(data).hexdigest()
            try:
                tree = ast.parse(text)
            except SyntaxError as err:
                for row in file_rows:
                    out.write(json.dumps({"key": [row[k] for k in ("snapshot", "path", "test_name", "line")], "sha256": sha, "error": f"syntax:{err.lineno}"}, ensure_ascii=False) + "\n")
                stats["syntax_error"] += len(file_rows)
                continue
            defs = local_defs(tree)
            for row in file_rows:
                node = locate(tree, row)
                packet = {
                    "key": [row[k] for k in ("snapshot", "path", "test_name", "line")],
                    "sha256": sha,
                    "review_status": row["review_status"],
                    "oracle_forms_index": row["oracle_forms"],
                    "object_candidates_index": row["object_candidates"],
                    "evidence": evidence(text, node, defs) if node else None,
                    "error": None if node else "definition_not_located",
                }
                out.write(json.dumps(packet, ensure_ascii=False) + "\n")
                stats["located" if node else "definition_not_located"] += 1
    (HERE / "source-packets-summary.json").write_text(json.dumps({"assigned": len(rows), "files": len(grouped), "status": stats}, indent=2, ensure_ascii=False))
    print(json.dumps({"assigned": len(rows), "files": len(grouped), "status": stats}))


if __name__ == "__main__":
    run()
