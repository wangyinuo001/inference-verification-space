"""Link direct product calls to a surrounding expected-exception context.

This records a source-level negative check, not a verified runtime path or
framework-independent property. Multi-call contexts remain ambiguous.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from collections import Counter
from pathlib import Path

from ast_scope import lexical_body_nodes
from build_test_catalog import REPOS, SNAPSHOTS, product_imports
from fourth_pass_helper_dataflow import product_target
from second_pass_source_checks import key, read_tsv
from third_pass_full_scan import definitions


HERE = Path(__file__).resolve().parent
SOURCE_REPOS = Path(os.environ.get("VERIFICATION_REFERENCE_REPOS", str(REPOS)))
FIELDS = ("snapshot", "path", "test_name", "line", "status", "expected_exception",
          "product_target", "product_line", "context_line", "product_expression",
          "candidate_count", "evidence_scope")
GROUP_FIELDS = ("snapshot", "product_target", "expected_exception", "entries",
                "example_path", "example_test_name", "example_line")


def expected_exception(item: ast.withitem) -> str | None:
    context = item.context_expr
    if not isinstance(context, ast.Call):
        return None
    name = ast.unparse(context.func)
    if name not in {"pytest.raises", "raises", "self.assertRaises",
                    "self.assertRaisesRegex", "cls.assertRaises"}:
        return None
    return ast.unparse(context.args[0]) if context.args else "unspecified"


def direct_statement_call(statement: ast.stmt) -> ast.Call | None:
    """Accept only a product invocation that is the operation under the oracle."""
    if isinstance(statement, ast.Expr):
        value = statement.value
    elif isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        value = statement.value
    elif isinstance(statement, ast.AnnAssign):
        value = statement.value
    else:
        return None
    if isinstance(value, ast.Await):
        value = value.value
    return value if isinstance(value, ast.Call) else None


def exception_links(node: ast.AST, imports: dict[str, str]) -> list[dict]:
    results = []
    for part in lexical_body_nodes(node):
        if not isinstance(part, (ast.With, ast.AsyncWith)):
            continue
        exceptions = [result for item in part.items
                      if (result := expected_exception(item)) is not None]
        if not exceptions:
            continue
        # The error could come from another statement or from an enclosing
        # operation that consumes a product-constructed argument. Keep those
        # contexts for review instead of assigning the exception to the call.
        body = ast.FunctionDef(name="_context_body", args=ast.arguments(posonlyargs=[],
            args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None,
            defaults=[]), body=part.body, decorator_list=[])
        calls = [(call, target) for call in lexical_body_nodes(body)
                 if isinstance(call, ast.Call)
                 if (target := product_target(call, imports))]
        # Nested expected-exception contexts need separate review. They can
        # otherwise cause one call to be attributed to two different oracles.
        nested = any(isinstance(child, (ast.With, ast.AsyncWith))
                     for child in lexical_body_nodes(body))
        direct = direct_statement_call(part.body[0]) if len(part.body) == 1 else None
        if (len(calls) == 1 and len(exceptions) == 1 and not nested
                and direct is calls[0][0]):
            call, target = calls[0]
            results.append({"status": "direct_product_exception_link",
                            "expected_exception": exceptions[0],
                            "product_target": target,
                            "product_line": call.lineno,
                            "context_line": part.lineno,
                            "product_expression": ast.unparse(call)[:500],
                            "candidate_count": 1})
        elif calls:
            results.append({"status": "ambiguous_exception_context",
                            "expected_exception": ";".join(exceptions),
                            "product_target": "", "product_line": "",
                            "context_line": part.lineno,
                            "product_expression": "", "candidate_count": len(calls)})
    return results


def file_links(snapshot: str, path: str) -> dict[tuple[str, int], list[dict]]:
    source = SOURCE_REPOS / SNAPSHOTS[snapshot][0] / path
    if source.suffix != ".py" or not source.is_file():
        return {}
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return {}
    tests, _, _ = definitions(tree)
    imports = product_imports(tree)
    return {identity: exception_links(node, imports) for identity, node in tests.items()}


def build(rows: list[dict[str, str]]) -> tuple[list[dict], dict]:
    output = []
    last_file = None
    links_by_test = {}
    for row in rows:
        file_key = (row["snapshot"], row["path"])
        if file_key != last_file:
            links_by_test = file_links(*file_key)
            last_file = file_key
        links = links_by_test.get((row["test_name"], int(row["line"])), [])
        direct = [link for link in links if link["status"] == "direct_product_exception_link"]
        if len(direct) == 1 and len(links) == 1:
            link = direct[0]
        elif links:
            link = {"status": "multiple_or_ambiguous_exception_contexts",
                    "candidate_count": sum(item["candidate_count"] for item in links)}
        else:
            link = {"status": "no_direct_exception_link", "candidate_count": 0}
        output.append({"snapshot": row["snapshot"], "path": row["path"],
                       "test_name": row["test_name"], "line": row["line"],
                       "status": link["status"],
                       "expected_exception": link.get("expected_exception", ""),
                       "product_target": link.get("product_target", ""),
                       "product_line": link.get("product_line", ""),
                       "context_line": link.get("context_line", ""),
                       "product_expression": link.get("product_expression", ""),
                       "candidate_count": link["candidate_count"],
                       "evidence_scope": "static_direct_call_inside_expected_exception"})
    if len(output) != len(rows) or len({key(row) for row in output}) != len(rows):
        raise ValueError("fifth pass lost or duplicated entries")
    summary = {"entries": len(output), "status": dict(Counter(row["status"] for row in output)),
               "interpretation": "Direct product call inside one expected-exception context; no execution or semantic unit claim."}
    return output, summary


def main() -> dict:
    missing = [name for name, (directory, _) in SNAPSHOTS.items()
               if not (SOURCE_REPOS / directory).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing source snapshots {missing} under {SOURCE_REPOS}")
    rows, summary = build(read_tsv(HERE / "third-pass-all-tests.tsv"))
    with (HERE / "fifth-pass-all-tests.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        if row["status"] != "direct_product_exception_link":
            continue
        group_key = (row["snapshot"], row["product_target"], row["expected_exception"])
        groups.setdefault(group_key, []).append(row)
    with (HERE / "fifth-pass-review-groups.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GROUP_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for group_key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
            first = members[0]
            writer.writerow(dict(zip(GROUP_FIELDS, (*group_key, len(members),
                                                   first["path"], first["test_name"],
                                                   first["line"]))))
    summary["review_groups"] = len(groups)
    summary["entries_in_multi_member_groups"] = sum(len(members) for members in groups.values()
                                                   if len(members) > 1)
    (HERE / "fifth-pass-all-tests.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
