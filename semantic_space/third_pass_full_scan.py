"""Full-inventory third pass: assertion forms and same-file helper-call clues.

Every first/second-pass row is retained. Assertion syntax and helper reachability
are evidence for review, not proof that a request path ran or a property holds.
"""

from __future__ import annotations

import ast
import csv
import json
import os
from collections import Counter
from pathlib import Path

from ast_scope import lexical_body_nodes
from build_test_catalog import REPOS, SNAPSHOTS, product_imports, static_test_evidence
from second_pass_source_checks import key, read_tsv


HERE = Path(__file__).resolve().parent
SOURCE_REPOS = Path(os.environ.get("VERIFICATION_REFERENCE_REPOS", str(REPOS)))
FIELDS = ("snapshot", "path", "test_name", "line", "candidate_area",
          "source_review_verdict", "reviewed_area", "direct_source_link",
          "oracle_link_status", "oracle_forms", "oracle_expression_count",
          "helper_names", "helper_product_targets", "helper_relation",
          "scan_status", "next_review_action")


def oracle_form(expression: str, kind: str = "") -> str:
    """Classify the *form* of a check, without assigning domain semantics."""
    value = expression.lower()
    if kind == "expected_exception" or any(x in value for x in
            ("pytest.raises(", ".assertraises(", ".assertraisesregex(")):
        return "expected_exception"
    if any(x in value for x in ("assert_close(", "allclose(", "assert_allclose(",
                                    "assertalmostequal(", "isclose(")):
        return "numerical_closeness"
    if any(x in value for x in ("torch.equal(", "assert_array_equal(",
                                    "assert_tensor_equal(")):
        return "exact_tensor_equality"
    if "status_code" in value or "httpstatus" in value or ".status ==" in value:
        return "protocol_status"
    if any(x in value for x in (".shape", ".size(", ".ndim", ".numel(")):
        return "shape_or_size"
    if ".dtype" in value:
        return "dtype"
    if any(x in value for x in (".stride(", ".is_contiguous(", ".layout")):
        return "memory_layout"
    if ".device" in value:
        return "device"
    if "len(" in value:
        return "cardinality"
    if any(x in value for x in ("assert_equal(", "assertequal(", "assertis(",
                                    "assert ==", " == ", " != ", " is ", " is not ")):
        return "value_comparison"
    if value.startswith("assert ") or ".asserttrue(" in value or ".assertfalse(" in value:
        return "boolean_predicate"
    return "other_assertion"


def source_items(catalog: dict) -> dict[tuple[str, ...], dict]:
    items = {}
    for snapshot in catalog["snapshots"]:
        for file in snapshot["files"]:
            for item in file["items"]:
                items[(snapshot["id"], file["path"], item["name"], str(item["line"]))] = item
    return items


def definitions(tree: ast.Module) -> tuple[dict[tuple[str, int], ast.AST],
                                           dict[str, ast.AST], dict[str, dict[str, ast.AST]]]:
    tests = {}
    global_helpers = {}
    class_helpers = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            global_helpers[node.name] = node
            tests[(node.name, node.lineno)] = node
        elif isinstance(node, ast.ClassDef):
            methods = {method.name: method for method in node.body
                       if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))}
            class_helpers[node.name] = methods
            for method in methods.values():
                tests[(f"{node.name}::{method.name}", method.lineno)] = method
    return tests, global_helpers, class_helpers


def invoked_helpers(node: ast.AST, global_helpers: dict[str, ast.AST],
                    methods: dict[str, ast.AST]) -> list[tuple[str, ast.AST]]:
    found = {}
    for part in lexical_body_nodes(node):
        if not isinstance(part, ast.Call):
            continue
        if isinstance(part.func, ast.Name) and part.func.id in global_helpers:
            found[part.func.id] = global_helpers[part.func.id]
        elif (isinstance(part.func, ast.Attribute) and
              isinstance(part.func.value, ast.Name) and
              part.func.value.id in {"self", "cls"} and
              part.func.attr in methods):
            found[f"self.{part.func.attr}"] = methods[part.func.attr]
    return list(found.items())


def helper_clues(node: ast.AST, global_helpers: dict[str, ast.AST],
                 methods: dict[str, ast.AST], imports: dict[str, str],
                 max_depth: int = 2) -> tuple[list[str], list[str]]:
    queue = [(name, helper, 1) for name, helper in invoked_helpers(node, global_helpers, methods)]
    visited = set()
    names, targets = [], set()
    while queue:
        name, helper, depth = queue.pop(0)
        if id(helper) in visited:
            continue
        visited.add(id(helper))
        names.append(name)
        calls, _, _, _ = static_test_evidence(helper, imports)
        targets.update(call["target"] for call in calls)
        if depth < max_depth:
            queue.extend((child, child_node, depth + 1)
                         for child, child_node in invoked_helpers(helper, global_helpers, methods))
    return sorted(names), sorted(targets)


def file_clues(snapshot: str, path: str) -> tuple[dict[tuple[str, int], tuple[list[str], list[str]]], str]:
    root = SOURCE_REPOS / SNAPSHOTS[snapshot][0]
    source = root / path
    if not source.is_file() or source.suffix != ".py":
        return {}, "source_unavailable_or_non_python"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return {}, "source_parse_failed"
    tests, globals_, classes = definitions(tree)
    imports = product_imports(tree)
    found = {}
    for (name, line), node in tests.items():
        class_name = name.split("::", 1)[0] if "::" in name else ""
        found[(name, line)] = helper_clues(node, globals_, classes.get(class_name, {}), imports)
    return found, "source_parsed"


def make_record(row: dict[str, str], catalog_item: dict | None,
                bulk: dict | None, helper: tuple[list[str], list[str]],
                source_status: str) -> dict[str, str]:
    linked = row["source_link_status"] == "call_result_to_assertion"
    if linked and bulk:
        oracles = json.loads(bulk["oracle_expressions_json"] or "[]")
        link_status = "linked_to_product_call"
    elif catalog_item:
        oracles = catalog_item.get("source_assertions", [])
        link_status = "unbound_source_assertions" if oracles else "no_source_assertion"
    else:
        oracles = []
        link_status = "no_source_assertion"
    forms = sorted({oracle_form(item.get("expression", ""), item.get("kind", ""))
                    for item in oracles})
    helpers, targets = helper
    relation = "same_file_helper_to_product_candidate" if targets else (
        "same_file_helper_without_product_call" if helpers else "no_same_file_helper")
    if row["source_review_verdict"]:
        action = "use_source_review_for_unit_mapping"
    elif linked:
        action = "check_asserted_property_and_input_conditions"
    elif targets:
        action = "check_helper_call_and_assertion_dataflow"
    elif oracles:
        action = "resolve_tested_operation_for_assertion"
    else:
        action = "find_runtime_or_non_python_oracle"
    return {"snapshot": row["snapshot"], "path": row["path"],
            "test_name": row["test_name"], "line": row["line"],
            "candidate_area": row["candidate_area"],
            "source_review_verdict": row["source_review_verdict"],
            "reviewed_area": row["reviewed_area"],
            "direct_source_link": "yes" if linked else "no",
            "oracle_link_status": link_status, "oracle_forms": ",".join(forms),
            "oracle_expression_count": str(len(oracles)),
            "helper_names": json.dumps(helpers[:16], ensure_ascii=False),
            "helper_product_targets": json.dumps(targets[:16], ensure_ascii=False),
            "helper_relation": relation, "scan_status": source_status,
            "next_review_action": action}


def build(rows: list[dict[str, str]], catalog: dict,
          bulk_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    catalog_items = source_items(catalog)
    bulk = {key(row, "test_line"): row for row in bulk_rows}
    last_file_key = None
    clues, source_status = {}, "source_unavailable_or_non_python"
    output = []
    for row in rows:
        file_key = (row["snapshot"], row["path"])
        if file_key != last_file_key:
            clues, source_status = file_clues(*file_key)
            last_file_key = file_key
        helper = clues.get((row["test_name"], int(row["line"])), ([], []))
        item_key = key(row)
        output.append(make_record(row, catalog_items.get(item_key), bulk.get(item_key),
                                  helper, source_status))
    if len(output) != len(rows) or len({key(row) for row in output}) != len(rows):
        raise ValueError("third pass lost or duplicated test entries")
    counts = Counter(row["oracle_link_status"] for row in output)
    helper_count = sum(row["helper_relation"] == "same_file_helper_to_product_candidate"
                       for row in output)
    summary = {"entries": len(output), "oracle_link_status": dict(counts),
               "entries_with_same_file_helper_product_clue": helper_count,
               "entries_with_oracle_form": sum(bool(row["oracle_forms"]) for row in output),
               "entries_with_source_review": sum(bool(row["source_review_verdict"]) for row in output),
               "interpretation": "Oracle forms are syntax only; helper links show possible source calls, not execution or verified properties."}
    return output, summary


def main() -> dict:
    missing = [name for name, (directory, _) in SNAPSHOTS.items()
               if not (SOURCE_REPOS / directory).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"missing fixed source snapshots for {missing} under {SOURCE_REPOS}; "
            "set VERIFICATION_REFERENCE_REPOS to their parent directory")
    rows, summary = build(
        read_tsv(HERE / "second-pass-all-tests.tsv"),
        json.loads((HERE / "test_catalog.json").read_text(encoding="utf-8")),
        read_tsv(HERE / "bulk-call-oracle-links.tsv"))
    with (HERE / "third-pass-all-tests.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "third-pass-all-tests.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
