"""Batch-check direct product-call to assertion links in fixed test sources.

This pass confirms only a local source dataflow relation. It deliberately does
not assign a semantic verification unit, infer runtime execution, or resolve
symbolic input values. Rejected entries remain in the review queue.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

from agent_contract_extract_v2 import (
    _direct_assignment_name,
    _is_assert_call,
    _oracle_side_risk,
    _uses_name,
    dotted,
    parsed,
)


HERE = Path(__file__).resolve().parent
INPUT = HERE / "agent_contract_extract_v2.tsv"
OUTPUT = HERE / "bulk-call-oracle-links.tsv"
SUMMARY = HERE / "bulk-call-oracle-links.json"


def test_method(tree: ast.AST, name: str, line: int) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    method_name = name.split("::")[-1]
    matches = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name == method_name and node.lineno == line]
    return matches[0] if len(matches) == 1 else None


def assertion_at(method: ast.AST, line: int) -> ast.AST | None:
    nodes = []
    for node in ast.walk(method):
        if getattr(node, "lineno", None) != line:
            continue
        if isinstance(node, ast.Assert):
            nodes.append(node)
        elif isinstance(node, ast.Call) and _is_assert_call(node) and dotted(node.func).split(".")[-1] not in {"raises", "raisesRegex"}:
            nodes.append(node)
    # An assert method Call may also be nested in an Assert. Use the outer assertion.
    outer = [n for n in nodes if isinstance(n, ast.Assert)]
    if outer:
        nodes = outer
    return nodes[0] if len(nodes) == 1 else None


def assigned_again(method: ast.AST, name: str, start: int, end: int) -> bool:
    for node in ast.walk(method):
        line = getattr(node, "lineno", -1)
        if not (start < line <= end):
            continue
        if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name) and node.target.id == name:
            return True
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(child, ast.Name) and child.id == name
                   for target in targets for child in ast.walk(target)):
                return True
    return False


def object_intervening_call(method: ast.AST, call: ast.Call, name: str, end: int) -> bool:
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or node is call or not (call.lineno < node.lineno <= end):
            continue
        if _is_assert_call(node):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == name:
            return True
        if any(_uses_name(arg, name) for arg in node.args) or any(_uses_name(kw.value, name) for kw in node.keywords):
            if dotted(func) not in {"len", "list", "tuple", "set", "isinstance"}:
                return True
    return False


def reference_value_name(name: str) -> bool:
    """Do not promote a locally computed reference as the checked result."""
    return bool(re.search(r"(?:^|_)(?:ref|reference|expected|golden)(?:_|$)", name.lower()))


def check(row: dict[str, str]) -> tuple[str, str, str]:
    if row["v2_status"] == "abstain":
        return "abstain", row["v2_abstain_reasons"], ""
    if row["v2_status"] == "mechanical_candidate":
        return "source_linked", "v2_direct_single_oracle", "literal" if all(
            item.get("literal_only") for item in json.loads(row["input_expressions_json"])
        ) else "symbolic"
    reasons = set(filter(None, row["abstain_reasons"].split(";")))
    if not reasons <= {"input_values_need_resolution", "multiple_oracles"}:
        return "needs_review", "other_risk:" + ";".join(sorted(reasons)), ""
    calls = json.loads(row["product_calls_json"])
    oracles = json.loads(row["oracle_expressions_json"])
    if len(calls) != 1 or not oracles or any(o["link_kind"] == "expected_exception" for o in oracles):
        return "needs_review", "non_unique_or_exception", ""
    try:
        tree, parents = parsed(row["snapshot"], row["path"])
    except (OSError, UnicodeError, SyntaxError, KeyError, ValueError):
        return "needs_review", "source_parse_failed", ""
    method = test_method(tree, row["test_name"], int(row["test_line"]))
    if method is None:
        return "needs_review", "test_method_unresolved", ""
    call_data = calls[0]
    matches = [node for node in ast.walk(method) if isinstance(node, ast.Call)
               and node.lineno == int(call_data["line"])
               and ast.unparse(node) == call_data.get("expression")]
    if len(matches) != 1:
        return "needs_review", "product_call_unresolved", ""
    call = matches[0]
    if _oracle_side_risk(call, parents):
        return "abstain", "expected_value_side", ""
    parent = parents.get(id(call))
    if isinstance(parent, ast.Call) and not _is_assert_call(parent):
        return "abstain", "input_to_other_operation", ""
    if isinstance(parent, ast.Attribute) and parent.value is call:
        return "abstain", "receiver_for_other_operation", ""
    assertions = [assertion_at(method, int(o["line"])) for o in oracles]
    if any(a is None for a in assertions):
        return "needs_review", "oracle_ast_unresolved", ""
    direct = all(any(child is call for child in ast.walk(a)) for a in assertions)
    name = _direct_assignment_name(call, parents)
    if not direct:
        if not name or not all(_uses_name(a, name) for a in assertions):
            return "needs_review", "result_not_used_by_each_oracle", ""
        if reference_value_name(name):
            return "abstain", "assigned_reference_value_side", ""
        end = max(int(o["line"]) for o in oracles)
        if assigned_again(method, name, call.lineno, end):
            return "needs_review", "result_reassigned", ""
        if object_intervening_call(method, call, name, end):
            return "needs_review", "result_mutated_or_passed_to_other_call", ""
    inputs = json.loads(row["input_expressions_json"])
    binding = "literal" if inputs and all(item.get("literal_only") for item in inputs) else "symbolic"
    return "source_linked", "direct_assertion_dataflow", binding


def main() -> dict:
    counts = Counter()
    with INPUT.open(newline="", encoding="utf-8") as src, OUTPUT.open("w", newline="", encoding="utf-8") as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=(reader.fieldnames or []) + ["bulk_status", "bulk_reason", "input_binding"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            status, reason, binding = check(row)
            row.update(bulk_status=status, bulk_reason=reason, input_binding=binding)
            writer.writerow(row)
            counts[status] += 1
            counts["binding:" + binding] += bool(binding)
            if status == "source_linked" and row["v2_status"] != "mechanical_candidate":
                counts["newly_linked"] += 1
    summary = {"entries": sum(counts[s] for s in ("source_linked", "needs_review", "abstain")),
               "source_linked": counts["source_linked"], "newly_linked": counts["newly_linked"],
               "needs_review": counts["needs_review"], "abstain": counts["abstain"],
               "literal_inputs": counts["binding:literal"], "symbolic_inputs": counts["binding:symbolic"],
               "scope": "direct source call-to-assertion relation only; semantic unit and runtime conditions remain unconfirmed"}
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
