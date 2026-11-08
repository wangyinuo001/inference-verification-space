"""Conservatively filter local call/oracle candidates using AST data flow.

This pass only downgrades risky mechanical candidates to review-needed.  It
does not assign verification units, infer runtime coverage, or promote entries.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOS = {
    "vllm": HERE.parent / "third_party/reference_repos/vllm-20261105",
    "sglang": HERE.parent / "third_party/reference_repos/sglang-v0.5.14-complete",
}
INPUT = HERE / "agent_contract_extract.tsv"
OUTPUT = HERE / "agent_contract_extract_v2.tsv"


def dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = dotted(node.value)
        return f"{root}.{node.attr}" if root else node.attr
    return ""


@lru_cache(maxsize=200)
def parsed(snapshot: str, path: str) -> tuple[ast.AST, dict[int, ast.AST]]:
    tree = ast.parse((REPOS[snapshot] / path).read_text(encoding="utf-8"))
    parents = {id(child): parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    return tree, parents


def ancestor(node: ast.AST, parents: dict[int, ast.AST], typ: type[ast.AST]) -> ast.AST | None:
    current = node
    while id(current) in parents:
        current = parents[id(current)]
        if isinstance(current, typ):
            return current
    return None


def contains(node: ast.AST, target: ast.AST) -> bool:
    return any(child is target for child in ast.walk(node))


def _oracle_side_risk(call: ast.Call, parents: dict[int, ast.AST]) -> str | None:
    """Recognize a call used only to construct the *expected* answer."""
    current: ast.AST = call
    while id(current) in parents:
        parent = parents[id(current)]
        if isinstance(parent, ast.Compare) and any(
            contains(comparator, call) for comparator in parent.comparators
        ):
            return "expected_value_side"
        if isinstance(parent, ast.Call) and dotted(parent.func).split(".")[-1] in {
            "assertEqual", "assertAlmostEqual", "assertIs", "assertIsNot",
        }:
            if len(parent.args) > 1 and contains(parent.args[1], call):
                return "expected_value_side"
        current = parent
    return None


def _direct_assignment_name(call: ast.Call, parents: dict[int, ast.AST]) -> str | None:
    parent = parents.get(id(call))
    if isinstance(parent, ast.Assign) and parent.value is call and len(parent.targets) == 1:
        target = parent.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(parent, ast.AnnAssign) and parent.value is call:
        return parent.target.id if isinstance(parent.target, ast.Name) else None
    return None


def _uses_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def _is_assert_call(call: ast.Call) -> bool:
    name = dotted(call.func).split(".")[-1]
    return name.startswith("assert") or name in {"raises", "raisesRegex"}


def risk_reasons(row: dict[str, str]) -> list[str]:
    """Return abstention reasons for one prior mechanical candidate."""
    if row["extraction_status"] != "mechanical_candidate":
        return []
    calls = json.loads(row["product_calls_json"])
    oracles = json.loads(row["oracle_expressions_json"])
    if len(calls) != 1 or len(oracles) != 1:
        return ["non_unique_call_or_oracle"]
    call_data, oracle_data = calls[0], oracles[0]
    tree, parents = parsed(row["snapshot"], row["path"])
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and node.lineno == int(call_data["line"])
               and ast.unparse(node) == call_data["expression"]]
    if len(matches) != 1:
        return ["call_ast_ambiguous"]
    call = matches[0]
    reasons: list[str] = []
    side = _oracle_side_risk(call, parents)
    if side:
        reasons.append(side)

    # A nested product call can be an argument to the operation under test,
    # or merely construct a receiver whose method is checked by the oracle.
    parent = parents.get(id(call))
    if isinstance(parent, ast.Attribute) and parent.value is call:
        outer = parents.get(id(parent))
        if isinstance(outer, ast.Call) and outer.func is parent:
            reasons.append("intervening_operation")
    if isinstance(parent, ast.Call) and not _is_assert_call(parent):
        reasons.append("input_to_other_operation")
    if isinstance(parent, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
        current = parent
        while id(current) in parents and isinstance(current, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            current = parents[id(current)]
        if isinstance(current, ast.Call) and not _is_assert_call(current):
            reasons.append("input_to_other_operation")

    name = _direct_assignment_name(call, parents)
    if name:
        oracle_line = int(oracle_data["line"])
        method = ancestor(call, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
        scope = method if method is not None else tree
        oracle_end = max(
            [oracle_line] + [
                int(node.end_lineno)
                for node in ast.walk(scope)
                if getattr(node, "lineno", None) == oracle_line
                and (isinstance(node, ast.Assert)
                     or isinstance(node, ast.Call) and _is_assert_call(node))
            ]
        )
        for other in ast.walk(scope):
            if not isinstance(other, ast.Call) or other is call:
                continue
            if other.lineno < call.lineno or other.lineno > oracle_end:
                continue
            if _is_assert_call(other):
                continue
            func = other.func
            # Calls on the created object change or inspect its state.
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == name:
                reasons.append("object_method_checked_or_mutated")
                break
            # Passing the object to another operation makes that operation
            # the likely subject of the assertion.  Ignore transparent local
            # adapters often used inside assertions.
            if any(_uses_name(arg, name) for arg in other.args) or any(
                _uses_name(kw.value, name) for kw in other.keywords
            ):
                if dotted(func) not in {"len", "list", "tuple", "set", "isinstance"}:
                    reasons.append("object_passed_to_other_operation")
                    break
    return list(dict.fromkeys(reasons))


def run() -> Counter[str]:
    counts: Counter[str] = Counter()
    with INPUT.open(newline="", encoding="utf-8") as src, OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=(reader.fieldnames or []) + [
            "v2_status", "v2_abstain_reasons"
        ], delimiter="\t")
        writer.writeheader()
        for row in reader:
            reasons = risk_reasons(row)
            row["v2_status"] = (
                "abstain" if reasons else row["extraction_status"]
            )
            row["v2_abstain_reasons"] = ";".join(reasons)
            writer.writerow(row)
            counts[row["v2_status"]] += 1
    return counts


if __name__ == "__main__":
    print(dict(run()))
