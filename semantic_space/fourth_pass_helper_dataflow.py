"""Fourth full pass: conservative same-file helper result-to-assertion links.

Only a direct return of a product call (or one simple assignment followed by
return) is followed through at most two local helpers. This proves a static
dataflow candidate, not execution, input binding, or a semantic space unit.
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
from second_pass_source_checks import key, read_tsv
from third_pass_full_scan import definitions, invoked_helpers


HERE = Path(__file__).resolve().parent
SOURCE_REPOS = Path(os.environ.get("VERIFICATION_REFERENCE_REPOS", str(REPOS)))
FIELDS = ("snapshot", "path", "test_name", "line", "third_pass_helper_relation",
          "fourth_pass_status", "product_target", "helper_chain", "product_line",
          "caller_line", "assertion_line", "oracle_form", "evidence_scope")


def product_target(call: ast.Call, imports: dict[str, str]) -> str | None:
    name = ast.unparse(call.func)
    root, dot, suffix = name.partition(".")
    target = imports.get(root)
    return target + ("." + suffix if dot else "") if target else None


def local_helper_for_call(call: ast.Call, globals_: dict[str, ast.AST],
                          methods: dict[str, ast.AST]) -> tuple[str, ast.AST] | None:
    if isinstance(call.func, ast.Name) and call.func.id in globals_:
        return call.func.id, globals_[call.func.id]
    if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
            and call.func.value.id in {"self", "cls"} and call.func.attr in methods):
        return f"self.{call.func.attr}", methods[call.func.attr]
    return None


def simple_assignments(node: ast.AST) -> dict[str, ast.Assign]:
    result = {}
    duplicated = set()
    for part in lexical_body_nodes(node):
        if isinstance(part, ast.Assign) and len(part.targets) == 1 and isinstance(part.targets[0], ast.Name):
            name = part.targets[0].id
            if name in result:
                duplicated.add(name)
            else:
                result[name] = part
        elif isinstance(part, (ast.AnnAssign, ast.AugAssign)) and isinstance(part.target, ast.Name):
            duplicated.add(part.target.id)
    for name in duplicated:
        result.pop(name, None)
    return result


def returned_product(node: ast.AST, imports: dict[str, str],
                     globals_: dict[str, ast.AST], methods: dict[str, ast.AST],
                     depth: int = 2, seen: frozenset[int] = frozenset()) -> tuple[str, int, list[str]] | None:
    """Resolve all lexical returns to the same product result or abstain."""
    if id(node) in seen:
        return None
    seen = seen | {id(node)}
    assignments = simple_assignments(node)
    returns = [part for part in lexical_body_nodes(node) if isinstance(part, ast.Return)]
    if not returns:
        return None
    resolved = []
    for statement in returns:
        value = statement.value
        if isinstance(value, ast.Name):
            if value.id.lower().startswith(("ref", "expected", "golden")):
                return None
            assignment = assignments.get(value.id)
            if assignment is None or assignment.lineno >= statement.lineno:
                return None
            if result_reassigned(node, value.id, assignment.lineno, statement.lineno):
                return None
            if object_intervening_call(node, assignment.value, value.id, statement.lineno):
                return None
            value = assignment.value
        if not isinstance(value, ast.Call):
            return None
        target = product_target(value, imports)
        if target:
            resolved.append((target, value.lineno, []))
            continue
        helper = local_helper_for_call(value, globals_, methods)
        if helper is None or depth <= 1:
            return None
        child = returned_product(helper[1], imports, globals_, methods,
                                 depth - 1, seen)
        if child is None:
            return None
        resolved.append((child[0], child[1], [helper[0], *child[2]]))
    if len({item[0] for item in resolved}) != 1:
        return None
    return resolved[0]


def assertion_sites(node: ast.AST) -> list[ast.AST]:
    sites = []
    for part in lexical_body_nodes(node):
        if isinstance(part, ast.Assert):
            sites.append(part)
        elif isinstance(part, ast.Call):
            name = ast.unparse(part.func).split(".")[-1].lower()
            if name.startswith(("assert", "check_equal", "check_close")):
                sites.append(part)
    return sites


def uses_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(part, ast.Name) and isinstance(part.ctx, ast.Load)
               and part.id == name for part in ast.walk(node))


TRANSPARENT_CHECKS = {"len", "isinstance", "torch.equal", "torch.allclose",
                      "np.array_equal", "numpy.array_equal"}


def directly_checked(site: ast.AST, value_node: ast.AST) -> bool:
    """The value may enter a comparison, but not another product operation."""
    parents = {id(child): parent for parent in ast.walk(site)
               for child in ast.iter_child_nodes(parent)}
    current = value_node
    while id(current) in parents:
        current = parents[id(current)]
        if current is site:
            return True
        if isinstance(current, ast.Call) and ast.unparse(current.func) not in TRANSPARENT_CHECKS:
            return False
    return False


def result_reassigned(node: ast.AST, name: str, start: int, end: int) -> bool:
    for part in lexical_body_nodes(node):
        line = getattr(part, "lineno", -1)
        if not (start < line <= end):
            continue
        if isinstance(part, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = part.targets if isinstance(part, ast.Assign) else [part.target]
            if any(isinstance(child, ast.Name) and child.id == name
                   for target in targets for child in ast.walk(target)):
                return True
    return False


def object_intervening_call(node: ast.AST, original: ast.AST, name: str,
                            end: int) -> bool:
    """Abstain if a call can alter the returned object before it is checked."""
    for part in lexical_body_nodes(node):
        if not isinstance(part, ast.Call) or part is original:
            continue
        if not (getattr(original, "lineno", -1) < part.lineno < end):
            continue
        if isinstance(part.func, ast.Attribute) and uses_name(part.func.value, name):
            return True
        if any(uses_name(arg, name) for arg in part.args):
            return True
        if any(uses_name(kw.value, name) for kw in part.keywords):
            return True
    return False


def assertion_for_call(node: ast.AST, call: ast.Call) -> ast.AST | None:
    parents = {id(child): parent for parent in ast.walk(node)
               for child in ast.iter_child_nodes(parent)}
    current = call
    while id(current) in parents:
        current = parents[id(current)]
        if isinstance(current, ast.Assert):
            return current if directly_checked(current, call) else None
        if isinstance(current, ast.Call) and current in assertion_sites(node):
            return current if directly_checked(current, call) else None
    assignment = parents.get(id(call))
    if not (isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)):
        return None
    name = assignment.targets[0].id
    if name.lower().startswith(("ref", "expected", "golden")):
        return None
    for site in assertion_sites(node):
        if getattr(site, "lineno", -1) <= call.lineno or not uses_name(site, name):
            continue
        occurrences = [part for part in ast.walk(site) if isinstance(part, ast.Name)
                       and isinstance(part.ctx, ast.Load) and part.id == name]
        if not any(directly_checked(site, part) for part in occurrences):
            continue
        if (result_reassigned(node, name, call.lineno, site.lineno)
                or object_intervening_call(node, call, name, site.lineno)):
            return None
        return site
    return None


def links_for_test(node: ast.AST, globals_: dict[str, ast.AST],
                   methods: dict[str, ast.AST], imports: dict[str, str]) -> list[dict]:
    links = []
    for part in lexical_body_nodes(node):
        if not isinstance(part, ast.Call):
            continue
        helper = local_helper_for_call(part, globals_, methods)
        if helper is None:
            continue
        returned = returned_product(helper[1], imports, globals_, methods)
        if returned is None:
            continue
        assertion = assertion_for_call(node, part)
        if assertion is None:
            continue
        links.append({"product_target": returned[0], "helper_chain": [helper[0], *returned[2]],
                      "product_line": returned[1], "caller_line": part.lineno,
                      "assertion_line": assertion.lineno})
    return links


def file_links(snapshot: str, path: str) -> dict[tuple[str, int], list[dict]]:
    source = SOURCE_REPOS / SNAPSHOTS[snapshot][0] / path
    if source.suffix != ".py" or not source.is_file():
        return {}
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError, ValueError):
        return {}
    tests, globals_, classes = definitions(tree)
    imports = product_imports(tree)
    result = {}
    for (name, line), node in tests.items():
        owner = name.split("::", 1)[0] if "::" in name else ""
        result[(name, line)] = links_for_test(node, globals_, classes.get(owner, {}), imports)
    return result


def build(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict]:
    output = []
    last_file = None
    links_by_test = {}
    for row in rows:
        file_key = (row["snapshot"], row["path"])
        if file_key != last_file:
            links_by_test = file_links(*file_key)
            last_file = file_key
        links = links_by_test.get((row["test_name"], int(row["line"])), [])
        if len(links) == 1:
            link = links[0]
            status = "helper_result_reaches_assertion"
        elif len(links) > 1:
            link = {}
            status = "multiple_helper_results_need_review"
        else:
            link = {}
            status = "no_conservative_helper_result_link"
        output.append({"snapshot": row["snapshot"], "path": row["path"],
                       "test_name": row["test_name"], "line": row["line"],
                       "third_pass_helper_relation": row["helper_relation"],
                       "fourth_pass_status": status,
                       "product_target": link.get("product_target", ""),
                       "helper_chain": json.dumps(link.get("helper_chain", []), ensure_ascii=False),
                       "product_line": link.get("product_line", ""),
                       "caller_line": link.get("caller_line", ""),
                       "assertion_line": link.get("assertion_line", ""),
                       "oracle_form": row["oracle_forms"],
                       "evidence_scope": "static_same_file_dataflow_only"})
    if len(output) != len(rows) or len({key(row) for row in output}) != len(rows):
        raise ValueError("fourth pass lost or duplicated entries")
    counts = Counter(row["fourth_pass_status"] for row in output)
    summary = {"entries": len(output), "status": dict(counts),
               "from_third_pass_helper_clues": sum(
                   row["third_pass_helper_relation"] == "same_file_helper_to_product_candidate"
                   for row in output),
               "interpretation": "Direct helper return-to-assertion source links only; no execution, input/state binding or semantic unit claim."}
    return output, summary


def main() -> dict:
    missing = [name for name, (directory, _) in SNAPSHOTS.items()
               if not (SOURCE_REPOS / directory).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing source snapshots {missing} under {SOURCE_REPOS}")
    rows, summary = build(read_tsv(HERE / "third-pass-all-tests.tsv"))
    with (HERE / "fourth-pass-all-tests.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "fourth-pass-all-tests.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
