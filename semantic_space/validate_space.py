"""Check the provenance and internal consistency of the source-grounded space seed."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
ALLOWED_RELATIONS = {"refines", "requires", "consumes_result_of"}


def rule_applies(rule: dict, conditions: dict) -> bool:
    for dimension_id, expected in rule["when"].items():
        actual = conditions.get(dimension_id)
        if expected == "present" and isinstance(actual, dict):
            continue
        if actual != expected:
            return False
    return True


def validate() -> dict[str, int]:
    data = json.loads((HERE / "space_seed.json").read_text(encoding="utf-8"))
    roots: dict[str, Path] = {}
    for framework, snapshot in data["snapshots"].items():
        root = (HERE / snapshot["directory"]).resolve()
        assert root.is_dir(), f"Missing {framework} snapshot: {root}"
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        assert actual == snapshot["commit"], f"Changed {framework} snapshot: {actual}"
        roots[framework] = root

    dimension_items = {item["id"]: item for item in data["condition_dimensions"]}
    dimensions = set(dimension_items)
    assert len(dimensions) == len(data["condition_dimensions"])
    for item in dimension_items.values():
        if item.get("type") == "structured":
            assert item["fields"] and len(item["fields"]) == len(set(item["fields"]))
        else:
            assert item["observed_values"], item["id"]
    ids = {unit["id"] for unit in data["units"]}
    assert len(ids) == len(data["units"]), "Duplicate unit ID"
    by_id = {unit["id"]: unit for unit in data["units"]}
    refs = 0
    for unit in data["units"]:
        assert unit["kind"] in {"obligation", "composite"}
        assert set(unit["conditions"]) <= dimensions, unit["id"]
        assert unit["property"] and unit["observable"] and unit["oracle"]
        if unit["kind"] == "obligation":
            scope = set(unit.get("framework_scope", roots))
            assert scope and scope <= set(roots), unit["id"]
            assert set(unit["evidence"]) == scope, unit["id"]
            assert all(
                any(ref["kind"] == "source" for ref in unit["evidence"][framework])
                for framework in scope
            ), f"Missing implementation mapping: {unit['id']}"
        for framework, evidence in unit["evidence"].items():
            assert framework in roots, (unit["id"], framework)
            for ref in evidence:
                path = roots[framework] / ref["path"]
                assert path.is_file(), f"Missing evidence: {path}"
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                assert 1 <= ref["line"] <= len(lines), (path, ref["line"])
                assert lines[ref["line"] - 1].strip(), (path, ref["line"])
                assert ref["kind"] in {"source", "existing_test", "existing_oracle"}
                if "runtime_guard" in ref:
                    assert ref["kind"] == "source" and isinstance(ref["runtime_guard"], dict)
                    assert ref["runtime_guard"], (unit["id"], framework)
                refs += 1
        for framework, contract in unit.get("direct_assertion_contracts", {}).items():
            assert framework in roots and contract["module"] and contract["constructor"]
            assert contract["input"] and contract["fields"]
            assert set(contract["fields"]) <= set(unit["observable"]), unit["id"]
            assert any(ref["kind"] == "existing_oracle" for ref in unit["evidence"][framework])
        for framework, contract in unit.get("exception_oracle_contracts", {}).items():
            assert framework in roots and all(contract.get(key) for key in
                                              ("module", "import_symbol", "exception_module", "exception", "context", "call"))
            assert any(ref["kind"] == "existing_oracle" for ref in unit["evidence"][framework])
        for framework, contract in unit.get("paired_call_contracts", {}).items():
            assert framework in unit["evidence"] and all(contract.get(key) for key in
                ("module", "function", "wrapper", "source_variable", "wrapped_variable",
                 "reference_variable", "guard_argument", "reference_type", "candidate_type"))
            assert any(ref["kind"] == "existing_oracle" for ref in unit["evidence"][framework])
        for framework, contract in unit.get("literal_result_contracts", {}).items():
            assert framework in unit["evidence"] and all(contract.get(key) for key in
                ("module", "function", "result_type", "source_path", "source_line"))
            assert contract["result_type"] in {"tuple", "scalar"}
            assert any(ref["kind"] == "source" and ref["path"] == contract["source_path"]
                       and ref["line"] == contract["source_line"]
                       for ref in unit["evidence"][framework])
            assert any(ref["kind"] == "existing_oracle" for ref in unit["evidence"][framework])
        for framework, contract in unit.get("paired_relation_contracts", {}).items():
            assert framework in unit["evidence"] and all(contract.get(key) for key in
                ("module", "function", "relation", "assertion", "source_path", "source_line"))
            assert contract["relation"] in {"same_tokens_different_order", "flat_bigram_same_tokens"}
            assert contract["assertion"] in {"assertEqual", "assertNotEqual"}
            assert any(ref["kind"] == "source" and ref["path"] == contract["source_path"]
                       and ref["line"] == contract["source_line"]
                       for ref in unit["evidence"][framework])
            assert any(ref["kind"] == "existing_oracle" for ref in unit["evidence"][framework])
        for framework, contract in unit.get("direct_result_contracts", {}).items():
            assert framework in unit["evidence"] and all(contract.get(key) for key in
                ("source_path", "method", "observable", "assertions"))
            assert contract["observable"] in unit["observable"]
            assert any(ref["kind"] == "source" and ref["path"] == contract["source_path"]
                       for ref in unit["evidence"][framework])
            assert any(ref["kind"] == "existing_oracle" for ref in unit["evidence"][framework])
        for framework, contract in unit.get("state_transition_contracts", {}).items():
            assert framework in unit["evidence"] and all(contract.get(key) for key in
                ("source_path", "getter", "append_method", "before_line", "append_line",
                 "after_line", "before_target", "after_target"))
            source_lines = {ref["line"] for ref in unit["evidence"][framework]
                            if ref["kind"] == "source" and ref["path"] == contract["source_path"]}
            assert {contract["before_line"], contract["append_line"],
                    contract["after_line"]} <= source_lines, unit["id"]
            assert any(ref["kind"] == "existing_oracle" for ref in unit["evidence"][framework])

    children: dict[str, list[str]] = {unit_id: [] for unit_id in ids}
    for relation in data["relations"]:
        parent, child = relation["parent"], relation["child"]
        assert parent in ids and child in ids and parent != child
        assert relation["type"] in ALLOWED_RELATIONS
        assert relation["when"], (parent, child)
        if relation["type"] == "refines":
            assert by_id[parent]["kind"] == "composite", parent
            children[parent].append(child)

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(unit_id: str) -> None:
        if unit_id in visited:
            return
        assert unit_id not in visiting, f"Refinement cycle at {unit_id}"
        visiting.add(unit_id)
        for child in children[unit_id]:
            visit(child)
        visiting.remove(unit_id)
        visited.add(unit_id)

    for unit_id in ids:
        visit(unit_id)
    assert all(children[u["id"]] for u in data["units"] if u["kind"] == "composite")

    relation_children: dict[str, set[str]] = {unit_id: set() for unit_id in ids}
    for relation in data["relations"]:
        relation_children[relation["parent"]].add(relation["child"])
    composition_ids: set[str] = set()
    for case in data.get("composition_cases", []):
        assert case["id"] not in composition_ids, case["id"]
        composition_ids.add(case["id"])
        assert by_id[case["target"]]["kind"] == "composite", case["id"]
        assert case["components"] and len(case["components"]) == len(set(case["components"]))
        assert case["applicability"] and case["join_observations"] and case["needed_check"]
        assert case["status"] == "composition_requirement_unverified", case["id"]
        reachable = set()
        pending = [case["target"]]
        while pending:
            next_id = pending.pop()
            for child in relation_children[next_id] - reachable:
                reachable.add(child)
                pending.append(child)
        assert set(case["components"]) <= reachable, case["id"]

    for trace in data["traces"]:
        assert trace["trace_type"] == "path_template", trace["id"]
        assert set(trace["conditions"]) <= dimensions, trace["id"]
        for dimension_id, value in trace["conditions"].items():
            item = dimension_items[dimension_id]
            if item.get("type") == "structured":
                assert isinstance(value, dict), (trace["id"], dimension_id)
                assert set(value) <= set(item["fields"]), (trace["id"], dimension_id)
            else:
                assert value in item["observed_values"], (trace["id"], dimension_id, value)
        assert all(step in ids for step in trace["steps"]), trace["id"]
        assert len(trace["steps"]) == len(set(trace["steps"])), trace["id"]
        assert isinstance(trace["unmapped_observations"], list)

    rule_ids: set[str] = set()
    for rule in data.get("applicability_rules", []):
        assert rule["id"] not in rule_ids, rule["id"]
        rule_ids.add(rule["id"])
        assert rule["when"] and rule["reason"]
        assert set(rule["when"]) <= dimensions
        assert set(rule.get("requires_conditions", {})) <= dimensions
        assert set(rule["required_units"]) <= ids
        assert set(rule["not_applicable_dimensions"]) <= dimensions
        for dimension_id, expected in rule["when"].items():
            item = dimension_items[dimension_id]
            if expected == "present":
                assert item.get("type") == "structured", rule["id"]
            else:
                assert expected in item["observed_values"], rule["id"]
        for dimension_id, fields in rule["concrete_fields"].items():
            item = dimension_items[dimension_id]
            assert item.get("type") == "structured", rule["id"]
            assert fields and set(fields) <= set(item["fields"]), rule["id"]
        for trace in data["traces"]:
            if not rule_applies(rule, trace["conditions"]):
                continue
            assert set(rule["required_units"]) <= set(trace["steps"]), (rule["id"], trace["id"])
            assert not set(rule["not_applicable_dimensions"]) & set(trace["conditions"]), (rule["id"], trace["id"])
            assert all(
                trace["conditions"].get(key) == value
                for key, value in rule.get("requires_conditions", {}).items()
            ), (rule["id"], trace["id"])

    handoff_data = json.loads((HERE / "handoff_evidence.json").read_text(encoding="utf-8"))
    assert handoff_data["status"].startswith("static source handoffs")
    trace_by_id = {trace["id"]: trace for trace in data["traces"]}
    handoff_ids: set[str] = set()
    handoff_refs = 0
    for handoff in handoff_data["handoffs"]:
        handoff_id = handoff["id"]
        assert handoff_id not in handoff_ids, handoff_id
        handoff_ids.add(handoff_id)
        steps = trace_by_id[handoff["path_template"]]["steps"]
        assert handoff["from_unit"] in steps and handoff["to_unit"] in steps
        assert handoff["transferred"] and handoff["execution_profile"] and handoff["remaining_check"]
        assert set(handoff["source_refs"]) == set(roots), handoff_id
        for framework, locations in handoff["source_refs"].items():
            assert locations, (handoff_id, framework)
            for ref in locations:
                source = roots[framework] / ref["path"]
                assert source.is_file(), source
                lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
                assert 1 <= ref["line"] <= len(lines), (source, ref["line"])
                assert lines[ref["line"] - 1].strip(), (source, ref["line"])
                handoff_refs += 1

    return {
        "snapshots": len(roots),
        "dimensions": len(dimensions),
        "units": len(ids),
        "relations": len(data["relations"]),
        "traces": len(data["traces"]),
        "composition_cases": len(composition_ids),
        "applicability_rules": len(rule_ids),
        "source_handoffs": len(handoff_ids),
        "handoff_source_locations": handoff_refs,
        "evidence_locations": refs,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, sort_keys=True))
