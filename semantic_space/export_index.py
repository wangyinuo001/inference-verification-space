"""Export source evidence, path templates, and composition audit tables."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from validate_space import HERE, rule_applies, validate


def main() -> None:
    validate()
    model = json.loads((HERE / "space_seed.json").read_text(encoding="utf-8"))

    with (HERE / "evidence-index.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(("unit_id", "framework", "kind", "path", "line", "pinned_url"))
        for unit in model["units"]:
            for framework, refs in unit["evidence"].items():
                snapshot = model["snapshots"][framework]
                repo = "vllm-project/vllm" if framework == "vllm" else "sgl-project/sglang"
                for ref in refs:
                    url = (
                        f"https://github.com/{repo}/blob/{snapshot['commit']}/"
                        f"{ref['path']}#L{ref['line']}"
                    )
                    writer.writerow((unit["id"], framework, ref["kind"], ref["path"], ref["line"], url))

    unit_ids = [unit["id"] for unit in model["units"]]
    units_by_id = {unit["id"]: unit for unit in model["units"]}
    with (HERE / "trace-unit-matrix.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(("trace_id", "trace_type", "conditions", *unit_ids, "unmapped_observations"))
        for trace in model["traces"]:
            active = set(trace["steps"])
            writer.writerow(
                (
                    trace["id"],
                    trace["trace_type"],
                    json.dumps(trace["conditions"], ensure_ascii=False, sort_keys=True),
                    *(int(unit_id in active) for unit_id in unit_ids),
                    " | ".join(trace["unmapped_observations"]),
                )
            )

    # An anchor is only an indexed source/test location. It does not establish
    # that a repository test exercises the full trace or proves the obligation.
    with (HERE / "trace-evidence-audit.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow((
            "trace_id", "framework", "obligation_count", "source_anchored",
            "test_anchored", "oracle_anchored", "without_test_anchor",
        ))
        for trace in model["traces"]:
            obligations = [units_by_id[unit_id] for unit_id in trace["steps"]
                           if units_by_id[unit_id]["kind"] == "obligation"]
            for framework in model["snapshots"]:
                source = test = oracle = 0
                without_test = []
                for unit in obligations:
                    kinds = {item["kind"] for item in unit["evidence"][framework]}
                    source += "source" in kinds
                    test += "existing_test" in kinds
                    oracle += "existing_oracle" in kinds
                    if "existing_test" not in kinds:
                        without_test.append(unit["id"])
                writer.writerow((trace["id"], framework, len(obligations), source,
                                 test, oracle, " | ".join(without_test)))

    with (HERE / "composition-audit.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(
            (
                "case_id", "target_unit", "component_unit", "applicability",
                "join_observations", "status", "needed_check",
            )
        )
        for case in model["composition_cases"]:
            for component in case["components"]:
                writer.writerow(
                    (
                        case["id"], case["target"], component,
                        case["applicability"], " | ".join(case["join_observations"]),
                        case["status"], case["needed_check"],
                    )
                )

    with (HERE / "applicability-audit.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(("trace_id", "rule_id", "required_units", "missing_concrete_fields", "reason"))
        for trace in model["traces"]:
            for rule in model["applicability_rules"]:
                if not rule_applies(rule, trace["conditions"]):
                    continue
                missing = [
                    f"{dimension_id}.{field}"
                    for dimension_id, fields in rule["concrete_fields"].items()
                    for field in fields
                    if field not in trace["conditions"].get(dimension_id, {})
                ]
                writer.writerow(
                    (trace["id"], rule["id"], " | ".join(rule["required_units"]),
                     " | ".join(missing), rule["reason"])
                )

    handoffs = json.loads((HERE / "handoff_evidence.json").read_text(encoding="utf-8"))["handoffs"]
    with (HERE / "handoff-index.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(
            ("handoff_id", "path_template", "from_unit", "to_unit", "transferred",
             "execution_profile", "framework", "path", "line", "source_line",
             "pinned_url", "remaining_check")
        )
        for handoff in handoffs:
            for framework, refs in handoff["source_refs"].items():
                snapshot = model["snapshots"][framework]
                repo = "vllm-project/vllm" if framework == "vllm" else "sgl-project/sglang"
                root = (HERE / snapshot["directory"]).resolve()
                for ref in refs:
                    line_text = (root / ref["path"]).read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[ref["line"] - 1].strip()
                    url = (
                        f"https://github.com/{repo}/blob/{snapshot['commit']}/"
                        f"{ref['path']}#L{ref['line']}"
                    )
                    writer.writerow(
                        (handoff["id"], handoff["path_template"], handoff["from_unit"],
                         handoff["to_unit"], " | ".join(handoff["transferred"]),
                         handoff["execution_profile"], framework, ref["path"], ref["line"],
                         line_text, url, handoff["remaining_check"])
                    )

    with (HERE / "condition-template-audit.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(("dimension", "kind", "value_or_field", "templates", "count"))
        for dimension in model["condition_dimensions"]:
            dimension_id = dimension["id"]
            if dimension.get("type") == "structured":
                entries = [("structured_field", field) for field in dimension["fields"]]
            else:
                entries = [("enum_value", value) for value in dimension["observed_values"]]
            for kind, value in entries:
                matching = [
                    trace["id"] for trace in model["traces"]
                    if (
                        value in trace["conditions"].get(dimension_id, {})
                        if kind == "structured_field"
                        else trace["conditions"].get(dimension_id) == value
                    )
                ]
                writer.writerow((dimension_id, kind, value, " | ".join(matching), len(matching)))

    with (HERE / "unit-template-audit.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(("unit_id", "kind", "templates", "count"))
        for unit in model["units"]:
            matching = [
                trace["id"] for trace in model["traces"] if unit["id"] in trace["steps"]
            ]
            writer.writerow((unit["id"], unit["kind"], " | ".join(matching), len(matching)))
    print("Exported evidence, handoff, applicability, and template-coverage audit tables")


if __name__ == "__main__":
    main()
