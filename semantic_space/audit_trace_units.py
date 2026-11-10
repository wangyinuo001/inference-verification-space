"""Export one evidence row per path, verification unit, and framework.

The output describes indexed source locations. A test anchor is not treated as
proof that the test exercises the full path or checks the whole property.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from validate_space import HERE, validate


FIELDS = (
    "trace_id", "trace_conditions", "step_index", "unit_id", "unit_kind",
    "phase", "state_scope", "unit_conditions", "framework", "evidence_level",
    "source_locations", "test_locations", "oracle_locations", "relations_in_trace",
    "composition_cases", "unmapped_observations",
)


def locations(refs: list[dict], kind: str) -> str:
    return " | ".join(
        f"{ref['path']}:{ref['line']}" for ref in refs if ref["kind"] == kind
    )


def export(seed_path: Path = HERE / "space_seed.json", output: Path = HERE / "trace-unit-evidence.tsv") -> dict:
    if seed_path == HERE / "space_seed.json":
        validate()
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    units = {unit["id"]: unit for unit in seed["units"]}
    rows = []
    for trace in seed["traces"]:
        step_ids = set(trace["steps"])
        relations = [
            relation for relation in seed["relations"]
            if relation["parent"] in step_ids and relation["child"] in step_ids
        ]
        cases = [case for case in seed["composition_cases"] if case["target"] in step_ids]
        for step_index, unit_id in enumerate(trace["steps"]):
            unit = units[unit_id]
            for framework in seed["snapshots"]:
                refs = unit.get("evidence", {}).get(framework, [])
                source = locations(refs, "source")
                test = locations(refs, "existing_test")
                oracle = locations(refs, "existing_oracle")
                level = (
                    "composite_requires_joint_check" if unit["kind"] == "composite"
                    else "oracle_location_indexed" if oracle
                    else "test_location_indexed" if test
                    else "source_only" if source
                    else "no_source_anchor"
                )
                rows.append({
                    "trace_id": trace["id"],
                    "trace_conditions": json.dumps(trace["conditions"], ensure_ascii=False, sort_keys=True),
                    "step_index": step_index,
                    "unit_id": unit_id,
                    "unit_kind": unit["kind"],
                    "phase": unit["phase"],
                    "state_scope": unit["state_scope"],
                    "unit_conditions": " | ".join(unit["conditions"]),
                    "framework": framework,
                    "evidence_level": level,
                    "source_locations": source,
                    "test_locations": test,
                    "oracle_locations": oracle,
                    "relations_in_trace": " | ".join(
                        f"{r['parent']} {r['type']} {r['child']} [{r['when']}]"
                        for r in relations if unit_id in (r["parent"], r["child"])
                    ),
                    "composition_cases": " | ".join(
                        case["id"] for case in cases
                        if unit_id == case["target"] or unit_id in case["components"]
                    ),
                    "unmapped_observations": " | ".join(trace["unmapped_observations"]),
                })
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return {"path_templates": len(seed["traces"]), "rows": len(rows), "output": str(output)}


if __name__ == "__main__":
    print(json.dumps(export(), ensure_ascii=False))
