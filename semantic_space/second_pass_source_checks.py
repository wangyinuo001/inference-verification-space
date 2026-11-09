"""Second full-inventory pass: compare candidate area with actual source calls.

This pass adds an independent product-module hint and distinguishes a call
whose result reaches an assertion from other source calls. A module hint may
describe setup or a cross-scope dependency, so conflicts are queued for review
rather than silently overwriting the first-pass candidate.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from build_test_catalog import PRODUCT_PREFIX_AREAS


HERE = Path(__file__).resolve().parent
# Source-module roles, used only to add review hints. The checked property can
# live at a different scope (e.g. a model module used during input processing).
MODULE_ROLE_HINTS = PRODUCT_PREFIX_AREAS + (
    ("sglang.srt.debug_utils.", "diagnostics"),
    ("vllm.distributed.kv_transfer.", "transfer"),
    ("vllm.distributed.ec_transfer.", "transfer"),
    ("vllm.v1.attention.", "model"),
    ("vllm.v1.worker.gpu.", "model"),
    ("vllm.tool_parsers.", "output"),
    ("sglang.srt.utils.auth.", "security"),
    ("sglang.srt.parser.reasoning_parser.", "output"),
    ("sglang.srt.parser.jinja_template_utils.", "ingress"),
    ("vllm.config.", "configuration"),
    ("sglang.multimodal_gen.runtime.", "image_generation"),
    ("vllm.multimodal.", "ingress"),
    ("vllm.envs.", "configuration"),
)
KEY = ("snapshot", "path", "test_name", "line")
FIELDS = (*KEY, "candidate_area", "candidate_areas", "claim_level",
          "source_link_status", "source_call_count", "linked_product_targets",
          "module_area_hints", "area_relation", "input_binding",
          "oracle_link_kind", "review_action", "candidate_areas_v2",
          "source_review_verdict", "reviewed_area", "checked_property",
          "review_evidence")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def key(row: dict[str, str], line_field: str = "line") -> tuple[str, ...]:
    return (row["snapshot"], row["path"], row["test_name"], row[line_field])


def source_calls_by_entry(catalog: dict,
                          inherited_rows: list[dict[str, str]] | None = None) -> dict[tuple[str, ...], list[str]]:
    result = {}
    for snapshot in catalog["snapshots"]:
        for file in snapshot["files"]:
            for item in file["items"]:
                item_key = (snapshot["id"], file["path"], item["name"], str(item["line"]))
                result[item_key] = [call["target"] for call in item["source_calls"]]
    for row in inherited_rows or []:
        if not row["method"]:
            continue
        item_key = (row["snapshot"], row["test_path"],
                    f"{row['test_class']}::{row['method']}", row["test_class_line"])
        result[item_key] = [call["target"] for call in json.loads(row["source_calls"] or "[]")]
    return result


def module_hints(targets: list[str]) -> list[str]:
    return sorted({area for target in targets for prefix, area in MODULE_ROLE_HINTS
                   if target.startswith(prefix)})


def inspect(row: dict[str, str], calls: list[str], bulk: dict[str, str] | None) -> dict[str, str]:
    bulk = bulk or {}
    linked = bulk.get("bulk_status") == "source_linked"
    checked_calls = json.loads(bulk["product_calls_json"]) if linked else []
    checked_targets = [call["target"] for call in checked_calls]
    # A linked call has more relevant evidence than any setup call in the test body.
    target_set = checked_targets if linked else calls
    hints = module_hints(target_set)
    existing = set(filter(None, row["candidate_areas"].split(",")))
    if not hints:
        relation = "no_module_hint"
    elif row["candidate_area"] in hints and len(hints) == 1:
        relation = "primary_area_supported_by_module"
    elif row["candidate_area"] in hints:
        relation = "primary_area_among_multiple_module_hints"
    elif linked:
        relation = "linked_call_crosses_area_candidate"
    else:
        relation = "unlinked_call_crosses_area_candidate"
    if relation == "linked_call_crosses_area_candidate":
        action = "inspect_checked_property_before_changing_area"
    elif relation == "unlinked_call_crosses_area_candidate":
        action = "separate_setup_call_from_checked_operation"
    elif linked:
        action = "bind_symbolic_inputs_and_asserted_property" if bulk.get("input_binding") == "symbolic" else "identify_asserted_property"
    else:
        action = row["next_action"]
    oracles = json.loads(bulk["oracle_expressions_json"]) if linked else []
    kinds = sorted({oracle["link_kind"] for oracle in oracles})
    return {**{field: row[field] for field in KEY},
            "candidate_area": row["candidate_area"],
            "candidate_areas": row["candidate_areas"],
            "claim_level": row["claim_level"],
            "source_link_status": "call_result_to_assertion" if linked else "no_direct_link",
            "source_call_count": len(calls),
            "linked_product_targets": json.dumps(checked_targets, ensure_ascii=False),
            "module_area_hints": ",".join(hints),
            "area_relation": relation,
            "input_binding": bulk.get("input_binding", "") if linked else "",
            "oracle_link_kind": ",".join(kinds),
            "review_action": action,
            "candidate_areas_v2": ",".join(sorted(existing | set(hints))) or row["candidate_area"]}


def reviewed_conflicts(sglang_rows: list[dict[str, str]],
                       vllm_rows: list[dict[str, str]]) -> dict[tuple[str, ...], dict[str, str]]:
    """Attach independently checked conclusions; never turn module hints into labels."""
    result = {}
    for row in sglang_rows:
        item_key = key(row)
        result[item_key] = {
            "source_review_verdict": row["review_result"],
            "reviewed_area": row["reviewed_area"],
            "checked_property": row["checked_behavior"],
            "review_evidence": "agent_sglang_cross_scope_review.tsv",
        }
    for row in vllm_rows:
        item_key = key(row, "test_line")
        if item_key in result:
            raise ValueError("duplicate reviewed source entry")
        result[item_key] = {
            "source_review_verdict": row["review_verdict"],
            "reviewed_area": row["suggested_area"],
            "checked_property": row["checked_property"],
            "review_evidence": "agent_vllm_prefix_conflicts_review.tsv",
        }
    return result


def build(rows: list[dict[str, str]], catalog: dict,
          bulk_rows: list[dict[str, str]],
          inherited_rows: list[dict[str, str]] | None = None,
          reviews: dict[tuple[str, ...], dict[str, str]] | None = None) -> tuple[list[dict[str, str]], dict]:
    calls = source_calls_by_entry(catalog, inherited_rows)
    bulk = {key(row, "test_line"): row for row in bulk_rows}
    if len(bulk) != len(bulk_rows):
        raise ValueError("duplicate bulk source entry")
    keys = {key(row) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate first-pass entry")
    if not set(bulk) <= keys:
        raise ValueError("bulk source link missing from first pass")
    reviews = reviews or {}
    if not set(reviews) <= keys:
        raise ValueError("reviewed source entry missing from first pass")
    output = []
    for row in rows:
        item_key = key(row)
        checked = inspect(row, calls.get(item_key, []), bulk.get(item_key))
        checked.update(reviews.get(item_key, {}))
        output.append(checked)
    relations = Counter(row["area_relation"] for row in output)
    summary = {
        "entries": len(output),
        "direct_source_links": sum(row["source_link_status"] == "call_result_to_assertion" for row in output),
        "entries_with_module_hint": sum(bool(row["module_area_hints"]) for row in output),
        "entries_with_source_review": sum(bool(row.get("source_review_verdict")) for row in output),
        "area_relations": dict(relations),
        "interpretation": "Module-prefix areas are review hints. A linked call reaches an assertion, but its module does not by itself determine the checked property or request-level scope.",
    }
    return output, summary


def main() -> dict:
    reviews = reviewed_conflicts(
        read_tsv(HERE / "agent_sglang_cross_scope_review.tsv"),
        read_tsv(HERE / "agent_vllm_prefix_conflicts_review.tsv"))
    rows, summary = build(read_tsv(HERE / "fast-pass-all-tests.tsv"),
                          json.loads((HERE / "test_catalog.json").read_text(encoding="utf-8")),
                          read_tsv(HERE / "bulk-call-oracle-links.tsv"),
                          read_tsv(HERE / "inherited-test-audit.tsv"),
                          reviews)
    with (HERE / "second-pass-all-tests.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (HERE / "second-pass-cross-scope.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in rows if row["area_relation"].endswith("crosses_area_candidate"))
    (HERE / "second-pass-all-tests.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
