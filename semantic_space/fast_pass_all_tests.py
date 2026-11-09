"""Route every indexed source test through one conservative review pass.

This pass combines existing source parsing, candidate area classification and
checked evidence. It produces a complete work queue, not 32k property claims.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
KEY = ("snapshot", "path", "test_name", "line")
FIELDS = (*KEY, "entry_kind", "collection_role", "candidate_area", "candidate_areas",
          "area_basis", "area_ambiguous", "oracle_forms", "source_route_class",
          "helper_source_link_status", "http_request_oracle_status", "bulk_status",
          "evidence_level",
          "mapped_unit_ids", "claim_level", "triage_bucket", "risk_flags",
          "next_action", "family_key")
FAMILY_FIELDS = ("family_key", "entries", "snapshot", "triage_bucket", "candidate_area",
                 "oracle_forms", "area_basis", "example_path", "example_test_name",
                 "example_line", "risk_flags")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in KEY)


def inherited_area_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = []
    for row in rows:
        if not row["method"]:
            continue
        classification = json.loads(row["classification"])
        candidates.append({
            "snapshot": row["snapshot"], "path": row["test_path"],
            "test_name": f"{row['test_class']}::{row['method']}",
            "line": row["test_class_line"], "kind": "python_inherited_test_candidate",
            "primary_area": classification["primary_area"],
            "candidate_areas": ",".join(area["id"] for area in classification["areas"]),
            "basis": "inherited_definition:" + classification["basis"],
            "ambiguous": "1" if classification["ambiguous"] else "0",
        })
    return candidates


def classify(entry: dict[str, str], area: dict[str, str] | None) -> dict[str, str]:
    evidence = entry["evidence_level"]
    status = entry["batch_review_status"]
    units = entry["batch_review_units"] if status == "mapped" else ""
    flags = []
    if area is None:
        flags.append("area_candidate_missing")
    else:
        area_basis = area["basis"].removeprefix("inherited_definition:")
        if area["basis"].startswith("inherited_definition:"):
            flags.append("inherited_class_context_needs_review")
        if area["ambiguous"] == "1":
            flags.append("area_ambiguous")
        if area_basis in {"test_name", "file_path", "component_test_directory",
                          "file_only_path", "protocol_test_name"}:
            flags.append("area_inferred_from_name_or_path")
    if entry["analysis_status"] not in {"parsed_exact_location", "parsed_unique_name"}:
        flags.append("definition_not_exactly_located")
    if entry["collection_role"] not in {"test_candidate", "rust_source_candidate"}:
        flags.append("collection_identity_unconfirmed")
    if entry.get("source_route_class") in {"fixture 注入", "测试辅助函数"}:
        flags.append("route_origin_not_checked_operation")

    if evidence in {"batch_source_property_mapped", "source_property_confirmed"}:
        claim, bucket, action = ("source_property_checked", "reviewed_property",
                                 "check_runtime_conditions_and_cross_framework_relation")
    elif evidence in {"source_out_of_request_scope", "batch_source_skipped_by_class"}:
        claim, bucket, action = ("source_scope_checked", "outside_or_skipped",
                                 "retain_inventory_scope_evidence")
    elif evidence in {"batch_source_review_needs_refinement", "source_candidate",
                      "batch_source_oracle_unresolved", "prior_source_review"}:
        claim, bucket, action = ("source_review_incomplete", "refine_space_or_oracle",
                                 "read_checked_property_and_refine_unit_or_oracle")
    elif evidence == "source_call_oracle_link_candidate":
        claim, bucket, action = ("source_relation_candidate", "bind_input_and_property",
                                 "resolve_input_state_and_identify_asserted_property")
        if entry["bulk_input_binding"] == "symbolic":
            flags.append("input_value_unresolved")
    elif evidence == "static_call_oracle_candidate":
        claim, bucket, action = ("static_candidate", "check_call_oracle_role",
                                 "inspect_dut_reference_role_and_dataflow")
    elif evidence == "http_static_request_oracle_candidate":
        claim, bucket, action = ("static_candidate", "check_http_request_oracle",
                                 "bind_request_route_response_and_assertion")
    elif evidence == "supplemental_static_candidate":
        claim, bucket, action = ("static_candidate", "check_indirect_source_link",
                                 "resolve_helper_chain_and_assertion")
    else:
        reason = entry["unlinked_triage_reason"]
        buckets = {
            "direct_oracle_without_direct_product_call": "locate_checked_operation",
            "product_call_without_direct_oracle_link": "locate_indirect_oracle",
            "oracle_in_helper_or_callback": "expand_helper_or_callback",
            "inherited_test_candidate": "bind_inherited_test_context",
            "rust_test_source": "inspect_rust_test_body",
            "helper_resolution_gap": "resolve_helper_definition",
            "helper_without_located_oracle": "find_helper_oracle",
            "file_or_unparsed_source_record": "resolve_file_level_test",
            "ambiguous_source_definition": "resolve_definition_location",
            "non_test_or_collection_helper": "verify_collection_role",
        }
        claim, bucket, action = ("unresolved", buckets.get(reason, "trace_unlinked_entry"),
                                 entry["next_check"] or "inspect_source_entry")
    if not entry["oracle_forms"] and not entry["batch_review_oracle_lines"]:
        flags.append("oracle_not_directly_visible")

    candidate_area = area["primary_area"] if area else ""
    candidate_areas = area["candidate_areas"] if area else entry["object_candidates"]
    area_basis = area["basis"] if area else "unavailable_for_entry_kind"
    area_ambiguous = area["ambiguous"] if area else ""
    oracle_forms = entry["oracle_forms"]
    family_key = "|".join((entry["snapshot"], bucket, candidate_area or "unplaced",
                            oracle_forms or "no_oracle", area_basis))
    return {**{field: entry[field] for field in KEY},
            "entry_kind": entry["entry_kind"],
            "collection_role": entry["collection_role"],
            "candidate_area": candidate_area,
            "candidate_areas": candidate_areas,
            "area_basis": area_basis,
            "area_ambiguous": area_ambiguous,
            "oracle_forms": oracle_forms,
            "source_route_class": entry.get("source_route_class", ""),
            "helper_source_link_status": entry.get("helper_source_link_status", ""),
            "http_request_oracle_status": entry.get("http_request_oracle_status", ""),
            "bulk_status": entry.get("bulk_status", ""),
            "evidence_level": evidence,
            "mapped_unit_ids": units,
            "claim_level": claim,
            "triage_bucket": bucket,
            "risk_flags": ";".join(flags),
            "next_action": action,
            "family_key": family_key}


def build(index: list[dict[str, str]], areas: list[dict[str, str]],
          inherited: list[dict[str, str]] | None = None) -> tuple[list[dict[str, str]], dict]:
    all_areas = areas + inherited_area_candidates(inherited or [])
    area_by_key = {key(row): row for row in all_areas}
    index_by_key = {key(row): row for row in index}
    if len(area_by_key) != len(all_areas) or len(index_by_key) != len(index):
        raise ValueError("duplicate source entry")
    superseded = set(area_by_key) - set(index_by_key)
    if any(area_by_key[item]["kind"] != "source_test_file_unresolved" for item in superseded):
        raise ValueError("non-placeholder area candidates missing from the master index")
    if any(not any(entry["snapshot"] == item[0] and entry["path"] == item[1]
                   for entry in index) for item in superseded):
        raise ValueError("file placeholder missing its discovered inherited entries")
    output = [classify(entry, area_by_key.get(key(entry))) for entry in index]
    counts = Counter(row["triage_bucket"] for row in output)
    claims = Counter(row["claim_level"] for row in output)
    risks = Counter(flag for row in output for flag in row["risk_flags"].split(";") if flag)
    families: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in output:
        families[row["family_key"]].append(row)
    summary = {
        "indexed_entries": len(output),
        "candidate_area_entries": len(set(area_by_key) & set(index_by_key)),
        "inherited_area_candidates": len(inherited or []) and sum(
            bool(row["method"]) for row in inherited or []),
        "superseded_file_placeholders": len(superseded),
        "entries_without_area_candidate": len(set(index_by_key) - set(area_by_key)),
        "triage_buckets": dict(counts),
        "claim_levels": dict(claims),
        "risk_flags": dict(risks),
        "review_families": len(families),
        "largest_review_family": max(map(len, families.values()), default=0),
        "interpretation": "Every indexed entry receives a review route. A source link is not a verified semantic unit, and candidate areas are not confirmed request coverage.",
    }
    return output, summary


def main() -> dict:
    rows, summary = build(read(HERE / "master-review-index.tsv"),
                          read(HERE / "test-entry-classification.tsv"),
                          read(HERE / "inherited-test-audit.tsv"))
    with (HERE / "fast-pass-all-tests.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    families: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        families[row["family_key"]].append(row)
    with (HERE / "fast-pass-review-families.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FAMILY_FIELDS, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        for family_key, members in sorted(families.items(), key=lambda item: (-len(item[1]), item[0])):
            example = members[0]
            writer.writerow({"family_key": family_key, "entries": len(members),
                             "snapshot": example["snapshot"],
                             "triage_bucket": example["triage_bucket"],
                             "candidate_area": example["candidate_area"],
                             "oracle_forms": example["oracle_forms"],
                             "area_basis": example["area_basis"],
                             "example_path": example["path"],
                             "example_test_name": example["test_name"],
                             "example_line": example["line"],
                             "risk_flags": ";".join(sorted({flag for row in members
                                 for flag in row["risk_flags"].split(";") if flag}))})
    (HERE / "fast-pass-all-tests.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
