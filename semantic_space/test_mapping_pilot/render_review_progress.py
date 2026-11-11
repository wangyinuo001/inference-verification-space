"""Build a self-contained, one-screen heatmap of all indexed test entries."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from build_stage_placement import main as build_stage_placement
from build_confirmed_crosswalk import main as build_confirmed_crosswalk
from resolve_inherited_methods import ROOTS, main as resolve_inherited_methods
from build_inherited_property_crosswalk import main as build_inherited_property_crosswalk


HERE = Path(__file__).resolve().parent
SPACE = HERE.parent
OUTPUT = HERE / "review_progress.html"
KEY = ("snapshot", "path", "test_name", "line")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row[field] for field in KEY)


def category(scan: dict[str, str], review: dict[str, str], placement: dict[str, str],
             confirmed: bool = False, inherited_source: bool = False) -> str:
    status = review["source_review_status"]
    if confirmed:
        return "g"
    if status == "mapped" and review["source_reviewed_units"]:
        return "g"  # Source-reviewed property mapped to the earlier candidate space.
    if status in {"needs_refinement", "sample_reviewed", "oracle_unresolved"}:
        return "r"
    if inherited_source:
        return "i"
    route = scan["scan_route"]
    if route == "reviewed_out_of_request_scope":
        return "o"
    if placement["placement_status"] == "linked_stage":
        return "l"
    if placement["placement_status"] == "linked_domain_stage":
        return "h"
    if route == "source_model_symbol_candidate":
        return "s"
    if route == "source_model_module_candidate":
        return "m"
    if route in {"unsupported_source", "definition_not_resolved", "oracle_location_missing"}:
        return "u"
    if placement["placement_status"] == "domain_stage_candidate":
        return "p"
    if placement["placement_status"] in {"multiple_stage_candidates", "cross_stage_review"}:
        return "x"
    if placement["placement_status"] == "adjacent_domain":
        return "a"
    return "d"


def main() -> None:
    build_stage_placement()
    build_confirmed_crosswalk()
    if all((root / ("vllm" if name == "vllm" else "python/sglang")).is_dir() for name, root in ROOTS.items()):
        resolve_inherited_methods()
    elif not (HERE / "inherited-method-sources.tsv").is_file():
        raise FileNotFoundError("Provide the checked inherited-method-sources.tsv or full source checkouts")
    build_inherited_property_crosswalk()
    scan = read(HERE / "full-scan-index.tsv")
    review = {key(row): row for row in read(SPACE / "oracle-review-queue.tsv")}
    placement = {key(row): row for row in read(HERE / "stage-placement-all.tsv")}
    confirmed = {key(row): row for row in read(HERE / "confirmed-property-crosswalk.tsv")}
    direct_properties = {key(row): row for row in read(HERE / "direct-property-crosswalk.tsv")}
    inherited_properties = {key(row): row for row in read(HERE / "inherited-property-crosswalk.tsv")}
    assert not ((confirmed.keys() & inherited_properties.keys()) |
                (confirmed.keys() & direct_properties.keys()) |
                (direct_properties.keys() & inherited_properties.keys()))
    inherited = {key(row): row for row in read(HERE / "inherited-method-sources.tsv")}
    unit_names = {unit["id"]: unit.get("title") or unit.get("property", "")
                  for unit in json.loads((SPACE / "space_seed.json").read_text(encoding="utf-8"))["units"]}
    assert len(scan) == len(review) == len(placement) == 31968
    files: list[list[str]] = []
    file_ids: dict[tuple[str, str], int] = {}
    entries: list[list[str | int]] = []
    counts: Counter[str] = Counter()
    framework_counts: dict[str, Counter[str]] = {"vllm": Counter(), "sglang": Counter()}
    stage_count = 0
    linked_stage_count = 0
    for row in scan:
        item_key = key(row)
        observed = review[item_key]
        placed = placement[item_key]
        checked = confirmed.get(item_key) or direct_properties.get(item_key) or inherited_properties.get(item_key)
        inherited_origin = inherited.get(item_key)
        stage_count += bool(placed["stage"])
        linked_stage_count += placed["placement_status"] in {"linked_stage", "linked_domain_stage"}
        file_key = (row["snapshot"], row["path"])
        if file_key not in file_ids:
            file_ids[file_key] = len(files)
            files.append(list(file_key))
        status = category(row, observed, placed, checked is not None,
                          bool(inherited_origin and inherited_origin["status"] == "resolved_unique"))
        counts[status] += 1
        framework_counts[row["snapshot"]][status] += 1
        units = (checked["unit_id"] if checked else observed["source_reviewed_units"]) if status == "g" else (
            row["source_model_symbol_units"] if status == "s" else
            row["source_model_module_units"] if status == "m" else ""
        )
        entries.append([
            file_ids[file_key], row["test_name"], int(row["line"]), status,
            units, row["oracle_forms"], row["observation_subjects"][:150],
            placed["stage"], placed["stage_source"], placed["product_target"],
            placed["input_binding"],
            checked["property"] if checked else observed["source_reviewed_property"] if status == "g" else "",
            checked["oracle_source"] if checked else observed["oracle_sites_preview"] if status == "g" else "",
            inherited_origin["source_path"] if inherited_origin and inherited_origin["status"] == "resolved_unique" else "",
            int(inherited_origin["source_line"]) if inherited_origin and inherited_origin["status"] == "resolved_unique" else 0,
            checked["conditions"] if checked else "",
        ])
    assert sum(counts.values()) == len(entries) == len({key(row) for row in scan})
    payload = json.dumps({
        "files": files, "entries": entries, "counts": dict(counts),
        "stageCount": stage_count,
        "linkedStageCount": linked_stage_count,
        "oracleCount": sum(bool(row["oracle_forms"]) for row in review.values()),
        "unitNames": unit_names,
        "frameworkCounts": {name: dict(value) for name, value in framework_counts.items()},
    }, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (HERE / "review_progress.template.html").read_text(encoding="utf-8")
    OUTPUT.write_text(template.replace("/*__DATA__*/", payload), encoding="utf-8")
    print(json.dumps({"entries": len(entries), "files": len(files), "counts": dict(counts),
                      "output": str(OUTPUT), "bytes": OUTPUT.stat().st_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()
