"""Group all indexed source entries into reproducible review families.

Families schedule source review; they do not assign verification properties.
Every indexed entry retains its own row, source location and review status.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEED = json.loads((HERE / "space_seed.json").read_text(encoding="utf-8"))
ROOTS = {
    key: (HERE / value["directory"]).resolve()
    for key, value in SEED["snapshots"].items()
}
ENTRY_FIELDS = (
    "snapshot", "path", "test_name", "line", "entry_kind", "collection_role",
    "definition_path", "source_method_line", "analysis_status", "exact_family",
    "shape_family", "triage_family", "target_family", "primary_oracle_target",
    "source_review_status", "object_candidates", "oracle_forms", "oracle_sources",
    "oracle_sites_preview",
)
GROUP_FIELDS = (
    "shape_family", "entries", "unreviewed_entries", "exact_families",
    "source_files", "representative", "analysis_status",
)
TRIAGE_FIELDS = (
    "triage_family", "entries", "unreviewed_entries", "shape_families",
    "source_files", "representative", "object_candidates", "oracle_forms",
    "oracle_sources", "entry_kind",
)
TARGET_FIELDS = (
    "target_family", "entries", "unreviewed_entries", "shape_families",
    "source_files", "representative", "primary_oracle_target",
    "object_candidates", "oracle_forms",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def canonical_function(node: ast.FunctionDef | ast.AsyncFunctionDef,
                       *, mask_literals: bool) -> str:
    """Retain calls, assertions, branches, names and decorators.

    The shape variant masks literal values only.  Thus equal shape is a
    reading group, never a claim that test inputs or outcomes are equivalent.
    """
    result = copy.deepcopy(node)
    result.name = "_test"
    if (result.body and isinstance(result.body[0], ast.Expr)
            and isinstance(result.body[0].value, ast.Constant)
            and isinstance(result.body[0].value.value, str)):
        result.body.pop(0)
    if mask_literals:
        for sub in ast.walk(result):
            if isinstance(sub, ast.Constant) and sub.value is not None:
                if isinstance(sub.value, bool):
                    continue
                sub.value = f"<literal:{type(sub.value).__name__}>"
    return ast.dump(result, annotate_fields=True, include_attributes=False)


@lru_cache(maxsize=96)
def parse_file(path: Path) -> tuple[dict[int, ast.AST], dict[str, list[ast.AST]], str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {}, {}, type(exc).__name__
    by_line: dict[int, ast.AST] = {}
    by_name: dict[str, list[ast.AST]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            by_line[node.lineno] = node
            by_name[node.name].append(node)
    return by_line, by_name, "parsed"


def source_function(entry: dict) -> tuple[ast.AST | None, str]:
    if not entry["entry_kind"].startswith("python_"):
        return None, "non_python_or_file_record"
    path = ROOTS[entry["snapshot"]] / entry["definition_path"]
    by_line, by_name, status = parse_file(path)
    if status != "parsed":
        return None, status
    method_name = entry["test_name"].rsplit("::", 1)[-1]
    line = int(entry["line"])
    direct = by_line.get(line)
    if isinstance(direct, (ast.FunctionDef, ast.AsyncFunctionDef)) \
            and direct.name == method_name:
        return direct, "parsed_exact_location"
    candidates = by_name.get(method_name, [])
    if len(candidates) == 1:
        return candidates[0], "parsed_unique_name"
    return None, "ambiguous_definition" if candidates else "definition_not_found"


def assign(entry: dict, targets: dict[tuple[str, str, str, str], Counter[str]]) -> dict:
    node, status = source_function(entry)
    identity = "|".join(entry[k] for k in ("snapshot", "path", "test_name", "line"))
    if node is None:
        exact = "unresolved_" + digest(identity)
        shape = exact
        method_line = ""
    else:
        exact_text = canonical_function(node, mask_literals=False)
        shape_text = canonical_function(node, mask_literals=True)
        exact = digest("|".join((entry["snapshot"], entry["definition_path"], exact_text)))
        shape = digest("|".join((
            entry["snapshot"], entry["object_candidates"], entry["oracle_forms"],
            entry["oracle_sources"], shape_text,
        )))
        method_line = str(node.lineno)
    triage = digest("|".join((
        entry["snapshot"], entry["entry_kind"], entry["object_candidates"],
        entry["oracle_forms"], entry["oracle_sources"],
    )))
    linked = targets.get(tuple(entry[k] for k in ("snapshot", "path", "test_name", "line")))
    primary_target = sorted(linked, key=lambda target: (-linked[target], target))[0] if linked else ""
    target_family = digest("|".join((
        entry["snapshot"], entry["object_candidates"], entry["oracle_forms"],
        primary_target,
    )))
    return {
        "snapshot": entry["snapshot"], "path": entry["path"],
        "test_name": entry["test_name"], "line": entry["line"],
        "entry_kind": entry["entry_kind"],
        "collection_role": entry["collection_role"],
        "definition_path": entry["definition_path"],
        "source_method_line": method_line, "analysis_status": status,
        "exact_family": exact, "shape_family": shape, "triage_family": triage,
        "target_family": target_family, "primary_oracle_target": primary_target,
        "source_review_status": entry["source_review_status"],
        "object_candidates": entry["object_candidates"],
        "oracle_forms": entry["oracle_forms"],
        "oracle_sources": entry["oracle_sources"],
        "oracle_sites_preview": entry["oracle_sites_preview"],
    }


def write_tsv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> dict:
    with (HERE / "oracle-review-queue.tsv").open(encoding="utf-8", newline="") as stream:
        entries = list(csv.DictReader(stream, delimiter="\t"))
    targets: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    with (HERE / "call-oracle-link-audit.tsv").open(encoding="utf-8", newline="") as stream:
        for link in csv.DictReader(stream, delimiter="\t"):
            if link["status"] == "static_oracle_link_candidate" and link["product_target"]:
                key = tuple(link[k] for k in ("snapshot", "path", "test_name", "test_line"))
                targets[key][link["product_target"]] += 1
    rows = [assign(entry, targets) for entry in entries]
    assert len(rows) == len(entries)
    write_tsv(HERE / "review-family-entries.tsv", rows, ENTRY_FIELDS)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["shape_family"]].append(row)
    group_rows = []
    for key, members in groups.items():
        group_rows.append({
            "shape_family": key, "entries": len(members),
            "unreviewed_entries": sum(not row["source_review_status"] for row in members),
            "exact_families": len({row["exact_family"] for row in members}),
            "source_files": len({(row["snapshot"], row["definition_path"]) for row in members}),
            "representative": "|".join(members[0][k] for k in ("snapshot", "path", "test_name")),
            "analysis_status": ";".join(sorted({row["analysis_status"] for row in members})),
        })
    group_rows.sort(key=lambda row: (-row["unreviewed_entries"], -row["entries"], row["shape_family"]))
    write_tsv(HERE / "review-family-groups.tsv", group_rows, GROUP_FIELDS)
    triage_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        triage_groups[row["triage_family"]].append(row)
    triage_rows = []
    for key, members in triage_groups.items():
        first = members[0]
        triage_rows.append({
            "triage_family": key, "entries": len(members),
            "unreviewed_entries": sum(not row["source_review_status"] for row in members),
            "shape_families": len({row["shape_family"] for row in members}),
            "source_files": len({(row["snapshot"], row["definition_path"]) for row in members}),
            "representative": "|".join(first[k] for k in ("snapshot", "path", "test_name")),
            "object_candidates": first["object_candidates"],
            "oracle_forms": first["oracle_forms"],
            "oracle_sources": first["oracle_sources"],
            "entry_kind": first["entry_kind"],
        })
    triage_rows.sort(key=lambda row: (-row["unreviewed_entries"], row["triage_family"]))
    write_tsv(HERE / "review-triage-groups.tsv", triage_rows, TRIAGE_FIELDS)
    target_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        target_groups[row["target_family"]].append(row)
    target_rows = []
    for key, members in target_groups.items():
        first = members[0]
        target_rows.append({
            "target_family": key, "entries": len(members),
            "unreviewed_entries": sum(not row["source_review_status"] for row in members),
            "shape_families": len({row["shape_family"] for row in members}),
            "source_files": len({(row["snapshot"], row["definition_path"]) for row in members}),
            "representative": "|".join(first[k] for k in ("snapshot", "path", "test_name")),
            "primary_oracle_target": first["primary_oracle_target"],
            "object_candidates": first["object_candidates"],
            "oracle_forms": first["oracle_forms"],
        })
    target_rows.sort(key=lambda row: (-row["unreviewed_entries"], row["target_family"]))
    write_tsv(HERE / "review-target-groups.tsv", target_rows, TARGET_FIELDS)
    parsed = [row for row in rows if row["analysis_status"].startswith("parsed_")]
    summary = {
        "entries": len(rows), "parsed_entries": len(parsed),
        "analysis_status": dict(Counter(row["analysis_status"] for row in rows)),
        "exact_families": len({row["exact_family"] for row in parsed}),
        "shape_families": len({row["shape_family"] for row in parsed}),
        "triage_groups": len(triage_groups),
        "target_groups": len(target_groups),
        "entries_with_linked_target": sum(bool(row["primary_oracle_target"]) for row in rows),
        "entries_in_multi_member_shape_families": sum(
            len(members) for members in groups.values() if len(members) > 1
        ),
        "largest_shape_family": max(map(len, groups.values()), default=0),
        "interpretation": "Families schedule source review. Matching shape masks literal values and does not prove matching inputs, paths, assertions or properties.",
    }
    (HERE / "review-family-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False))
