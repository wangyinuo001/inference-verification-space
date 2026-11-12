"""Audit every locally reviewed round-eight row against pinned source and green keys.

This script reports progress only. It does not write the public crosswalk.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PILOT = HERE.parent
ROOT = REPO / "../../experiments/third_party/reference_repos"
ROOTS = {"vllm": ROOT / "vllm-20261105", "sglang": ROOT / "sglang-v0.5.14-complete"}
BATCHES = ("reviewed-reference-batch1.tsv", "reviewed-sampling-batch1.tsv",
           "reviewed-tool-parser-batch1.tsv", "reviewed-tool-parser-batch2.tsv",
           "reviewed-ngram-draft-batch1.tsv")
FIELDS = ("snapshot", "path", "test_name", "line", "unit_id", "property",
          "oracle_source", "conditions", "rule_id")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["snapshot"], row["path"], row["test_name"], row["line"]


def nodes(source: bytes) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result = {}
    for outer in ast.parse(source.decode("utf-8")).body:
        if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[outer.name] = outer
        elif isinstance(outer, ast.ClassDef):
            for item in outer.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{outer.name}::{item.name}"] = item
    return result


def main() -> dict[str, object]:
    candidates = {key(r): r for r in read(HERE / "candidates.tsv")}
    assert len(candidates) == 6330
    green = set()
    for name in ("confirmed-property-crosswalk.tsv", "direct-property-crosswalk.tsv",
                 "inherited-property-crosswalk.tsv"):
        green.update(key(r) for r in read(PILOT / name))
    green.update(key(r) for r in read(REPO / "semantic_space/oracle-review-queue.tsv")
                 if r["source_review_status"] == "mapped" and r["source_reviewed_units"])
    assert len(green) == 5330
    units = {u["id"] for u in json.loads((REPO / "semantic_space/space_seed.json").read_text())["units"]}
    reviewed = []
    for filename in BATCHES:
        for row in read(HERE / filename):
            assert tuple(row) == FIELDS, filename
            reviewed.append(row)
    keys = [key(row) for row in reviewed]
    assert len(keys) == len(set(keys)) and not (set(keys) & green)
    source_cache = {}
    for row in reviewed:
        k = key(row)
        candidate = candidates[k]
        assert row["unit_id"] in units and row["property"] and row["conditions"] and row["rule_id"]
        file_key = row["snapshot"], row["path"]
        if file_key not in source_cache:
            source = (ROOTS[row["snapshot"]] / row["path"]).read_bytes()
            assert hashlib.sha256(source).hexdigest() == candidate["source_sha256"], file_key
            source_cache[file_key] = nodes(source)
        node = source_cache[file_key][row["test_name"]]
        assert node.lineno == int(row["line"]), k
        available = set(candidate["oracle_lines"].split(";"))
        locations = row["oracle_source"].split(";")
        assert locations and all(location.startswith(row["path"] + ":") for location in locations), k
        sites = {location.rsplit(":", 1)[-1] for location in locations}
        assert sites <= available and all(node.lineno < int(line) <= node.end_lineno for line in sites), k
    result = {
        "green_before": len(green), "local_reviewed": len(reviewed),
        "files": len(source_cache),
        "scope": "First five source-review batches; complete integration counts are in reviewed-supplement-summary.json",
        "batches": {name: len(read(HERE / name)) for name in BATCHES},
        "units": dict(sorted(Counter(r["unit_id"] for r in reviewed).items())),
    }
    (HERE / "local-review-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
