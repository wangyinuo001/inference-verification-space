"""Publish audited source reviews without waiting for an arbitrary batch quota."""
from __future__ import annotations

import csv
import json
import ast
import hashlib
from pathlib import Path

from round8_work.audit_local_review import BATCHES, FIELDS, main as audit

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reviewed-supplement-crosswalk.tsv"
EXTRAS = ("reviewed-parser-batch3", "reviewed-reference-batch2", "reviewed-reference-batch3",
          "reviewed-sglang-runtime-batch", "reviewed-rendering-batch1",
          "reviewed-sglang-logit-transform-proposal")


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def key(row):
    return tuple(row[k] for k in ("snapshot", "path", "test_name", "line"))


def extra_rows(stem):
    path = HERE / "round8_work" / (stem + ".tsv")
    if not path.exists():
        return []
    evidence = json.loads(path.with_name(stem + "-evidence.json").read_text())
    files = evidence.get("file_reviews")
    if files is None:
        files = list(evidence["files"].values()) if "files" in evidence else evidence["decisions"]
    hashes = {(f["snapshot"], f["path"]): f["source_sha256"] for f in files}
    project = HERE.parents[3]
    roots = {"vllm": project / "experiments/third_party/reference_repos/vllm-20261105",
             "sglang": project / "experiments/third_party/reference_repos/sglang-v0.5.14-complete"}
    index = {key(r) for r in read(HERE / "full-scan-index.tsv")}
    units = {u["id"] for u in json.loads((HERE.parent / "space_seed.json").read_text())["units"]}
    cache = {}; rows = read(path)
    for row in rows:
        assert tuple(row) == FIELDS and key(row) in index, key(row)
        assert row["unit_id"] in units and all(row[f] for f in ("property", "conditions", "rule_id")), key(row)
        source_key = row["snapshot"], row["path"]
        if source_key not in cache:
            raw = (roots[row["snapshot"]] / row["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == hashes[source_key], source_key
            cache[source_key] = {n.lineno: n for n in ast.walk(ast.parse(raw.decode()))
                                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        node = cache[source_key][int(row["line"])]
        assert node.name == row["test_name"].split("::")[-1], key(row)
        for site in row["oracle_source"].split(";"):
            source_path, line = site.strip().rsplit(":", 1)
            assert source_path == row["path"] and node.lineno < int(line) <= node.end_lineno, (key(row), site)
    return rows


def main() -> dict:
    result = audit()
    rows = []
    for filename in BATCHES:
        with (HERE / "round8_work" / filename).open(encoding="utf-8", newline="") as f:
            rows.extend(csv.DictReader(f, delimiter="\t"))
    batches = {filename: len(read(HERE / "round8_work" / filename)) for filename in BATCHES}
    for stem in EXTRAS:
        extra = extra_rows(stem)
        if extra:
            batches[stem + ".tsv"] = len(extra)
            rows.extend(extra)
    keys = [key(r) for r in rows]
    assert len(keys) == len(set(keys)), "Duplicate supplementary review"
    existing = set()
    for filename in ("confirmed-property-crosswalk.tsv", "direct-property-crosswalk.tsv", "inherited-property-crosswalk.tsv"):
        existing.update(key(r) for r in read(HERE / filename))
    existing.update(key(r) for r in read(HERE.parent / "oracle-review-queue.tsv")
                    if r["source_review_status"] == "mapped" and r["source_reviewed_units"])
    assert not existing.intersection(keys), "Supplement overlaps a previously reviewed entry"
    rows.sort(key=lambda r: (r["snapshot"], r["path"], int(r["line"]), r["test_name"]))
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    summary = {"source_reviewed_additions": len(rows), "files": len({(r['snapshot'], r['path']) for r in rows}),
               "batches": batches,
               "basis": "Pinned source review, not runtime pass or whole-space completeness"}
    (HERE / "reviewed-supplement-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    pending = []
    for stem in EXTRAS:
        evidence_path = HERE / "round8_work" / (stem + "-evidence.json")
        if not evidence_path.exists():
            continue
        evidence = json.loads(evidence_path.read_text())
        for item in evidence.get("entries", evidence.get("decisions", [])):
            unresolved = item.get("status", item.get("decision", "")) in {"needs_refinement", "needs_refinement_or_unconfirmed"}
            unresolved |= "unit_id" in item and item["unit_id"] is None
            if not unresolved:
                continue
            item = {**item, "line": str(item.get("line", item.get("definition_line")))}
            if key(item) in set(keys) | existing:
                continue
            pending.append({**{f: str(item[f]) for f in ("snapshot", "path", "test_name", "line")},
                            "reason": item.get("reason") or item.get("review_reason") or "已核对源码，仍需明确检查性质或细分对应单元。",
                            "source_sha256": item["source_sha256"],
                            "evidence": "round8_work/" + evidence_path.name})
    assert len(pending) == len({key(r) for r in pending})
    (HERE / "reviewed-supplement-pending.json").write_text(json.dumps(pending, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
