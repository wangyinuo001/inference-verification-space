"""Separate explicit numerical comparisons for closer semantic review."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATTERN = re.compile(r"(?:assert_close|allclose|assert_array_equal|assert_equal|torch\.equal)\(", re.I)
REFERENCE = re.compile(r"\b(?:ref|reference|expected|baseline|target|want|truth|torch_output|unfused)\b|\bref_", re.I)
FAMILIES = {"kernel", "attention", "layers", "model"}

with (HERE / "candidates.tsv").open(encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))

selected = []
for row in rows:
    if row["family"] not in FAMILIES:
        continue
    comparisons = [expr for expr in row["checks"].split(" || ")
                   if PATTERN.search(expr) and REFERENCE.search(expr)]
    if not comparisons:
        continue
    selected.append({**row, "reference_comparisons": " || ".join(comparisons),
                     "status": "reference_candidate_only"})

assert len(selected) == len({(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in selected})
output = HERE / "reference-candidates.tsv"
fields = list(selected[0])
with output.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(selected)

result = {"source_anchored_reference_candidates": len(selected),
          "files": len({(r["snapshot"], r["path"]) for r in selected}),
          "by_family": dict(Counter(r["family"] for r in selected)),
          "status": "candidate_only_not_green"}
(HERE / "reference-summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
