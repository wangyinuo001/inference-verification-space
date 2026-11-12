"""Independently reopen candidate source anchors; semantic promotion remains separate."""

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REPOS = PROJECT / "../../experiments/third_party/reference_repos"
ROOTS = {"vllm": REPOS / "vllm-20261105", "sglang": REPOS / "sglang-v0.5.14-complete"}

with (HERE / "candidates.tsv").open(encoding="utf-8", newline="") as stream:
    candidates = list(csv.DictReader(stream, delimiter="\t"))

seen = set()
counts = Counter()
source_cache = {}
for row in candidates:
    key = (row["snapshot"], row["path"], row["test_name"], row["line"])
    assert key not in seen, key
    seen.add(key)
    file_key = key[:2]
    if file_key not in source_cache:
        source = ROOTS[row["snapshot"]] / row["path"]
        raw = source.read_bytes()
        tree = ast.parse(raw.decode("utf-8"))
        nodes = {f"{outer.name}::{node.name}": node
                 for outer in tree.body if isinstance(outer, ast.ClassDef)
                 for node in outer.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        nodes.update({node.name: node for node in tree.body
                      if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))})
        source_cache[file_key] = (hashlib.sha256(raw).hexdigest(), nodes)
    digest, nodes = source_cache[file_key]
    assert digest == row["source_sha256"], file_key
    node = nodes[row["test_name"]]
    assert str(node.lineno) == row["line"], key
    body_lines = {item.lineno for item in ast.walk(node) if hasattr(item, "lineno")}
    oracle_lines = [int(line) for line in row["oracle_lines"].split(";")]
    assert oracle_lines and all(line in body_lines for line in oracle_lines), key
    assert row["checks"] and row["proposed_unit"] and row["status"] == "candidate_only", key
    counts[row["snapshot"]] += 1

result = {"candidate_rows_verified": len(candidates), "by_snapshot": dict(counts),
          "status": "source_anchors_only_not_semantic_promotion"}
(HERE / "candidate-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False))
