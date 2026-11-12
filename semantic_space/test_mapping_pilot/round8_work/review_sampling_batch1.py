"""Freeze source-reviewed local custom-logit-processor checks, not public green."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_PATH = "test/registered/unit/sampling/test_custom_logit_processor.py"
SOURCE = HERE.parents[2] / "../../experiments/third_party/reference_repos/sglang-v0.5.14-complete" / SOURCE_PATH
SOURCE_HASH = "2f47e6241930cd51fac2f99de7bffe77fdc499b144e620e02af024aa081a5ddc"
FIELDS = ("snapshot", "path", "test_name", "line", "unit_id", "property",
          "oracle_source", "conditions", "rule_id")
APPROVED_CLASSES = {"TestDisallowedTokensLogitsProcessor", "TestThinkingBudgetLogitProcessor",
                    "TestDeepseekOCRNoRepeatNGramLogitProcessor"}


def main() -> dict[str, int]:
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SOURCE_HASH
    nodes = {f"{cls.name}::{node.name}": node
             for cls in ast.parse(raw.decode("utf-8")).body if isinstance(cls, ast.ClassDef)
             for node in cls.body if isinstance(node, ast.FunctionDef)}
    with (HERE / "candidates.tsv").open(encoding="utf-8", newline="") as stream:
        candidates = [r for r in csv.DictReader(stream, delimiter="\t")
                      if r["snapshot"] == "sglang" and r["path"] == SOURCE_PATH
                      and r["test_name"].split("::")[0] in APPROVED_CLASSES]
    assert len(candidates) == 27
    rows = []
    for row in candidates:
        node = nodes[row["test_name"]]
        assert node.lineno == int(row["line"])
        sites = {int(n) for n in row["oracle_lines"].split(";")}
        assert sites and all(node.lineno < site <= node.end_lineno for site in sites)
        calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
        assert any(ast.unparse(call.func).endswith(("processor", "proc")) for call in calls), row["test_name"]
        doc = ast.get_docstring(node)
        assert doc, row["test_name"]
        rows.append({
            "snapshot": "sglang", "path": SOURCE_PATH, "test_name": row["test_name"],
            "line": row["line"], "unit_id": "sampling.policy",
            "property": doc.splitlines()[0] + " Direct checks: " + row["checks"],
            "oracle_source": ";".join(f"{SOURCE_PATH}:{n}" for n in sorted(sites)),
            "conditions": "Pinned local custom-logit-processor input logits, request-token history and parameter values in the source test; no model or server execution asserted.",
            "rule_id": "round8_sampling_batch1_source_review",
        })
    assert len({(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in rows}) == len(rows)
    output = HERE / "reviewed-sampling-batch1.tsv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"reviewed_local_only": len(rows), "published_green_added": 0}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
