"""Create local source-reviewed rows for two new text-level obligations."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from review_tool_parser_batch1 import FIELDS, REPOS, rows, test_nodes

HERE = Path(__file__).resolve().parent
UNIT_IDS = {"input.prompt_text_rendering", "input.media_marker_rendering"}
REVIEW = {
    "tests/renderers/test_inkling.py": "Direct checks of decoded prompt text or media marker text under fixed Inkling messages; individual subproperties vary by assertion.",
    "tests/tokenizers_/test_deepseek_v4.py": "DeepSeek-V4 prompt-string assertions bind role, tool, continuation and image marker conditions; tokenize=False for these checks.",
    "tests/tokenizers_/test_deepseek_v41.py": "DeepSeek-V4.1 prompt-string and image-marker assertions include reference fixture, continuation and textual separator cases.",
}


def main() -> dict[str, object]:
    refinements = json.loads((HERE / "reviewed-parser-batch3-refinement.json").read_text(encoding="utf-8"))
    selected = [r for r in refinements["entry_suggestions"] if r["proposed_unit"] in UNIT_IDS]
    assert len(selected) == 28
    assert Counter(r["proposed_unit"] for r in selected) == {
        "input.prompt_text_rendering": 24, "input.media_marker_rendering": 4}
    ontology = {u["id"] for u in json.loads(
        (HERE.parents[1] / "space_seed.json").read_text(encoding="utf-8"))["units"]}
    assert UNIT_IDS <= ontology
    review_rows, evidence_rows, file_reviews = [], [], []
    for path in REVIEW:
        source = (REPOS["vllm"] / path).read_bytes()
        sha = hashlib.sha256(source).hexdigest()
        nodes = test_nodes(source)
        group = [r for r in selected if r["path"] == path]
        file_reviews.append({"snapshot": "vllm", "path": path, "source_sha256": sha,
                             "reviewed_entries": len(group), "review": REVIEW[path]})
        for entry in group:
            assert entry["source_sha256"] == sha
            node = nodes[entry["test_name"]]
            assert node.lineno == entry["line"]
            sites = [int(loc.rsplit(":", 1)[1]) for loc in entry["oracle_source"].split(";")]
            assert sites and all(node.lineno < n <= node.end_lineno for n in sites)
            assert all(any(isinstance(x, (ast.Assert, ast.With, ast.AsyncWith))
                           and x.lineno <= n <= x.end_lineno for x in ast.walk(node))
                       for n in sites)
            summary = (ast.get_docstring(node) or entry["test_name"].split("::")[-1]
                       .removeprefix("test_").replace("_", " ")).splitlines()[0]
            condition = ("Fixed renderer format, message content/order and controls in cited source; "
                         "the mapped property is limited to the listed text assertion. "
                         "No token-ID sequence is inferred.")
            if entry["proposed_unit"] == "input.media_marker_rendering":
                condition += " Media payload binding is not inferred."
            review_rows.append({
                "snapshot": "vllm", "path": path, "test_name": entry["test_name"],
                "line": str(entry["line"]), "unit_id": entry["proposed_unit"],
                "property": summary + ". Source asserts: " + entry["asserted_checks"],
                "oracle_source": entry["oracle_source"], "conditions": condition,
                "rule_id": "round8_rendering_batch1_source_review",
            })
            evidence_rows.append({"snapshot": "vllm", "path": path,
                                  "test_name": entry["test_name"], "line": entry["line"],
                                  "source_sha256": sha, "unit_id": entry["proposed_unit"],
                                  "oracle_source": entry["oracle_source"],
                                  "asserted_checks": entry["asserted_checks"],
                                  "review_status": "local_source_review_only"})
    assert len(review_rows) == 28
    review_rows.sort(key=lambda r: (r["path"], int(r["line"])))
    with (HERE / "reviewed-rendering-batch1.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(review_rows)
    evidence = {"status": "local_source_review_only", "entry_count": 28,
                "unit_counts": dict(Counter(r["unit_id"] for r in review_rows)),
                "file_reviews": file_reviews, "entries": evidence_rows,
                "semantic_boundary": "Rendered prompt text precedes token-ID encoding; textual media markers precede payload processing and tensor binding.",
                "public_crosswalk_modified": False}
    (HERE / "reviewed-rendering-batch1-evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reviewed": len(review_rows), "files": len(file_reviews),
                      "units": evidence["unit_counts"]}, ensure_ascii=False))
    return evidence


if __name__ == "__main__":
    main()
