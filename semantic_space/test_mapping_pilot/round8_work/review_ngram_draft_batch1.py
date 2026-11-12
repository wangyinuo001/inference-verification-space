"""Review n-gram draft candidate checks after refining the verification space."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "../../experiments/third_party/reference_repos"
REPOS = {"vllm": ROOT / "vllm-20261105", "sglang": ROOT / "sglang-v0.5.14-complete"}
FIELDS = ("snapshot", "path", "test_name", "line", "unit_id", "property",
          "oracle_source", "conditions", "rule_id")
PATHS = {
    ("vllm", "tests/v1/spec_decode/test_ngram.py"):
        (2, "d2b7498396c0872c33a438afbdc52b23654b1149d18c2e9b1022b14d81715715"),
    ("sglang", "test/registered/unit/spec/test_ngram_corpus.py"):
        (50, "5521ec083dd5e670bfbc912ecd2bc47c20decd87b66727bfe514da01cf1251a8"),
}
EXCLUDED_CLASSES = {"TestNgramCorpusMultiSam", "TestMultiSamHttpMock"}
EXCLUDED_TEST = "TestNgramCorpusExternalSam::test_external_corpus_iterator_rejects_oversized_corpus"


def nodes(source: bytes) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(source.decode("utf-8"))
    result = {node.name: node for node in tree.body
              if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef):
            result.update({f"{cls.name}::{node.name}": node for node in cls.body
                           if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))})
    return result


def main() -> dict[str, int]:
    with (HERE / "candidates.tsv").open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream, delimiter="\t"))
    selected = []
    excluded = []
    for (snapshot, path), (count, digest) in PATHS.items():
        source = (REPOS[snapshot] / path).read_bytes()
        assert hashlib.sha256(source).hexdigest() == digest, path
        group = [r for r in candidates if (r["snapshot"], r["path"]) == (snapshot, path)]
        assert len(group) == count, (path, len(group), count)
        test_nodes = nodes(source)
        for row in group:
            name = row["test_name"]
            if name.split("::")[0] in EXCLUDED_CLASSES or name == EXCLUDED_TEST:
                excluded.append(name)
                continue
            assert row["proposed_unit"] == "spec.draft_candidate_binding"
            assert row["source_sha256"] == digest
            node = test_nodes[name]
            assert node.lineno == int(row["line"]), (path, name)
            sites = {int(line) for line in row["oracle_lines"].split(";")}
            assert sites and all(node.lineno < line <= node.end_lineno for line in sites), (path, name)
            doc = ast.get_docstring(node)
            summary = doc.splitlines()[0] if doc else name.split("::")[-1].removeprefix("test_").replace("_", " ")
            selected.append({
                "snapshot": snapshot, "path": path, "test_name": name, "line": row["line"],
                "unit_id": "spec.draft_candidate_binding",
                "property": summary + ". Direct source checks: " + row["checks"],
                "oracle_source": ";".join(f"{path}:{line}" for line in sorted(sites)),
                "conditions": "Pinned n-gram corpus/history, matching mode or window, request state and draft budget in the source test; local draft IDs, length or mask only, not target acceptance or final output.",
                "rule_id": "round8_ngram_draft_batch1_source_review",
            })
    assert len(excluded) == 15 and len(selected) == 37
    keys = [(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in selected]
    assert len(keys) == len(set(keys))
    output = HERE / "reviewed-ngram-draft-batch1.tsv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda r: (r["snapshot"], r["path"], int(r["line"]))))
    result = {"reviewed_local_only": len(selected), "excluded_management_or_admission": len(excluded),
              "space_unit": "spec.draft_candidate_binding", "published_green_added": 0}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
