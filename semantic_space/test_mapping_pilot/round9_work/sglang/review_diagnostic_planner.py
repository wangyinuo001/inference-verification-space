# -*- coding: utf-8 -*-
"""Source-linked diagnostic unsharding planner family review."""

import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PATH = "test/registered/debug_utils/comparator/aligner/unsharder/test_planner.py"
SOURCE = ROOT / "experiments/third_party/reference_repos/sglang-v0.5.14-complete" / PATH
TARGETS = {
    "compute_unsharder_plan": ("diagnostic.parallel_tensor_reconstruction_plan", "The planned sequence and grouping of parallel tensor reconstruction operations match the asserted axis, rank grouping, operation parameters or invalid-layout rejection."),
    "_compute_dependent_axes": ("diagnostic.parallel_axis_dependency", "Declared parallel-axis metadata yields the asserted dependent-axis relation."),
    "_is_dependent_axis": ("diagnostic.parallel_axis_dependency", "Declared parallel-axis metadata yields the asserted dependent-axis predicate."),
    "_is_jointly_determined": ("diagnostic.parallel_axis_dependency", "Declared parent axes jointly determine the child axis exactly in the asserted cases."),
    "_validate_explicit_replicated": ("diagnostic.parallel_axis_dependency", "Replicated-axis declarations satisfy the asserted admissibility or rejection condition."),
    "_validate_replicated_axes_orthogonal": ("diagnostic.parallel_axis_dependency", "Replicated parallel axes satisfy the asserted orthogonality condition or are rejected."),
}
COLS = ["snapshot", "path", "test_name", "line", "unit_id", "property", "oracle_source", "conditions", "rule_id"]


def node_for_line(tree, line):
    hits = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= line <= n.end_lineno]
    return min(hits, key=lambda n: n.end_lineno - n.lineno) if hits else None


def main():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    entries = list(csv.DictReader((HERE / "pending.tsv").open(newline=""), delimiter="\t"))
    keep, reviewed, exceptions = [], [], []
    for row in entries:
        if row["path"] != PATH:
            keep.append(row); continue
        node = node_for_line(tree, int(row["line"]))
        if node is None:
            row["review_reason"] = "Test definition not found in parsed source."
            keep.append(row); exceptions.append(row["test_name"]); continue
        calls = [(n.lineno, n.func.id) for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in TARGETS]
        assertion_lines = [(n.lineno, ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), n) or "") for n in ast.walk(node) if isinstance(n, ast.Assert)]
        assertion_lines += [(n.lineno, ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), n) or "") for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr.startswith("assert")]
        assertion_lines = sorted(set(assertion_lines))
        if len({name for _, name in calls}) != 1 or not assertion_lines:
            row["triage_status"] = "needs_manual_planner_link_review"
            row["review_reason"] = f"Planner calls={calls}; direct assertion count={len(assertion_lines)}. Need relate each asserted value to a specific planner result."
            keep.append(row); exceptions.append(row["test_name"]); continue
        target = calls[0][1]
        unit, basis = TARGETS[target]
        oracle_source = ";".join(f"{PATH}:{line}" for line, _ in assertion_lines[:8])
        assertion_text = " | ".join(" ".join(text.split())[:170] for _, text in assertion_lines[:4])
        reviewed.append({"snapshot": "sglang", "path": PATH, "test_name": row["test_name"], "line": row["line"],
                         "unit_id": unit,
                         "property": f"{basis} This case asserts: {assertion_text}",
                         "oracle_source": oracle_source,
                         "conditions": f"Pinned SGLang 0.5.14 diagnostic comparator; target={target}; test's explicit dim specs and parallel_infos; this checks a reconstruction plan, not an actual distributed model execution.",
                         "rule_id": "sglang_diagnostic_unsharder_planner_" + target})
    with (HERE / "proposed-crosswalk.tsv").open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLS, delimiter="\t").writerows(reviewed)
    with (HERE / "pending.tsv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=entries[0].keys(), delimiter="\t");w.writeheader();w.writerows(keep)
    needs = json.loads((HERE / "needs-space.json").read_text())
    needs.extend([
        {"id":"diagnostic.parallel_tensor_reconstruction_plan","kind":"obligation","phase":"diagnostic","state_scope":"diagnostic_artifact","name":"Parallel tensor reconstruction plan follows declared sharding","property":"Given tensor dimension annotations and rank topology, a diagnostic comparator chooses a reconstruction sequence and rank groups consistent with the declared sharding, and rejects inconsistent layouts.","conditions":["execution_layout"],"sglang_evidence":[f"{PATH}:24","python/sglang/srt/debug_utils/comparator/aligner/unsharder/planner.py"],"vllm_evidence":[],"difference":"No matching vLLM diagnostic reconstruction test located in this batch; this is an SGLang-specific auxiliary diagnostic obligation, not whole-request correctness."},
        {"id":"diagnostic.parallel_axis_dependency","kind":"obligation","phase":"diagnostic","state_scope":"diagnostic_artifact","name":"Parallel axis dependency and replication declarations are consistent","property":"Given dimension and rank annotations, dependent and replicated axes are identified consistently; invalid dependent or overlapping replication claims are rejected.","conditions":["execution_layout"],"sglang_evidence":[f"{PATH}:1540","python/sglang/srt/debug_utils/comparator/aligner/unsharder/planner.py"],"vllm_evidence":[],"difference":"Auxiliary comparator planning; no corresponding vLLM obligation established in this batch."}])
    (HERE / "needs-space.json").write_text(json.dumps(needs, ensure_ascii=False, indent=2)+"\n")
    evidence=json.loads((HERE / "evidence.json").read_text())
    evidence.setdefault("reviewed_clusters",[]).append({"path":PATH,"sha256":digest,"proposed_new_unit":len(reviewed),"exceptions":exceptions,"review_method":"Full pinned source import and per-definition AST: one direct planner target call plus direct assertion or expected exception; exact assertion lines retained. Plan property only, not execution correctness."})
    evidence["total_pending_after_review"]=len(keep)
    (HERE / "evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print("reviewed",len(reviewed),"exceptions",len(exceptions),"pending",len(keep))


if __name__=="__main__":main()
