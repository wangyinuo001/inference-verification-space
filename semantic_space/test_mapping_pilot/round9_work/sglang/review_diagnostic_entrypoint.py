# -*- coding: utf-8 -*-
"""Source-linked comparator entrypoint helper family."""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
PATH="test/registered/debug_utils/comparator/test_entrypoint.py"
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"/PATH
COLS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]

def main():
    text=SOURCE.read_text(encoding="utf-8")
    tree=ast.parse(text)
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    entries=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"))
    keep=[];out=[]
    for row in entries:
        if row["path"]!=PATH:keep.append(row);continue
        line=int(row["line"])
        nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno<=line<=n.end_lineno]
        node=min(nodes,key=lambda n:n.end_lineno-n.lineno) if nodes else None
        calls=[n.lineno for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="_run_and_parse"] if node else []
        assertions=sorted([n for n in ast.walk(node) if isinstance(n,ast.Assert)],key=lambda n:n.lineno) if node else []
        if not calls or not assertions:
            row["triage_status"]="needs_diagnostic_helper_review"
            row["review_reason"]="No direct _run_and_parse call with a test-body assertion; inspect alternate helper or exception context."
            keep.append(row);continue
        excerpts=[" ".join((ast.get_source_segment(text,a) or "").split())[:180] for a in assertions[:5]]
        out.append({"snapshot":"sglang","path":PATH,"test_name":row["test_name"],"line":row["line"],
                    "unit_id":"diagnostic.comparator_report_relation",
                    "property":"For the test's constructed baseline and target dumps, comparator records, status or report satisfy the asserted relation: "+" | ".join(excerpts),
                    "oracle_source":";".join(f"{PATH}:{a.lineno}" for a in assertions[:8]),
                    "conditions":f"Pinned SGLang 0.5.14; `_run_and_parse` at {PATH}:2775 calls parse_args/run and parses emitted JSON records; explicit dump tensors, CLI options and assertions in this case. This is diagnostic report behavior, not inference-request correctness.",
                    "rule_id":"sglang_diagnostic_comparator_entrypoint_helper"})
    with (HERE/"proposed-crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=COLS).writerows(out)
    with (HERE/"pending.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,delimiter="\t",fieldnames=entries[0].keys());w.writeheader();w.writerows(keep)
    needs=json.loads((HERE/"needs-space.json").read_text())
    needs.append({"id":"diagnostic.comparator_report_relation","kind":"obligation","phase":"diagnostic","state_scope":"diagnostic_artifact","name":"Comparator report reflects declared baseline/target dump relation","property":"For specified baseline and target diagnostic dumps and comparison options, emitted comparison records, summary, exit status and optional report follow the measured tensor relation and declared filtering/alignment policy.","conditions":["execution_layout"],"sglang_evidence":[f"{PATH}:2775", "python/sglang/srt/debug_utils/comparator/entrypoint.py"],"vllm_evidence":[],"difference":"Diagnostic comparator extension; a vLLM analogue has not been established in this source batch."})
    (HERE/"needs-space.json").write_text(json.dumps(needs,ensure_ascii=False,indent=2)+"\n")
    evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"path":PATH,"sha256":digest,"proposed_new_unit":len(out),"exception_count":126-len(out),"review_method":"Full common helper at line 2775 read: parse_args then run then parse JSON. Every accepted entry calls helper and asserts result in its own body; exact assertion lines recorded."});evidence["total_pending_after_review"]=len(keep)
    (HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print("reviewed",len(out),"remaining",sum(r["path"]==PATH for r in keep))

if __name__=="__main__":main()
