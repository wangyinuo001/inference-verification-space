# -*- coding: utf-8 -*-
"""Confirm direct target rejection checks in planner test contexts."""
import ast
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
PATH="test/registered/debug_utils/comparator/aligner/unsharder/test_planner.py"
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"/PATH
TARGETS={"compute_unsharder_plan":"diagnostic.parallel_tensor_reconstruction_plan","_validate_explicit_replicated":"diagnostic.parallel_axis_dependency","_validate_replicated_axes_orthogonal":"diagnostic.parallel_axis_dependency","_compute_dependent_axes":"diagnostic.parallel_axis_dependency","_is_dependent_axis":"diagnostic.parallel_axis_dependency","_is_jointly_determined":"diagnostic.parallel_axis_dependency"}

def main():
    tree=ast.parse(SOURCE.read_text(encoding="utf-8"))
    entries=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"))
    keep=[]; out=[]
    for row in entries:
        if row["path"]!=PATH:keep.append(row);continue
        line=int(row["line"])
        nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno<=line<=n.end_lineno]
        node=min(nodes,key=lambda n:n.end_lineno-n.lineno) if nodes else None
        matches=[]
        for w in ast.walk(node) if node else []:
            if not isinstance(w,ast.With):continue
            for item in w.items:
                expr=item.context_expr
                if not (isinstance(expr,ast.Call) and isinstance(expr.func,ast.Attribute) and expr.func.attr=="raises"):continue
                errors=[ast.unparse(arg) for arg in expr.args]
                errors += [f"{kw.arg}={ast.unparse(kw.value)}" for kw in expr.keywords]
                calls=[c.func.id for statement in w.body for c in ast.walk(statement) if isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id in TARGETS]
                if len(set(calls))==1:matches.append((w.lineno,calls[0],", ".join(errors)))
        if len(matches)!=1:keep.append(row);continue
        oracle_line,target,error=matches[0]
        out.append({"snapshot":"sglang","path":PATH,"test_name":row["test_name"],"line":row["line"],"unit_id":TARGETS[target],"property":f"The specified parallel-axis or shard-layout input is rejected by {target} with {error}.","oracle_source":f"{PATH}:{oracle_line}","conditions":"Pinned SGLang 0.5.14 comparator planning; explicit per-case dim specs and rank metadata; rejection only, not actual tensor reconstruction.","rule_id":"sglang_diagnostic_planner_expected_rejection"})
    cols=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]
    with (HERE/"proposed-crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=cols).writerows(out)
    with (HERE/"pending.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,delimiter="\t",fieldnames=entries[0].keys());w.writeheader();w.writerows(keep)
    evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"path":PATH,"expected_rejections":len(out),"rule":"AST `with pytest.raises` contains exactly one call to the imported planner target; expected exception type and match preserved."});evidence["total_pending_after_review"]=len(keep)
    (HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print("expected rejection",len(out),"remaining planner",sum(r["path"]==PATH for r in keep))

if __name__=="__main__":main()
