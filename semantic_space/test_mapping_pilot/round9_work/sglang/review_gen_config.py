# -*- coding: utf-8 -*-
"""Review image/video generation server configuration test family."""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
PATH="python/sglang/multimodal_gen/test/unit/test_server_args.py"
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"/PATH
COLS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]
HELPERS={"self._from_dict_without_model_resolution":100,"_from_dict_without_model_resolution":87,"self._from_dict_with_pipeline_config":626,"self._from_dict_with_task_type":670,"self._from_dict":1610,"self._resolve":502,"ServerArgs.from_dict":87}

def main():
    text=SOURCE.read_text(encoding="utf-8")
    tree=ast.parse(text)
    entries=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"))
    keep=[];mapped=[]
    for row in entries:
        if row["path"]!=PATH:keep.append(row);continue
        line=int(row["line"])
        nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno<=line<=n.end_lineno]
        node=min(nodes,key=lambda n:n.end_lineno-n.lineno) if nodes else None
        if node is None:keep.append(row);continue
        helper_calls=[(n.lineno,ast.unparse(n.func),ast.unparse(n)) for n in ast.walk(node) if isinstance(n,ast.Call) and ast.unparse(n.func) in HELPERS]
        assertions=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith("assert")]
        assertions=sorted(assertions,key=lambda n:n.lineno)
        if not helper_calls or not assertions:
            row["triage_status"]="needs_gen_config_helper_review"
            row["review_reason"]="No recognized ServerArgs resolution helper with direct unittest assertion; inspect alternate config/parser/path logic."
            keep.append(row);continue
        call=sorted(helper_calls)[0]
        oracle_lines=";".join(f"{PATH}:{n.lineno}" for n in assertions[:8])
        oracle_text=" | ".join(" ".join((ast.get_source_segment(text,n) or "").split())[:160] for n in assertions[:5])
        rejects=any(n.func.attr.startswith("assertRaises") for n in assertions)
        unit="deployment.configuration_admissibility" if rejects else "deployment.effective_configuration"
        basis="Specified unsupported configuration is rejected as asserted: " if rejects else "For the generation pipeline's specified options and mocked hardware/configuration, resolved fields satisfy this case's assertions: "
        mapped.append({"snapshot":"sglang","path":PATH,"test_name":row["test_name"],"line":row["line"],"unit_id":unit,"property":basis+oracle_text,"oracle_source":oracle_lines,"conditions":f"Image/video generation extension, not text-request inference. Resolver call {PATH}:{call[0]}: {call[2][:250]}; helper definition {PATH}:{HELPERS[call[1]]}. Values are configuration results, not actual model execution.","rule_id":"sglang_multimodal_gen_server_args_resolution"})
    with (HERE/"crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=COLS).writerows(mapped)
    with (HERE/"pending.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,delimiter="\t",fieldnames=entries[0].keys());w.writeheader();w.writerows(keep)
    evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"path":PATH,"sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"mapped_existing":len(mapped),"exceptions":sum(r["path"]==PATH for r in keep),"review_method":"Read all configuration helpers and product imports; accepted tests call recognized ServerArgs resolver and directly assert resulting fields or expected exception. Exact call arguments/assertion lines retained."});evidence["total_pending_after_review"]=len(keep)
    (HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print("mapped",len(mapped),"remaining",sum(r["path"]==PATH for r in keep))

if __name__=="__main__":main()
