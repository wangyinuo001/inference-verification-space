# -*- coding: utf-8 -*-
"""Source-linked run_eval score threshold family across SGLang files."""
import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"
COLS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]

def eval_assigned_names(node):
    out=[]
    for n in ast.walk(node):
        if not isinstance(n,(ast.Assign,ast.AnnAssign)):continue
        value=n.value
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=="run_eval":
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            names=[x.id for t in targets for x in ast.walk(t) if isinstance(x,ast.Name)]
            out.append((n.lineno,names,ast.unparse(value)))
    return out

def score_aliases(node,metric_names):
    aliases=set(metric_names)
    for n in ast.walk(node):
        if not isinstance(n,ast.Assign):continue
        if not any(isinstance(t,ast.Name) for t in n.targets):continue
        v=ast.unparse(n.value)
        if any(v.startswith(f"{m}['score']") or v.startswith(f'{m}["score"]') or v.startswith(f"{m}.get('score'") or v.startswith(f'{m}.get("score"') or v.startswith(f"{m}['accuracy']") for m in metric_names):
            aliases.update(t.id for t in n.targets if isinstance(t,ast.Name))
    return aliases

def main():
    rows=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"))
    grouped=defaultdict(list)
    for r in rows:grouped[r["definition_path"]].append(r)
    mapped=[];selected=set();files={};sample_exceptions=[]
    for path,entries in grouped.items():
        source=SOURCE/path
        try:text=source.read_text(encoding="utf-8");tree=ast.parse(text)
        except (OSError,SyntaxError,UnicodeDecodeError):continue
        imports=[n for n in tree.body if isinstance(n,ast.ImportFrom) and any(alias.name=="run_eval" for alias in n.names)]
        import_modules={n.module for n in imports}
        if not import_modules.intersection({"sglang.test.run_eval","sglang.test.few_shot_gsm8k"}):continue
        funcs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
        for row in entries:
            if row["path"]!=path:continue  # inherited subclass conditions require a separate pass
            line=int(row["definition_line"])
            hits=[n for n in funcs if n.lineno<=line<=n.end_lineno]
            if not hits:continue
            node=min(hits,key=lambda n:n.end_lineno-n.lineno)
            assignments=eval_assigned_names(node)
            if len(assignments)!=1:continue
            call_line,metric_names,call_expr=assignments[0]
            aliases=score_aliases(node,metric_names)
            checks=[]
            for n in ast.walk(node):
                if isinstance(n,ast.Assert):
                    expr=ast.unparse(n.test)
                    if any(m in expr for m in aliases) and any(k in expr for k in ("score","accuracy","threshold","thres","min_score")):
                        checks.append((n.lineno," ".join((ast.get_source_segment(text,n) or "").split())))
                elif isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ("assertGreater","assertGreaterEqual","assertLess","assertLessEqual","assertEqual"):
                    expr=ast.unparse(n)
                    if any(m in expr for m in aliases) and any(k in expr for k in ("score","accuracy","threshold","thres","min_score")):
                        checks.append((n.lineno," ".join((ast.get_source_segment(text,n) or "").split())))
            checks=sorted(set(checks))
            if not checks:
                sample_exceptions.append((path,row["test_name"],"run_eval result not directly score-threshold asserted"));continue
            # The assertion records the exact test standard; run_eval computes a task score.
            oracle=";".join(f"{path}:{ln}" for ln,_ in checks)
            property_text="Task evaluation score from run_eval satisfies the test's declared threshold or relation: "+" | ".join(t[:230] for _,t in checks[:4])
            mapped.append({"snapshot":"sglang","path":path,"test_name":row["test_name"],"line":row["line"],"unit_id":"quality.task_score","property":property_text,"oracle_source":oracle,"conditions":f"Pinned SGLang 0.5.14; source run_eval call {path}:{call_line} ({call_expr[:250]}); imported from {','.join(sorted(import_modules))}; benchmark, model, request count and threshold as configured in this test. This is a task-score gate, not token-by-token computation equivalence.","rule_id":"sglang_run_eval_direct_score_threshold"})
            selected.add((row["path"],row["test_name"],row["line"]))
            files[path]=hashlib.sha256(source.read_bytes()).hexdigest()
    with (HERE/"crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=COLS).writerows(mapped)
    remaining=[r for r in rows if (r["path"],r["test_name"],r["line"]) not in selected]
    with (HERE/"pending.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,delimiter="\t",fieldnames=rows[0].keys());w.writeheader();w.writerows(remaining)
    evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"family":"direct_run_eval_score_gate","mapped_existing":len(mapped),"files_sha256":files,"exceptions_sample":sample_exceptions[:20],"review_method":"Per-definition AST confirms imported SGLang run_eval, single assigned result, and direct score/accuracy threshold assertion on that result; call and oracle lines retained. Inherited subclass cases excluded."});evidence["total_pending_after_review"]=len(remaining);(HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print("mapped",len(mapped),"files",len(files),"remaining",len(remaining))

if __name__=="__main__":main()
