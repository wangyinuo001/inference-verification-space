# -*- coding: utf-8 -*-
"""Resolve scripted-runtime test wrapper to its actual assertion-bearing script."""
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

def main():
    rows=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"))
    grouped=defaultdict(list)
    for r in rows:grouped[r["path"]].append(r)
    audited=[];mapped=[];selected=set();files={}
    for path,entries in grouped.items():
        source=SOURCE/path
        try:text=source.read_text(encoding="utf-8");tree=ast.parse(text)
        except (OSError,SyntaxError,UnicodeDecodeError):continue
        classes=[n for n in ast.walk(tree) if isinstance(n,ast.ClassDef)]
        for row in entries:
            line=int(row["line"])
            candidates=[n for n in classes if n.lineno<=line<=n.end_lineno]
            if not candidates:continue
            cls=min(candidates,key=lambda n:n.end_lineno-n.lineno)
            methods=[n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
            method=next((n for n in methods if n.name==row["test_name"].split("::")[-1]),None)
            if method is None:continue
            calls=[n for n in ast.walk(method) if isinstance(n,ast.Call) and ast.unparse(n.func)=="self.server.execute_script"]
            if len(calls)!=1 or not calls[0].args or not isinstance(calls[0].args[0],ast.Attribute):continue
            script_name=calls[0].args[0].attr
            script=next((n for n in methods if n.name==script_name),None)
            if script is None:continue
            assertions=sorted([n for n in ast.walk(script) if isinstance(n,ast.Assert)],key=lambda n:n.lineno)
            ops=[(n.lineno,ast.unparse(n.func),n) for n in ast.walk(script) if isinstance(n,ast.Call)]
            aborts={ast.unparse(n.args[0]) for _,name,n in ops if name=="t.abort" and n.args}
            checks=[(n.lineno,ast.unparse(n.test)) for n in assertions]
            cleanup=[(ln,expr) for ln,expr in checks if any(expr.startswith(f"{name}.kv_pages == 0") or expr.startswith(f"{name}.lock_refs == 0") or f"{name}.req_pool_idx is None" in expr for name in aborts)]
            chunks=[(ln,expr) for ln,expr in checks if "chunks_done" in expr and isinstance(next((n for n in assertions if n.lineno==ln),None).test,ast.Compare)]
            status=[(ln,expr) for ln,expr in checks if any(f"{name}.status" in expr for name in aborts)]
            has_start=any(name=="t.start_req" for _,name,_ in ops)
            has_finish_wait=any(name in ("run_until_finished","run_until") for _,name,_ in ops)
            if cleanup:
                unit="abort.cleanup";selected_checks=cleanup;basis="After scripted abort, the aborted handle's KV pages, lock refs or request-pool ownership meet the asserted cleanup condition"
            elif not aborts and chunks and has_start and has_finish_wait:
                unit="schedule.segment_continuity";selected_checks=chunks;basis="For the scripted request and chunk-size configuration, chunk progress matches the asserted count or boundary"
            elif status:
                unit="abort.terminal_status";selected_checks=status;basis="After scripted abort, the aborted handle has the asserted terminal status"
            else:
                unit="";selected_checks=[];basis=""
            audited.append({"snapshot":"sglang","path":path,"test_name":row["test_name"],"line":row["line"],"script_name":script_name,"script_line":script.lineno,"execute_script_line":calls[0].lineno,"assertion_lines":";".join(str(n.lineno) for n in assertions),"assertion_conditions":" | ".join(expr[:150] for _,expr in checks[:8]),"operations":";".join(sorted(set(name for _,name,_ in ops if name in ("t.abort","t.start_req","run_until","run_until_finished")))),"mapped_unit":unit,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest()})
            if not unit:continue
            mapped.append({"snapshot":"sglang","path":path,"test_name":row["test_name"],"line":row["line"],"unit_id":unit,"property":basis+": "+" | ".join(expr for _,expr in selected_checks[:5]),"oracle_source":";".join(f"{path}:{ln}" for ln,_ in selected_checks),"conditions":f"Pinned SGLang 0.5.14. Wrapper {path}:{calls[0].lineno} invokes {script_name} at {path}:{script.lineno}; ScriptedHttpServer.execute_script at python/sglang/test/scripted_runtime/http_server.py:81 propagates script failure. ENGINE_KWARGS/class setup at {path}:{cls.lineno}. Explicit request arguments and state transitions in script apply; this is internal scripted runtime, not public HTTP behavior.","rule_id":"sglang_scripted_runtime_resolved_assertion_"+unit})
            selected.add((path,row["test_name"],row["line"]))
            files[path]=hashlib.sha256(source.read_bytes()).hexdigest()
    with (HERE/"scripted-runtime-audit.tsv").open("w",newline="") as f:
        fields=list(audited[0].keys());w=csv.DictWriter(f,delimiter="\t",fieldnames=fields);w.writeheader();w.writerows(audited)
    with (HERE/"crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=COLS).writerows(mapped)
    remaining=[]
    for r in rows:
        if (r["path"],r["test_name"],r["line"]) in selected:continue
        if any(a["path"]==r["path"] and a["test_name"]==r["test_name"] and a["line"]==r["line"] for a in audited):
            audit=next(a for a in audited if a["path"]==r["path"] and a["test_name"]==r["test_name"] and a["line"]==r["line"])
            r["triage_status"]="scripted_assertions_need_property_review"
            r["source_evidence"] += ";" + r["path"] + ":" + str(audit["script_line"])
            r["review_reason"]="Wrapper resolves to scripted-runtime callback; inspect actual script assertions, operations, and state sequence before mapping. See scripted-runtime-audit.tsv."
        remaining.append(r)
    with (HERE/"pending.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,delimiter="\t",fieldnames=rows[0].keys());w.writeheader();w.writerows(remaining)
    evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"family":"scripted_runtime_wrapper_callback","linked":len(audited),"mapped_existing":len(mapped),"files_sha256":files,"review_method":"Resolve each execute_script callback to same-class script function; inspect callback operations and explicit state assertions. Only exact abort-handle cleanup/status or chunk-count relation mapped. Runtime runner propagates callback failures."});evidence["total_pending_after_review"]=len(remaining);(HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print("linked",len(audited),"mapped",len(mapped),"remaining",len(remaining))

if __name__=="__main__":main()
