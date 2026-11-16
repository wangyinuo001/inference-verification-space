# -*- coding: utf-8 -*-
"""Source-linked legacy few-shot GSM8K threshold family."""
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
 rows=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"));by=defaultdict(list)
 for r in rows:by[r["path"]].append(r)
 mapped=[];selected=set();files={}
 for path,entries in by.items():
  source=SOURCE/path
  try:text=source.read_text(encoding="utf-8");tree=ast.parse(text)
  except (OSError,SyntaxError,UnicodeDecodeError):continue
  imported=any(isinstance(n,ast.ImportFrom) and n.module=="sglang.test.few_shot_gsm8k" and any(a.name=="run_eval" and a.asname=="run_eval_few_shot_gsm8k" for a in n.names) for n in tree.body)
  if not imported:continue
  funcs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
  for row in entries:
   if row["path"]!=row["definition_path"]:continue
   hits=[n for n in funcs if n.lineno<=int(row["line"])<=n.end_lineno]
   if not hits:continue
   node=min(hits,key=lambda n:n.end_lineno-n.lineno)
   assigns=[n for n in ast.walk(node) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=="run_eval_few_shot_gsm8k"]
   if len(assigns)!=1:continue
   names={x.id for t in assigns[0].targets for x in ast.walk(t) if isinstance(x,ast.Name)}
   asserts=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith(("assertGreater","assertEqual")) and "accuracy" in ast.unparse(n) and any(name in ast.unparse(n) for name in names)]
   asserts=sorted(asserts,key=lambda n:n.lineno)
   if not asserts:continue
   oracle=";".join(f"{path}:{n.lineno}" for n in asserts)
   text_assert=" | ".join(" ".join((ast.get_source_segment(text,n) or "").split())[:240] for n in asserts)
   mapped.append({"snapshot":"sglang","path":path,"test_name":row["test_name"],"line":row["line"],"unit_id":"quality.task_score","property":"Legacy few-shot GSM8K answer-extraction accuracy meets this test's threshold: "+text_assert,"oracle_source":oracle,"conditions":f"Pinned SGLang 0.5.14; {path}:{assigns[0].lineno} calls run_eval_few_shot_gsm8k imported from python/sglang/test/few_shot_gsm8k.py:58, which extracts numeric answers and averages equality with GSM8K labels. Model, shots, question count, parallelism and server args follow this test's SimpleNamespace and fixture; score is not computational equivalence.","rule_id":"sglang_legacy_few_shot_gsm8k_accuracy_gate"})
   selected.add((path,row["test_name"],row["line"]));files[path]=hashlib.sha256(source.read_bytes()).hexdigest()
 with (HERE/"crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=COLS).writerows(mapped)
 remaining=[r for r in rows if (r["path"],r["test_name"],r["line"]) not in selected]
 with (HERE/"pending.tsv").open("w",newline="") as f:
  w=csv.DictWriter(f,delimiter="\t",fieldnames=rows[0].keys());w.writeheader();w.writerows(remaining)
 evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"family":"legacy_few_shot_gsm8k_accuracy_gate","mapped_existing":len(mapped),"files_sha256":files,"review_method":"Direct import alias, assigned legacy run_eval result, and accuracy assertion on that result; helper's answer extraction and label equality read at few_shot_gsm8k.py:58-134."});evidence["total_pending_after_review"]=len(remaining);(HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
 print("mapped",len(mapped),"remaining",len(remaining))
if __name__=="__main__":main()
