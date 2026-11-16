# -*- coding: utf-8 -*-
"""Confirm progress assertions inside already-linked scripted callbacks."""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"
COLS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]
def main():
 rows=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"));keep=[];out=[];files={}
 cache={}
 for row in rows:
  if row['triage_status']!='scripted_assertions_need_property_review':keep.append(row);continue
  path=row['path']
  if path not in cache:cache[path]=ast.parse((SOURCE/path).read_text(encoding='utf-8'))
  t=cache[path];cl=min((n for n in ast.walk(t) if isinstance(n,ast.ClassDef) and n.lineno<=int(row['line'])<=n.end_lineno),key=lambda n:n.end_lineno-n.lineno)
  m=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==row['test_name'].split('::')[-1]);call=next(n for n in ast.walk(m) if isinstance(n,ast.Call) and ast.unparse(n.func)=='self.server.execute_script');script=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==call.args[0].attr)
  ops=[ast.unparse(n.func) for n in ast.walk(script) if isinstance(n,ast.Call)]
  finished=[n for n in ast.walk(script) if isinstance(n,ast.Assert) and isinstance(n.test,ast.Attribute) and n.test.attr=='finished']
  if 't.abort' in ops or 't.start_req' not in ops or 'run_until_finished' not in ops or not finished:keep.append(row);continue
  checks=sorted(finished,key=lambda n:n.lineno)
  out.append({'snapshot':'sglang','path':path,'test_name':row['test_name'],'line':row['line'],'unit_id':'schedule.progress','property':'A scripted request that was admitted and driven until completion reaches the asserted finished state: '+' | '.join(ast.unparse(n.test) for n in checks),'oracle_source':';'.join(f'{path}:{n.lineno}' for n in checks),'conditions':f'Pinned SGLang 0.5.14; wrapper {path}:{call.lineno} executes callback {path}:{script.lineno} through ScriptedHttpServer.execute_script at python/sglang/test/scripted_runtime/http_server.py:81; request arguments and ENGINE_KWARGS are in the source callback/class. This asserts completion, not token correctness.','rule_id':'sglang_scripted_runtime_positive_completion'})
  files[path]=hashlib.sha256((SOURCE/path).read_bytes()).hexdigest()
 with (HERE/'crosswalk.tsv').open('a',newline='') as f:csv.DictWriter(f,delimiter='\t',fieldnames=COLS).writerows(out)
 with (HERE/'pending.tsv').open('w',newline='') as f:w=csv.DictWriter(f,delimiter='\t',fieldnames=rows[0].keys());w.writeheader();w.writerows(keep)
 ev=json.loads((HERE/'evidence.json').read_text());ev.setdefault('reviewed_clusters',[]).append({'family':'scripted_runtime_positive_completion','mapped_existing':len(out),'files_sha256':files,'review_method':'Already-linked callback has start_req and run_until_finished, no abort call, and a direct positive assert handle.finished. No token-value conclusion.'});ev['total_pending_after_review']=len(keep);(HERE/'evidence.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n')
 print('mapped',len(out),'pending',len(keep))
if __name__=='__main__':main()
