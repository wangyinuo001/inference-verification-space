import ast,csv,json,re,hashlib,collections
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=OUT.parents[1];ROOT=P.parents[3];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'
seed=json.loads((P.parent/'space_seed.json').read_text());props=json.loads((OUT/'proposed-units.json').read_text());units={u['id']:u for u in seed['units']};dims={x['id'] for x in seed['condition_dimensions']}
for u in props:
 assert u['id'] not in units
 assert all(k in u for k in ['id','name','kind','phase','state_scope','conditions','property','observable','oracle','framework_scope','evidence'])
 assert set(u['conditions'])<=dims,(u['id'],set(u['conditions'])-dims)
 assert u['framework_scope']==['vllm']
 assert any(e['kind']=='source' and e['path'].startswith('vllm/') for e in u['evidence']['vllm'])
 units[u['id']]=u
r=list(csv.DictReader((OUT/'crosswalk.tsv').open(),delimiter='\t'));ev=json.loads((OUT/'evidence.json').read_text());idx=list(csv.DictReader((P/'full_delivery/all-entries.tsv').open(),delimiter='\t'))
def key(x):return tuple(str(x[k]) for k in ['snapshot','path','test_name','line'])
index={key(x):x for x in idx};keys={key(x) for x in r};assert len(keys)==len(r)==len(ev['entry_reviews'])==ev['confirmed_rows']; assert keys=={key(x) for x in ev['entry_reviews']}
cache={}
def anchor(a):
 assert re.fullmatch(r'[^:;\s]+:\d+',a),a
 p,n=a.rsplit(':',1);n=int(n)
 if p not in cache:cache[p]=(SRC/p).read_text().splitlines()
 assert 1<=n<=len(cache[p]),a
 assert cache[p][n-1].strip(),('blank anchor',a)
assert set(r[0])==set(['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'])
for row in r:
 assert key(row) in index and index[key(row)]['review_status']!='property_reviewed',key(row)
 assert row['unit_id'] in units,row['unit_id']
 cond=json.loads(row['conditions']);assert cond['input_and_call_source'] and cond['validation'].endswith('未使用GPU')
 for a in row['oracle_source'].split(';'):anchor(a)
 for a in cond['implementation_anchors']:anchor(a)
 t=ast.parse((SRC/row['path']).read_text());name=row['test_name'];parts=name.split('::')
 if len(parts)==2:t=next(x for x in t.body if isinstance(x,ast.ClassDef) and x.name==parts[0])
 f=next(x for x in t.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==parts[-1]);assert f.lineno==int(row['line'])
for file in ev['file_reviews']+ev['supporting_source_reviews']:
 assert hashlib.sha256((SRC/file['path']).read_bytes()).hexdigest()==file['source_sha256']
for u in props:
 for e in u['evidence']['vllm']:anchor(f"{e['path']}:{e['line']}")
other=[]
for rd in ['round12_work','round13_work','round14_work','round15_work','round16_work','round17_work']:
 for f in (P/rd).rglob('crosswalk.tsv'):
  if f.parent==OUT:continue
  matches=keys&{key(x) for x in csv.DictReader(f.open(),delimiter='\t')}
  assert not matches,(f,matches)
report={'validated_rows':len(r),'files':len(ev['file_reviews']),'supporting_source_files':len(ev['supporting_source_reviews']),'proposed_units':[u['id'] for u in props],'per_file':dict(collections.Counter(x['path'] for x in r)),'checks':['九字段与唯一索引入口对应','所有条目仍pending且排除round12/13/14/15/16/17其他crosswalk','断言及实现锚点为纯path:line且指向非空行','测试def行号匹配索引','每条conditions JSON含真实输入、fixture/helper/class及生产锚点','file_reviews与supporting_source_reviews SHA256一致','性质ID存在于seed或完整提案且条件维度有效'],'upstream_execution':False,'gpu_used':False}
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
