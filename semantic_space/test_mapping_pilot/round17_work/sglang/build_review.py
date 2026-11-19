# -*- coding: utf-8 -*-
import ast,csv,json,hashlib,re,collections
from pathlib import Path
O=Path(__file__).parent;R=O.parents[3];P=R.parents[1];B=P/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
FILES=['test/manual/openai_server/function_call/test_tool_choice.py','sgl-model-gateway/e2e_test/chat_completions/test_function_calling.py','test/registered/openai_server/features/test_openai_server_hidden_states.py']
PROD=['python/sglang/srt/entrypoints/openai/serving_chat.py','python/sglang/srt/entrypoints/openai/serving_completions.py','sgl-model-gateway/src/routers/grpc/utils.rs','sgl-model-gateway/e2e_test/fixtures/setup_backend.py']
cache={p:(B/p).read_text()for p in FILES};trees={p:ast.parse(s)for p,s in cache.items()};defs={};classes={};helpers={}
def seg(p,n):return '\n'.join(cache[p].splitlines()[n.lineno-1:n.end_lineno])
for p,t in trees.items():
 classes[p]={n.name:n for n in t.body if isinstance(n,ast.ClassDef)}
 defs[p]={n.lineno:n for n in ast.walk(t)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 helpers[p]=[{'line':n.lineno,'source':seg(p,n)}for n in ast.walk(t)if isinstance(n,ast.FunctionDef)and not n.name.startswith('test')]
def assertions(p,n,guards=()):
 out=[]
 if isinstance(n,ast.If):
  for c in n.body:out+=assertions(p,c,guards+(ast.unparse(n.test),))
  for c in n.orelse:out+=assertions(p,c,guards+('not ('+ast.unparse(n.test)+')',))
  return out
 if isinstance(n,ast.Assert)or isinstance(n,ast.Call)and (ast.unparse(n.func).split('.')[-1].startswith('assert')or ast.unparse(n.func)in ['pytest.raises','pytest.fail','self.fail']):
  out.append({'path':p,'line':n.lineno,'expression':ast.unparse(n),'guards':list(guards)})
 for c in ast.iter_child_nodes(n):out+=assertions(p,c,guards)
 return out
already=set()
for f in [R/'semantic_space/test_mapping_pilot/round16_work/sglang/crosswalk.tsv']:
 for r in csv.DictReader(f.open(),delimiter='\t'):already.add((r['snapshot'],r['path'],r['test_name'],r['line']))
rows=[];ev=[];support=[];missing=[]
for r in csv.DictReader((R/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'):
 if r['snapshot']!='sglang'or r['definition_path']not in FILES or r['review_status']in ['property_reviewed','supporting_record_reviewed','outside_current_scope']or tuple(r[k]for k in ['snapshot','path','test_name','line'])in already:continue
 p=r['definition_path'];n=defs[p][int(r['definition_line'])];owner=next(c for c in classes[p].values()if n in c.body);entryclass=r['test_name'].split('::')[0];cc=classes[p].get(entryclass,owner);resolved=n
 # Explicit skipped override only delegates to the parent method.
 if not assertions(p,n)and any(isinstance(x,ast.Call)and ast.unparse(x.func).startswith('super().test_')for x in ast.walk(n)):
  parent=next(c for c in classes[p].values()if c.name in [ast.unparse(b)for b in owner.bases]);resolved=next(x for x in parent.body if isinstance(x,ast.FunctionDef)and x.name==n.name)
 aa=assertions(p,resolved);dep=[];name=n.name;u='response.observed_payload_contract';note='工具选择/输出接口检查，实际断言保留于本项；不同模型使用不同parser与FLAKY_TESTS分支。'
 if p==FILES[2]:
  u='response.batch_metadata_binding';fn=name.replace('test_','run_',1);h=next(x for x in classes[p]['BaseTestOpenAIServerWithHiddenStates'].body if isinstance(x,ast.FunctionDef)and x.name==fn);aa+=assertions(p,h);dep.append({'line':h.lineno,'source':seg(p,h)});note='只核对hidden_states是否按开关出现、非空和流式片段计数；不比较向量值。消费类决定并行样本数、CUDA Graph、EAGLE配置。'
 elif name in ['test_error_handling_invalid_tool_choice','test_invalid_tool_missing_name','test_invalid_json_schema_in_tool','test_conflicting_defs_required_tool_choice']:u='input.error_response_classification'
 elif name=='test_tool_choice_auto_streaming':u='service.configured_request_success';note='流消费完成后只断言本地收集器是list；不检查返回内容是否非空或是否发生工具调用。'
 elif name in ['test_streaming_multiple_choices_finish_reason','test_streaming_multiple_choices_without_tools']:u='response.delivery'
 if not aa:missing.append(r);continue
 body=seg(p,resolved);entry={**{k:r[k]for k in ['snapshot','path','test_name','line','definition_path','definition_line','entry_kind']},'unit_id':u,'actual_method_line':resolved.lineno,'test_source':body,'test_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'assertions':aa,'helper_sources':helpers[p],'extra_helper_sources':dep,'consuming_class_source':seg(p,cc),'definition_class':owner.name,'decorators':[ast.unparse(x)for x in cc.decorator_list+n.decorator_list],'module_constants':[{'line':x.lineno,'source':seg(p,x)}for x in trees[p].body if isinstance(x,(ast.Assign,ast.AnnAssign,ast.Import,ast.ImportFrom))],'notes':note};ev.append(entry)
 rr={k:r[k]for k in ['snapshot','path','test_name','line']};rr.update(unit_id=u,property='本项实际检查：'+'；'.join(('条件['+' and '.join(a['guards'])+'] 'if a['guards']else '')+' '.join(a['expression'].split())for a in aa),oracle_source='; '.join(f'{a["path"]}:{a["line"]}'for a in aa),conditions=f'消费类 {entryclass}；具体请求、工具schema、采样参数、stream/n及启动配置见evidence；方法和类装饰器含skip时不表示实际执行。{note}',rule_id='round17_sglang_tool_and_hidden_state_family_review');rows.append(rr)
fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
with (O/'crosswalk.tsv').open('w')as f:w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)
for e in ev:
 if e['definition_path']==FILES[2]:
  e['production_anchors']=[{'path':PROD[0],'line':1115},{'path':PROD[1],'line':368},{'path':PROD[1],'line':533}]
 elif e['definition_path']==FILES[0]:
  e['production_anchors']=[{'path':PROD[0],'line':430},{'path':PROD[0],'line':1449},{'path':PROD[0],'line':1796}]
 else:
  e['production_anchors']=[{'path':PROD[2],'line':232},{'path':PROD[2],'line':564}]
  e['fixture_anchor']={'path':PROD[3],'line':23}
reviewed=FILES+PROD
(O/'evidence.json').write_text(json.dumps({'file_reviews':[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((B/p).read_bytes()).hexdigest()}for p in reviewed],'entries':ev,'unconfirmed':missing,'source_review_scope':'工具选择服务与gateway端到端响应，及隐藏状态API；不把其视为底层数值正确性测试。'},ensure_ascii=False,indent=2));(O/'proposed-units.json').write_text('[]\n')
units={x['id']for x in json.load((R/'semantic_space/space_seed.json').open())['units']};assert all(r['unit_id']in units for r in rows);assert len(rows)==len({tuple(r[k]for k in ['snapshot','path','test_name','line'])for r in rows})
for r in rows:
 for a in r['oracle_source'].split('; '):assert re.fullmatch(r'.+:\d+',a)
v={'status':'passed','count':len(rows),'inherited_candidates':sum(e['entry_kind']=='python_inherited_test_candidate'for e in ev),'file_hashes':len(reviewed),'unit_counts':dict(collections.Counter(r['unit_id']for r in rows)),'unconfirmed':len(missing),'upstream_tests_run':False};(O/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));(O/'notes.md').write_text('共享方法按定义核查，继承方法保留实际消费类、启动参数、skip和flaky条件。Llama/Mistral/LFM2的多工具检查可以进入宽松分支，Qwen要求两个特定工具；不将这些判定强度等同。auto streaming只检查本地list及遍历完成，不归为工具选择正确。hidden_states检查是接口字段与片段数量，不是数值比较。\n');print(v)
