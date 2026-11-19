# -*- coding: utf-8 -*-
import ast,csv,json,hashlib
from pathlib import Path
O=Path(__file__).parent; R=O.parents[3]; P=R.parents[1]; B=P/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
N={'basic_decode_correctness_kit.py','basic_api_contract_kit.py','matched_stop_kit.py','json_constrained_kit.py','regex_constrained_kit.py','ebnf_constrained_kit.py','spec_server_kits.py'}
rows=[r for r in csv.DictReader((R/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t')if r['snapshot']=='sglang' and r['definition_path'].split('/')[-1]in N and r['review_status']not in ['property_reviewed','supporting_record_reviewed','outside_current_scope']]
fixture='python/sglang/test/server_fixtures/spec_eagle_fixture.py'
cache={}; trees={}; sources=set()
def load(p):
 if p not in cache:cache[p]=(B/p).read_text();trees[p]=ast.parse(cache[p])
 sources.add(p);return trees[p]
def seg(p,n):return '\n'.join(cache[p].splitlines()[n.lineno-1:n.end_lineno])
def checks(p,n):
 return [{'path':p,'line':x.lineno,'expression':ast.unparse(x)}for x in ast.walk(n)if isinstance(x,ast.Assert)or isinstance(x,ast.Call)and (ast.unparse(x.func).split('.')[-1].startswith('assert')or ast.unparse(x.func)=='self.fail')]
load(fixture);out=[];ev=[];deferred=[]
for r in rows:
 p=r['definition_path'];tree=load(p);n=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.lineno==int(r['definition_line']));name=n.name;file=p.split('/')[-1];u=None;helper_names=[];note=''
 if file=='basic_decode_correctness_kit.py':
  u='model.execution_variant_relation'if name=='test_determinism_temp_zero'else 'response.observed_payload_contract'if name=='test_max_token_one'else 'quality.sample_semantic_expectation';helper_names=['_decode_generate'];note='生成内容按字面片段、字符比例或重复率检查，不证明模型计算等价；重复调用仅比较strip后的短答案。'
 elif file=='basic_api_contract_kit.py':u='service.lifecycle_health'if name in ['test_health','test_health_generate']else 'service.endpoint_observed_fields'
 elif file=='matched_stop_kit.py':u='output.stop_boundary';helper_names=['_run_completions_generation','_run_chat_completions_generation']
 elif file=='json_constrained_kit.py':
  if name=='test_json_invalid':deferred.append({**r,'reason':'helper 对 INVALID 直接返回，没有拒绝状态或错误字段断言；不能确认输入校验性质。'});continue
  u='sampling.structured_constraint';helper_names=[]if name=='test_json_openai'else ['_run_decode_json'];note='只检查JSON可解析和name/population类型；混合请求中None分支直接返回，没有内容断言。'
 elif file in ['regex_constrained_kit.py','ebnf_constrained_kit.py']:
  u='sampling.structured_constraint';helper_names=['_run_decode_regex'if file.startswith('regex')else '_run_decode_ebnf','regex_match'];note='每项请求n=3或5；以独立正则re.match检查允许输出，并检查列表与非空内容。'
 elif file=='spec_server_kits.py':
  mapping={'test_parity_vs_reference':'spec.output_relation','test_eos_token':'output.stop_boundary','test_first_token_finish':'response.observed_payload_contract','test_logprob_start_len':'response.logprob_identity','test_logprob_match':'response.logprob_value','test_logprob_spec_v2_match':'response.logprob_value','test_token_ids_logprob_ragged':'response.observed_payload_contract','test_penalty_mixed':'service.configured_request_success','test_constrained_decoding':'sampling.structured_constraint','test_return_hidden_states':'response.batch_metadata_binding','test_grammar_structured_output_no_trailing_tokens':'sampling.structured_constraint','test_grammar_logprob_count_matches_completion_tokens':'response.logprob_identity'}
  u=mapping.get(name)
  if not u:deferred.append({**r,'reason':'未将接受长度、外部评测helper或线程内断言自动归入本批。'});continue
  if name.startswith('test_grammar'):helper_names=['_generate_grammar']
  if name=='test_parity_vs_reference':helper_names=['_greedy','setUpClass']
  note='Spec套件；字段存在检查不等同数值检查；隐藏状态检查仅计数与行形状；共同top-k只比较交集，额外指定token集合另检查相等。'
 owner=next(c for c in tree.body if isinstance(c,ast.ClassDef)and n in c.body)
 hs=[x for x in owner.body if isinstance(x,ast.FunctionDef)and x.name in helper_names]+[x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name in helper_names]
 assertions=checks(p,n)+[a for h in hs for a in checks(p,h)]
 helper_sources=[{'path':p,'line':h.lineno,'source':seg(p,h)}for h in hs]
 if name=='test_penalty_mixed':
  h=next(x for x in ast.walk(trees[fixture])if isinstance(x,ast.FunctionDef)and x.name=='run_decode');assertions+=checks(fixture,h);helper_sources.append({'path':fixture,'line':h.lineno,'source':seg(fixture,h)});note+='run_decode只断言HTTP200，未核对惩罚数值。'
 if not assertions:raise ValueError((p,name))
 cp=r['path'];ct=load(cp);clname=r['test_name'].split('::')[0];cl=next((c for c in ct.body if isinstance(c,ast.ClassDef)and c.name==clname),None)
 context=[{'line':x.lineno,'source':seg(cp,x)}for x in ct.body if isinstance(x,(ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign))]
 # Full consuming class and base classes retain launch overrides and multiple-inheritance context.
 context += [{'line':x.lineno,'source':seg(cp,x)}for x in ct.body if isinstance(x,ast.ClassDef)]
 body=seg(p,n);entry={**{k:r[k]for k in ['snapshot','path','test_name','line','definition_path','definition_line','entry_kind']},'unit_id':u,'test_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'test_source':body,'assertions':assertions,'helper_sources':helper_sources,'definition_class_source':seg(p,owner),'consuming_file_context':context,'launch_fixture_source':cache[fixture],'note':note};ev.append(entry)
 rr={k:r[k]for k in ['snapshot','path','test_name','line']};rr.update(unit_id=u,property='本项实际检查：'+'；'.join(' '.join(a['expression'].split())for a in assertions),oracle_source='; '.join(f'{a["path"]}:{a["line"]}'for a in assertions),conditions=f'入口 {cp}:{r["line"]}；实际方法 {p}:{n.lineno}；输入、装饰器、条件分支、共享helper及消费类启动参数完整存于evidence。{note}',rule_id='round16_sglang_shared_kits_source_review_v1');out.append(rr)
fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
with (O/'crosswalk.tsv').open('w')as f:w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(out)
(O/'evidence.json').write_text(json.dumps({'file_reviews':[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((B/p).read_bytes()).hexdigest()}for p in sorted(sources)],'entries':ev,'deferred':deferred,'review_scope':'阅读7个共享套件及消费者启动配置；继承项保留原入口和实际定义两个位置。未执行测试。'},ensure_ascii=False,indent=2));(O/'proposed-units.json').write_text('[]\n');(O/'notes.md').write_text(f'本批确认 {len(out)} 条来源入口的实际检查性质。共享方法核查一次，每个消费类保留其模型、并行、后端、推测和调度配置。INVALID JSON 请求没有拒绝断言，不计为该性质已确认。通过线程启动但不传播线程异常的检查未计入本批。所有条目为源码审查，不代表执行通过。\n');print(len(out),len(deferred),len(sources))
