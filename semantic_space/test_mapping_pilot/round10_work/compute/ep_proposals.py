import ast,csv,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[4]; ROOT=P.parents[1];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).parent
path='tests/model_executor/model_loader/test_ep_weight_filter.py';product='vllm/model_executor/model_loader/ep_weight_filter.py'
src=(SRC/path).read_text();tree=ast.parse(src)
index={r['test_name']:r for r in csv.DictReader((P/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t') if r['snapshot']=='vllm' and r['path']==path and r['review_status']!='property_reviewed'}
rows=[]
for c in tree.body:
 if not isinstance(c,ast.ClassDef):continue
 for f in c.body:
  if not isinstance(f,ast.FunctionDef) or c.name+'::'+f.name not in index:continue
  r=index[c.name+'::'+f.name]
  checks=[(a.lineno,ast.unparse(a)) for a in ast.walk(f) if isinstance(a,ast.Assert)]
  if not checks:continue
  desc={'TestParseExpertId':'根据检查点名称中显式专家编号解析专家身份，普通权重、共享专家和未标专家编号的融合权重返回None。','TestComputeLocalExpertIds':'根据专家总数、EP规模、rank和linear/round_robin放置方式计算本rank归属集合；保留本项具体比较的成员、数量、并集和不重叠要求。','TestShouldSkipWeight':'给定rank归属集合，仅跳过不属于该rank的.weight/.weight_packed，保留缩放元数据、共享专家和普通权重；无过滤配置保持不跳过。','TestSafetensorsWeightsIteratorWithEpFilter':'GPT2无专家权重检查点在None或空本地专家集合过滤下保持读取出的权重名称集合。','TestEpFilterOnSyntheticMoeWeights':'合成8专家safetensors在给定EP/rank过滤下返回预期的专家权重数量与归属，并保留本项明确检查的普通/共享权重；数值项将返回张量与无过滤路径逐项比较。'}[c.name]
  rows.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'unit_id':'model.checkpoint_expert_selection','property':'验证类别：'+desc+' 本项实际检查：'+'；'.join(f'L{l}: {x}' for l,x in checks),'oracle_source':'; '.join(f'{path}:{l}' for l,x in checks),'conditions':json.dumps({'source_test':f'{path}:{f.lineno}-{f.end_lineno}','setup_fixture_class':f'{path}:{c.lineno}-{c.end_lineno}','implementation':product,'evidence':'源码核对，未执行；解析/过滤阶段，不声称实际参数装载或分布式推理正确'},ensure_ascii=False),'rule_id':'round10_source_review_'+c.name})
with (OUT/'proposed-crosswalk.tsv').open('w') as h:
 w=csv.DictWriter(h,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
p=OUT/'proposed-units.json';obj=json.loads(p.read_text());obj['proposals'].append({'id':'model.checkpoint_expert_selection','name':'专家检查点识别与rank过滤','kind':'obligation','phase':'preflight','state_scope':'deployment','conditions':['parameter_mapping','execution_layout'],'property':'根据专家检查点的名字、放置方式和rank归属确定应读取的权重集合；声明保留的非专家权重与缩放元数据不得被过滤。','observable':['parsed_expert_id','rank_expert_ids','skip_decision','returned_weight_names','returned_weight_values'],'oracle':'按声明名称语法和linear/round_robin规则检查ID及集合；以合成safetensors或无过滤读取对照返回名称和张量。','evidence':{'vllm':[{'path':product,'line':21,'kind':'source'},{'path':path,'line':27,'kind':'existing_test'},{'path':path,'line':338,'kind':'existing_test'}]}});obj['additional_proposed_rows']=len(rows);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
e=OUT/'evidence.json';obj=json.loads(e.read_text());obj['file_reviews'].append({'snapshot':'vllm','path':path,'source_sha256':hashlib.sha256((SRC/path).read_bytes()).hexdigest(),'implementation_path':product,'implementation_sha256':hashlib.sha256((SRC/product).read_bytes()).hexdigest(),'status':'proposed-unit-not-in-confirmed-crosswalk'});e.write_text(json.dumps(obj,ensure_ascii=False,indent=2))
print(len(rows))
