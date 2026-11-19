"""Manual source-reviewed assignments; AST exports exact evidence and helper checks."""
import ast,csv,json,hashlib
from pathlib import Path
ROOT=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集');PUB=ROOT/'github_preview/inference-verification-space';SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).resolve().parent
assign={};helpers={};proposals=[];product_refs={}
def group(path,lines,unit,condition):
 for line in lines:
  assert (path,line) not in assign
  assign[path,line]=(unit,condition)
def all_lines(p):
 return [n.lineno for n in ast.walk(ast.parse((SRC/p).read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
def unit(id,name,phase,conditions,prop,obs,refs):
 proposals.append(dict(id=id,name=name,kind='obligation',framework_scope=['vllm'],phase=phase,state_scope='shared',conditions=conditions,property=prop,observable=obs,oracle='逐项比较测试声明的值、结构、异常或交互记录；实际检查表达式与替身条件见各入口证据。',evidence={'vllm':[dict(path=p,line=l,kind='source') for p,l in refs]}));product_refs[id]=refs


D='tests/v1/sample/test_dry.py'
group(D,[50,59,68,79,100,107,114,236,262,380,419,485,626,742,999],'sampling.logit_transform_relation','CPU人工token历史、logits与DRY参数，实际示例LogitsProcessor及独立朴素参考；999为CUDA两种分块预算比较；不加载模型。')
group(D,[148,159,183,191,289,439,452,950,968],'sampling.modifier_admissibility','真实DRY参数/支持范围检查；SimpleNamespace请求配置和CPU张量；950经生产插件校验器包装，错误类型按实际断言；452同时检查拒绝前未改logits。')
group(D,[405,478,671,728,978],'sampling.modifier_state_initialization','真实CPU DryState，tokenizer替身或禁用；检查逐请求开关、默认/复用breaker集合、未解析提示与初始化字段；不声称真实生成质量。')
group(D,[130,764,800],'sampling.breaker_token_resolution','人工tokenizer.batch_decode词表，真实字符索引/缓存解析；800固定seed生成词表并与全词表substring扫描比较。')
group(D,[658],'cache.eviction_outcome','CPU DryState的breaker-mask字典缓存；按序插入不同集合，检查容量上限、最后集合存在及首集合缺失，不测KV缓存。')
group(D,[552],'compute.execution_relation','CPU请求槽位/历史与DRY；真实BatchSharder但TP组与plan kernel替身，比较分片logits与全批对应行以及覆盖索引。')
unit('sampling.modifier_admissibility','自定义采样修饰器参数与支持范围','sampling',['sampling_mode','logit_modifier_profile','speculative_mode'],'自定义修饰器校验参数类型、数值边界、未知字段和执行配置兼容性，在不支持的请求或展开布局进入计算前拒绝。',['validation_error','accepted_parameters','unchanged_logits'],[('examples/features/logits_processor/dry.py',570),('examples/features/logits_processor/dry.py',499),('examples/features/logits_processor/dry.py',709)])
unit('sampling.modifier_state_initialization','采样修饰状态的默认值与请求更新','sampling',['sampling_mode','logit_modifier_profile','sampling_batch_transition_context'],'修饰器初始化及请求登记解析默认参数、开关和共享常量；复用槽位时清除旧请求条件，缺失外部解析条件按声明回退并记录提示。',['enabled_rows','breaker_ids','default_identity','warnings'],[('examples/features/logits_processor/dry.py',499),('examples/features/logits_processor/dry.py',659)])
unit('sampling.breaker_token_resolution','采样中断文本映射到词表token集合','sampling',['tokenization_profile','logit_modifier_profile'],'将声明中断字符串映射到包含其文本的token集合；范围含新增token，结果去重有序，缓存返回值不被调用者修改污染。',['token_ids','index_keys','cached_results'],[('examples/features/logits_processor/dry.py',427),('examples/features/logits_processor/dry.py',443)])

T='tests/v1/sample/test_topk_topp_sampler.py'
group(T,[64],'deployment.effective_configuration','实际Sampler构造，检查FP64噪声开关传入下层采样器；不运行采样。')
group(T,[74,174,209],'compute.kernel_dispatch_binding','实际后端选择，平台探测/可用性/JIT能力由monkeypatch控制；74真实小张量采样且强制错误导入守卫；174/209检查自动回退与显式启用错误。')
group(T,[123],'compute.execution_relation','相同seed和声明FP64指数噪声，与显式指数竞速argmax参考比较；有限三行分布，不作统计独立性保证。')
group(T,[146,298,314,328,349,365,376,395,414,426,468,519,553,580,609,638,683,725,757,787,826,857,896,922],'sampling.logit_transform_relation','实际PyTorch/Triton过滤，人工logits、行级top-k/top-p、种子及平台门限。辅助比较器在top-k只允许阈值并列差异；top-p只检查保留数量差异在3以内或小于0.5%，不是逐token数值等价。其他入口以各自断言限定NaN、有限支持集、布局、分块和索引边界；826对不应启动的分裂路径加失败守卫。')
group(T,[1060],'sampling.policy','真实FlashInfer人工异常logits；只检查返回形状、batch长度和token索引在词表范围内，未证明未污染其他行或分布正确。')
group(T,[1137],'sampling.distribution_relation','GPU实际FlashInfer与本地采样，32词表固定seed、50000次；理论参考为PyTorch过滤后softmax，支持集外零样本，支持集内卡方p值>1e-6。')
group(D,[689],'sampling.modifier_state_initialization','实际CPU初始化warmup加调用spy；检查调用次数、参数签名及请求缓存未被预热污染，不测CUDA编译性能。')
group(D,[911],'deployment.extension_dispatch_contract','真实插件FQCN解析与构造，人工CPU请求状态，检查构造对象类型为DryState。')
helper_links={}
for l in [298,314,328,349,376,395,414,787]: helper_links[T,l]=[(T,'_compare_results')]
helper_links[T,1060]=[(T,'_check_tokens')]
helper_links[T,1137]=[(T,'_chi2_check')]

pending_units=json.loads((PUB/'semantic_space/test_mapping_pilot/round12_work/runtime/proposed-units.json').read_text())
existing=json.loads((PUB/'semantic_space/space_seed.json').read_text());labels={u['id']:u['name'] for u in existing['units']+pending_units+proposals};valid=set(labels)
rows=[];reviews=[];hashes={};skipped=[];missing=[];trees={}
index=list(csv.DictReader((PUB/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
def extract_checks(node,origin='entry'):
 result=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Assert):result.append(dict(line=n.lineno,expression=ast.unparse(n),origin=origin))
  elif isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and (n.func.attr.startswith('assert') or n.func.attr=='raises'):result.append(dict(line=n.lineno,expression=ast.unparse(n),origin=origin))
 return result
for entry in index:
 if entry['snapshot']!='vllm':continue
 path=entry['path'];line=int(entry['line']);key=(path,line)
 if key not in assign:continue
 if entry['review_status']=='property_reviewed':skipped.append(key);continue
 unit,cond=assign[key];assert unit in valid,unit
 if path not in trees:
  data=(SRC/path).read_bytes();source=data.decode();tree=ast.parse(source);trees[path]=(source,tree);hashes[path]=hashlib.sha256(data).hexdigest()
 source,tree=trees[path]
 nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno==line];assert len(nodes)==1,key
 node=nodes[0];checks=extract_checks(node)
 for hp,helper in helper_links.get(key,[]):
  ht=ast.parse((SRC/hp).read_text());hn=[n for n in ast.walk(ht) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==helper];assert len(hn)==1,(key,helper)
  checks += [dict(**x,path=hp) for x in extract_checks(hn[0],helper)]
 assert checks,key
 prop=labels[unit]+'。本项断言：'+'；'.join(dict.fromkeys(c['expression'] for c in checks))
 oracle=f"{checks[0].get('path',path)}:{checks[0]['line']}";decorators='; '.join(ast.unparse(n) for n in node.decorator_list)
 calls=[dict(line=n.lineno,expression=ast.unparse(n)) for n in ast.walk(node) if isinstance(n,ast.Call)]
 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+decorators,rule_id='round17_source_review_runtime'))
 reviews.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,end_line=node.end_lineno,unit_id=unit,property=prop,source_sha256=hashes[path],definition_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),checks=checks,calls=calls,source_excerpt=ast.get_source_segment(source,node),helper_checks=helper_links.get(key,[]),review_method='逐项核对调用、fixture/辅助函数与断言；AST 仅导出证据；未执行测试'))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
file_reviews=[]
for p,h in hashes.items():
 s,t=trees[p];helpers=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(s,n)) for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('test')]
 file_reviews.append(dict(snapshot='vllm',path=p,source_sha256=h,helper_sources=helpers))
for helper_path in []:
 helper_data=(SRC/helper_path).read_bytes();helper_source=helper_data.decode();helper_tree=ast.parse(helper_source)
 file_reviews.append(dict(snapshot='vllm',path=helper_path,source_sha256=hashlib.sha256(helper_data).hexdigest(),review_role='shared_fixture',helper_sources=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(helper_source,n)) for n in helper_tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('test')]))
(OUT/'evidence.json').write_text(json.dumps(dict(file_reviews=file_reviews,entry_reviews=reviews),ensure_ascii=False,indent=2))
source_reviews=[]
for u in proposals:
 for v in u['evidence']['vllm']:
  data=(SRC/v['path']).read_bytes();s=data.decode();t=ast.parse(s);candidates=[n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno==v['line']]
  excerpt=ast.get_source_segment(s,candidates[0]) if candidates else '\n'.join(s.splitlines()[max(0,v['line']-1):v['line']+60])
  source_reviews.append(dict(**v,snapshot='vllm',source_sha256=hashlib.sha256(data).hexdigest(),source_excerpt=excerpt))
(OUT/'product-source-evidence.json').write_text(json.dumps(source_reviews,ensure_ascii=False,indent=2))
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2))
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
summary=dict(mapped_entries=len(rows),reviewed_files=len(hashes),already_reviewed_excluded=len(skipped),by_file={p:sum(r['path']==p for r in rows) for p in hashes},by_unit={u:sum(r['unit_id']==u for r in rows) for u in sorted({r['unit_id'] for r in rows})},runtime_execution=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

# Actual implementation functions supporting reused verification units.
extra_refs={
 'sampling.logit_transform_relation':[('examples/features/logits_processor/dry.py',141),('examples/features/logits_processor/dry.py',154),('examples/features/logits_processor/dry.py',238),('examples/features/logits_processor/dry.py',709),('vllm/v1/sample/ops/topk_topp_sampler.py',527),('vllm/v1/sample/ops/topk_topp_triton.py',1618)],
 'compute.execution_relation':[('vllm/v1/worker/gpu/sample/batch_shard.py',230),('vllm/v1/sample/ops/topk_topp_sampler.py',609)],
 'compute.kernel_dispatch_binding':[('vllm/v1/sample/ops/topk_topp_sampler.py',103),('vllm/v1/sample/ops/topk_topp_sampler.py',237),('vllm/v1/sample/ops/topk_topp_sampler.py',405)],
 'sampling.distribution_relation':[('vllm/v1/sample/ops/topk_topp_sampler.py',634)],
 'sampling.policy':[('vllm/v1/sample/ops/topk_topp_sampler.py',634)],
 'cache.eviction_outcome':[('examples/features/logits_processor/dry.py',691)],
 'deployment.extension_dispatch_contract':[('vllm/v1/worker/gpu/sample/logits_processor/loader.py',79),('vllm/v1/worker/gpu/sample/logits_processor/loader.py',172)],
 'sampling.modifier_admissibility':[('vllm/v1/worker/gpu/sample/logits_processor/loader.py',208)],
 'sampling.modifier_state_initialization':[('examples/features/logits_processor/dry.py',536)],
}
# Resolve source functions by name where file layout differs; all anchors must be definitions.
loader_path='vllm/v1/worker/gpu/sample/logits_processor/loader.py'
loader_tree=ast.parse((SRC/loader_path).read_text())
loader_defs={n.name:n.lineno for n in ast.walk(loader_tree) if isinstance(n,ast.FunctionDef)}
extra_refs['deployment.extension_dispatch_contract']=[(loader_path,loader_defs['_load_v2_logitsprocs_by_fqcns']),(loader_path,loader_defs['build_custom_logits_processors'])]
extra_refs['sampling.modifier_admissibility']=[(loader_path,loader_defs['build_custom_logits_processors_params_validator'])]
for unit_id,refs in extra_refs.items():
 for path,line in refs:
  data=(SRC/path).read_bytes();source=data.decode();tree=ast.parse(source)
  nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.lineno==line]
  assert len(nodes)==1,(path,line)
  source_reviews.append(dict(snapshot='vllm',path=path,line=line,kind='source',unit_id=unit_id,source_sha256=hashlib.sha256(data).hexdigest(),source_excerpt=ast.get_source_segment(source,nodes[0])))
(OUT/'product-source-evidence.json').write_text(json.dumps(source_reviews,ensure_ascii=False,indent=2))
(OUT/'prior-source-anchor-corrections.json').write_text(json.dumps([dict(batch='round16_work/runtime',unit_id='diagnostic.analytic_work_estimate',snapshot='vllm',path='vllm/v1/metrics/perf.py',old_line=1460,new_line=1465,reason='1460是空行；1465为ModelMetrics定义，1492起聚合分项计算量与字节估算。其余16个提议锚点逐个核查均为对应类或函数。')],ensure_ascii=False,indent=2))
(OUT/'review.md').write_text('本批逐项核对67个入口：DRY采样修饰36项，top-k/top-p采样31项。源码审查，未运行测试。\n\nDRY保留了参数拒绝、词表解析、请求初始化、数值变换、缓存淘汰和批分片比较之间的不同性质。top-k/top-p有共同参考比较、异常输入支持集检查、配置回退和统计分布检查；共同比较器并不保证逐token完全一致，阈值及并列值差异已记录。FlashInfer异常值测试只检查合法token范围，未把测试注释中的跨行隔离承诺视为已验证。\n\n共享比较器源码与各入口调用、参数、断言均附证据。DRY峰值显存和子进程导入两项暂未导入；不因扫描到入口而认定分类完成。\n')
