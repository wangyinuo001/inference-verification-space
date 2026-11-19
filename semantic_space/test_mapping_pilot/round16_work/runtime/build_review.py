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

P='tests/v1/metrics/test_perf_metrics.py'
D='tests/v1/metrics/test_stats.py'
group(P,all_lines(P),'diagnostic.analytic_work_estimate','纯CPU解析HF配置与手工ExecutionContext，MockModelConfig调用实际getter，真实解析器/估算公式；检查声明FLOPs/字节数或比例/聚合，不运行模型也不验证实测性能。具体参数、控制分支和断言见源码。')
group(D,all_lines(D),'diagnostic.request_statistics_record','CPU手工请求统计、缓存来源和EngineCore输出；真实统计累加及Msgpack往返；检查记录数值/字段。prefill扣减部分在测试自身计算，不代表真实scheduler执行；不存在缓存传输或模型计算。')
unit('diagnostic.analytic_work_estimate','模型计算量与内存流量估算','diagnostic',['diagnostic_record_contract','parallel_topology','numeric_format','request_phase'],'根据声明模型结构、并行配置、计算精度与执行上下文生成FLOPs和读写字节估算，分项聚合及变更比例符合指定公式；不能表达的组件按约定拒绝。',['parsed_fields','flops_breakdown','read_bytes','write_bytes','per_rank_average','invalid_component'],[('vllm/v1/metrics/perf.py',333),('vllm/v1/metrics/perf.py',407),('vllm/v1/metrics/perf.py',491),('vllm/v1/metrics/perf.py',773),('vllm/v1/metrics/perf.py',1134),('vllm/v1/metrics/perf.py',1460)])
unit('diagnostic.request_statistics_record','请求统计记录与来源累加','diagnostic',['diagnostic_record_contract','request_phase','cache_state'],'请求与调度统计保留声明字段，缓存来源、抢占和计算量分类按规则累加；输出状态更新和序列化保留对应请求记录。',['token_counts','hit_sources','request_id','scheduler_details','watermark_flag','serialized_fields'],[('vllm/v1/metrics/stats.py',121),('vllm/v1/metrics/stats.py',308),('vllm/v1/metrics/stats.py',426),('vllm/v1/metrics/stats.py',474)])
for u in proposals:u.update(state_scope='diagnostic_artifact',semantic_domain='diagnostic_support')

N='tests/utils_/test_numa_utils.py'
group(N,[x for x in all_lines(N) if x not in [512,564,572,582,682]],'deployment.numa_process_binding','CPU真实NUMA配置控制函数，硬件探测/sysfs/numactl返回以monkeypatch替换；核查分rank参数、回退和上下文恢复，不执行真实CPU亲和/物理内存绑定。')
group(N,[512],'diagnostic.record_schema_and_encoding','numactl子进程返回与logger均为替身，核查格式化CPU affinity字符串及成功状态；不读取真实绑定。')
unit('deployment.numa_process_binding','进程NUMA绑定的拓扑解析与回退','preflight',['deployment_configuration','parallel_topology','configuration_resolution_profile'],'根据显式设置或可检测的本地设备拓扑生成CPU与内存绑定参数，自动策略遵守环境可用性与现有亲和限制；拒绝的策略降级并记录实际配置，子进程上下文退出后恢复环境和解释器。',['numactl_args','numa_nodes','effective_bind_flag','environment','spawn_executable','probe_failure'],[('vllm/utils/numa_utils.py',60),('vllm/utils/numa_utils.py',144),('vllm/utils/numa_utils.py',346),('vllm/utils/numa_utils.py',421),('vllm/utils/numa_utils.py',463),('vllm/utils/numa_utils.py',558),('vllm/utils/numa_utils.py',637)])
group(N,[564,582],'deployment.configuration_admissibility','真实ParallelConfig，人工非法NUMA节点和CPU列表，检查ValueError，不启用绑定。')
group(N,[572,682],'deployment.effective_configuration','真实ParallelConfig与deepcopy/replace，核查显式列表推导numa_bind、冲突拒绝以及自动策略副本和计算hash，不启动进程。')
helper_links={}
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
 oracle=f"{path}:{checks[0]['line']}";decorators='; '.join(ast.unparse(n) for n in node.decorator_list)
 calls=[dict(line=n.lineno,expression=ast.unparse(n)) for n in ast.walk(node) if isinstance(n,ast.Call)]
 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+decorators,rule_id='round16_source_review_runtime'))
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

# Additional reviewed production anchors for reused units.
for source_path,source_lines in [('vllm/config/parallel.py',[523,534,997]),('vllm/utils/numa_utils.py',[497])]:
 data=(SRC/source_path).read_bytes();ss=data.decode();tt=ast.parse(ss)
 for ll in source_lines:
  nn=next(n for n in ast.walk(tt) if isinstance(n,ast.FunctionDef) and n.lineno==ll)
  source_reviews.append(dict(path=source_path,line=ll,kind='source',snapshot='vllm',source_sha256=hashlib.sha256(data).hexdigest(),source_excerpt=ast.get_source_segment(ss,nn)))
(OUT/'product-source-evidence.json').write_text(json.dumps(source_reviews,ensure_ascii=False,indent=2))
assert set(assign)=={(r['path'],int(r['line'])) for r in rows}|set(skipped)
