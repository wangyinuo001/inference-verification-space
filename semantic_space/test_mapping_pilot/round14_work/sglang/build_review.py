# -*- coding: utf-8 -*-
"""Export source reviews of explicitly read lifecycle/adapter/configuration tests."""
import ast,csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; PROJECT=ROOT.parents[1]
SRC=PROJECT/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'; OUT=Path(__file__).parent; BASE='test/registered/unit/'
TRACE='diagnostic.trace_context_lifecycle'; METRIC='diagnostic.metric_adapter_contract'
STORE='diagnostic.baseline_artifact_lifecycle'; HOOK='deployment.hook_execution_semantics'
LOAD='scheduling.load_snapshot_delivery'; REG='deployment.extension_dispatch_contract'
S='python/sglang/srt/'; T='python/sglang/test/'
sources={
'observability/test_trace.py':[S+'observability/trace.py'],
'observability/test_ray_wrappers.py':[S+'observability/ray_wrappers.py',T+'observability/fake_ray.py',S+'observability/metrics_collector.py'],
'test_precision_baseline_store.py':[T+'precision_baseline_store.py'],
'plugins/test_hook_registry.py':[S+'plugins/hook_registry.py'],
'mem_cache/test_registry.py':[S+'mem_cache/registry.py'],
'managers/test_load_snapshot_backends.py':[S+'managers/load_snapshot.py'],
'model_executor/test_pool_configurator.py':[S+'model_executor/pool_configurator.py',S+'utils/common.py',S+'mem_cache/common.py'],
 'tokenizer/test_tiktoken_tokenizer.py':[S+'tokenizer/tiktoken_tokenizer.py']}
contexts={
'observability/test_trace.py':'本地 TraceReqContext，显式 start/end/event 时间戳、trace level、模块过滤与线程状态；启用组以本地 OTel provider 构造 span，未验证远端导出。无断言的方法仅记录无异常调用。',
'observability/test_ray_wrappers.py':'用 fake_ray 向 sys.modules 注入记录调用的假 Counter/Gauge/Histogram 与 ReplicaId；检查 API 适配、标签隔离、bucket 参数和调用实参，没有部署 Ray。',
'test_precision_baseline_store.py':'测试辅助设施；临时本地文件、模拟 HF API、manifest 行与 capture_signature；检查基线选择、上传准备、归档和重试。未比较模型数值，也未实际访问 HF。',
'plugins/test_hook_registry.py':'隔离 HookRegistry，构造临时 Python 模块、函数、类及继承关系；根据 BEFORE/AFTER/AROUND/REPLACE 组合检查调用结果、顺序、引用传播与异常。',
'mem_cache/test_registry.py':'构造缓存后端配置与能力标志，用 Mock 替代各缓存构造器；检查注册、选路、构造实参及包装关系，不执行实际 KV 缓存操作。',
'managers/test_load_snapshot_backends.py':'显式 dp_rank 与负载字段，经临时 SHM 文件和本机 ZMQ IPC 收发；配置检查另构造 node_rank、worker 数和负载均衡策略。测试源码含 sleep 与 warmup 重试，本轮未运行。',
'model_executor/test_pool_configurator.py':'Mock ModelRunner 及 CPU 环境，固定元素字节数、head/layer/SWA规格、page、chunk、并行与推测配置；以测试中独立容量公式和精确上界核对内存计划，未分配真实 GPU 池。',
'tokenizer/test_tiktoken_tokenizer.py':'Mock 底层 tokenizer 与显式 Jinja 模板；检查包装层的调用参数、返回字段或文本片段，不验证真实词表的编码正确性。'}
def unit_for(s,n):
 cls=n.split('::')[0]; short=n.split('::')[-1]
 if s=='observability/test_trace.py':
  return None if cls in ['TestDataclasses','TestSpanAttributes'] else TRACE
 if s=='observability/test_ray_wrappers.py':return METRIC
 if s=='test_precision_baseline_store.py':return STORE
 if s=='plugins/test_hook_registry.py':return HOOK
 if s=='mem_cache/test_registry.py':return REG if cls in ['TestRegisterRadixCacheBackend','TestCreateTreeCacheRouting'] else 'deployment.runtime_mode_binding'
 if s=='managers/test_load_snapshot_backends.py':
  return 'deployment.effective_configuration' if cls in ['TestFactoryFunctions','TestZmqReaderOwner','TestZmqAddr'] else LOAD
 if s=='model_executor/test_pool_configurator.py':return 'deployment.runtime_mode_binding' if cls=='TestFactory' else 'cache.capacity_plan'
 if s=='tokenizer/test_tiktoken_tokenizer.py':
  if cls=='TestConstants':return None
  if cls=='TestTiktokenProcessor':return 'input.multimodal_binding'
  if short=='test_apply_chat_template_no_tokenize':return 'input.prompt_text_rendering'
  if short=='test_apply_chat_template_with_tokenize':return 'input.chat_template_tokenization'
  return 'request.field_binding'
 raise AssertionError((s,n))
implicit={
'TestTraceReqContextDisabled::test_all_methods_noop':{'ctx.trace_req_start','ctx.trace_req_finish','ctx.trace_slice_start','ctx.trace_slice_end','ctx.trace_slice','ctx.trace_event','ctx.trace_set_root_attrs','ctx.trace_set_thread_attrs','ctx.abort','ctx.rebuild_thread_context'},
'TestTraceReqContextDisabled::test_trace_set_thread_info_disabled':{'trace_set_thread_info'},
'TestTraceReqContextEnabled::test_trace_req_finish_without_start':{'ctx.trace_req_finish'},
'TestTraceReqContextEnabled::test_trace_slice_end_empty_stack':{'ctx.trace_slice_end'},
'TestTraceReqContextEnabled::test_trace_set_root_attrs':{'ctx.trace_set_root_attrs'},
'TestTraceReqContextEnabled::test_trace_set_root_attrs_no_span':{'ctx.trace_set_root_attrs'},
'TestTraceReqContextEnabled::test_trace_set_thread_attrs':{'ctx.trace_set_thread_attrs'},
'TestTraceReqContextEnabled::test_trace_slice_nested_parent':{'ctx.trace_slice','ctx.trace_slice_end'},
'TestZmqRoundTrip::test_zmq_writer_noblock_without_reader':{'writer.write'}}
cache={}; nodes={};helpers={};contextsrc={}
def seg(t,n):return '\n'.join(t.splitlines()[n.lineno-1:n.end_lineno])
for suffix in sources:
 p=BASE+suffix;t=(SRC/p).read_text();cache[p]=t;tree=ast.parse(t);helpers[p]=[];contextsrc[p]=[]
 for top in tree.body:
  if isinstance(top,ast.ClassDef):
   items=[(top.name+'::'+x.name,x) for x in top.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
   contextsrc[p].append({'line':top.lineno,'class':top.name,'bases':[ast.unparse(x)for x in top.bases],'decorators':[ast.unparse(x)for x in top.decorator_list]})
   for x in top.body:
    if isinstance(x,(ast.Assign,ast.AnnAssign)):contextsrc[p].append({'line':x.lineno,'source':seg(t,x)})
  elif isinstance(top,(ast.FunctionDef,ast.AsyncFunctionDef)):items=[(top.name,top)]
  else:
   if isinstance(top,(ast.Assign,ast.AnnAssign,ast.Import,ast.ImportFrom)):contextsrc[p].append({'line':top.lineno,'source':seg(t,top)})
   continue
  for name,n in items:
   if n.name.startswith('test'):nodes[(p,name,n.lineno)]=n
   else:helpers[p].append({'name':name,'line':n.lineno,'source':seg(t,n)})
results=[];entries=[];support=[];missing=[]
for r in csv.DictReader((ROOT/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'):
 if r['snapshot']!='sglang' or r['review_status'] in ['property_reviewed','supporting_record_reviewed','outside_current_scope']:continue
 key=(r['path'],r['test_name'],int(r['line']))
 if key not in nodes:continue
 p=r['path'];t=cache[p];n=nodes[key];s=p[len(BASE):];u=unit_for(s,r['test_name']);body=seg(t,n)
 assertions=[];calls=[];inputs=[]
 for a in ast.walk(n):
  if isinstance(a,ast.Assert):assertions.append({'line':a.lineno,'kind':'assert','expression':ast.unparse(a)})
  elif isinstance(a,ast.Call):
   fn=ast.unparse(a.func);expr=ast.unparse(a);calls.append({'line':a.lineno,'target':fn,'expression':expr})
   if fn.split('.')[-1].startswith('assert') or fn.endswith('.raises'):assertions.append({'line':a.lineno,'kind':'assertion_call','expression':expr})
  elif isinstance(a,(ast.Assign,ast.AnnAssign,ast.For,ast.AsyncFor)):inputs.append({'line':a.lineno,'expression':ast.unparse(a)})
 for c in calls:
  if c['target'] in implicit.get(r['test_name'],set()):assertions.append({'line':c['line'],'kind':'implicit_no_exception','expression':'完成调用且无异常：'+c['expression']})
 assertions.sort(key=lambda a:a['line'])
 ev={**{k:r[k] for k in ['snapshot','path','test_name','line']},'unit_id':u,'test_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'assertions':assertions,'input_statements':inputs,'calls':calls,'test_source':body,'fixture_and_helper_sources':helpers[p],'module_and_class_constants':contextsrc[p],'implementation_files':sources[s],'review_basis':'实际阅读测试体、共同辅助函数与对应实现；按本项断言描述，不将测试标题或注释当作验证结果。'}
 if u is None:
  ev['role']='常量表、枚举或记录容器的结构检查；不是推理请求路径的执行检查';support.append(ev);continue
 if not assertions:missing.append(ev);continue
 rr={k:r[k] for k in ['snapshot','path','test_name','line']};rr.update(unit_id=u,property='本项实际检查：'+'；'.join(' '.join(a['expression'].split())for a in assertions),oracle_source='; '.join(f'{p}:{a["line"]}'for a in assertions),conditions=contexts[s],rule_id='round14_sglang_lifecycle_adapters_source_review_v1');results.append(rr);entries.append(ev)
new={TRACE,METRIC,STORE,HOOK,LOAD};fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
for file,rs in [('crosswalk.tsv',results),('existing-units-crosswalk.tsv',[r for r in results if r['unit_id'] not in new]),('proposed-units-crosswalk.tsv',[r for r in results if r['unit_id'] in new])]:
 with (OUT/file).open('w') as fp:w=csv.DictWriter(fp,fields,delimiter='\t');w.writeheader();w.writerows(rs)
paths=set(cache)|{x for xs in sources.values() for x in xs}
reviews=[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()}for p in sorted(paths)]
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':reviews,'entries':entries,'supporting_records':support,'unconfirmed':missing,'count':len(results),'scope':'配置和缓存容量、负载快照、插件扩展、分词包装与诊断支持设施；本轮只做源码核对。'},ensure_ascii=False,indent=2)+'\n')
proposals=[]
def proposal(uid,name,phase,state,cond,prop,obs,oracle,anchors,diagnostic=False):
 ev=[]
 for p,target in anchors:
  t=(SRC/p).read_text();tree=ast.parse(t);line=next(n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==target)
  ev.append({'path':p,'line':line,'kind':'source','source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()})
 proposals.append({'id':uid,'name':name,'kind':'obligation','phase':phase,'state_scope':state,'conditions':[cond],'property':prop,'observable':obs,'oracle':oracle,'framework_scope':['sglang'],'semantic_domain':'diagnostic_support'if diagnostic else'inference_request','evidence':{'sglang':ev}})
proposal(TRACE,'追踪上下文与事件生命周期','diagnostic','diagnostic_artifact','trace_lifecycle_context','追踪开关、模块和等级控制记录是否建立；请求与线程上下文在创建、嵌套片段、事件归属、序列化、结束和中止时按声明规则保存与清理。',['enabled','root_span','slice_stack','event_cache','serialized_context','thread_identity'],'构造显式时间戳与调用序列，检查上下文、栈、事件和序列化字段；无显式断言的入口仅检查调用无异常。',[(S+'observability/trace.py','TraceReqContext'),(S+'observability/trace.py','get_otlp_span_exporter')],True)
proposal(METRIC,'指标适配与标签隔离','diagnostic','diagnostic_artifact','metric_adapter_context','指标适配器按声明规则转换名称、桶边界和调用参数，将数值与对应标签传入指标后端；不同标签对象保持隔离，依赖缺失与非法标记调用按约定处理。',['metric_name','metric_kind','boundaries','value','tags','backend_call','dependency_error'],'用记录调用的指标后端核对操作、数值、标签及桶参数，检查标签副本、错误调用及采集器类绑定。',[(S+'observability/ray_wrappers.py','RayPrometheusMetric')],True)
proposal(STORE,'验证基线材料的选择与保存','diagnostic','diagnostic_artifact','baseline_artifact_context','验证基线按模型、采集签名和记录次序定位；保存、覆盖、归档和删除遵守声明的材料命名与保留策略，远端临时故障按约定重试。',['selected_run_path','manifest_rows','staged_files','retained_runs','deleted_runs','retry_calls'],'用本地临时材料和模拟存储 API，核对选中路径、manifest 字段、暂存文件及上传删除重试调用；此处不比较模型计算结果。',[(T+'precision_baseline_store.py','_select_latest_run'),(T+'precision_baseline_store.py','push_run'),(T+'precision_baseline_store.py','prune_old_runs')],True)
proposal(HOOK,'扩展钩子的调用与替换语义','preflight','deployment','hook_execution_context','扩展钩子按声明的前置、后置、环绕或替换模式传递参数和结果，保留类方法与静态方法的绑定，并遵守组合次序、引用传播和重复应用约束。',['return_value','argument_sequence','call_order','descriptor_binding','propagated_reference','conflict_warning'],'在隔离的临时模块中构造函数和类，检查钩子组合后的返回值、调用次序、继承绑定、冲突信息与重复调用次数。',[(S+'plugins/hook_registry.py','HookRegistry'),(S+'plugins/hook_registry.py','_wrap_fn')])
proposal(LOAD,'负载快照的按副本传递','scheduling','shared','load_snapshot_transport_context','负载快照按副本标识传给读取方，读取结果保留对应字段及最新观测值；尚未发布时返回空结果，传输资源按声明方式关闭。',['rank','load_fields','timestamp','latest_snapshot','empty_result','transport_cleanup'],'在本地共享文件和 IPC 通道写入带 rank 的显式快照，核对读取字段、更新与空结果；关闭后检查 IPC 资源。',[(S+'managers/load_snapshot.py','ShmLoadSnapshotReader'),(S+'managers/load_snapshot.py','ZmqShmLoadSnapshotReader')])
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2)+'\n')
conditions=[
('trace_lifecycle_context',['enabled','trace_level','module_filter','request_id','thread_rank','event_timestamps','slice_stack','serialization_state'],'区分采集器配置、上下文操作与实际导出；本地对象检查不保证远端观测已收到。'),
('metric_adapter_context',['metric_kind','name','label_names','label_values','replica_identity','bucket_boundaries','backend_availability','operation'],'固定指标后端与标签接口；Mock 调用证据只支持适配关系。'),
('baseline_artifact_context',['model','capture_signature','revision','manifest_rows','push_index','tensor_files','force','retention_days','weekly_archive','dry_run','HTTP_failure'],'验证支持材料按显式身份、采集配置和保留策略操作；不把下载或上传成功等同于模型数值正确。'),
('hook_execution_context',['target','hook_type','registration_order','descriptor_type','subclass','shared_references','apply_count'],'固定目标、描述符和钩子组合；每项以实际输入与调用顺序为准。'),
('load_snapshot_transport_context',['dp_size','dp_rank','transport','snapshot_sequence','timestamp','reader_state','endpoint','node_rank','reader_owner'],'保留每副本身份及更新顺序，区分共享内存和 IPC；本机时序检查不代表所有网络故障。')]
(OUT/'proposed-conditions.json').write_text(json.dumps([{'id':i,'type':'structured','fields':f,'observed_values':[],'constraint':c}for i,f,c in conditions],ensure_ascii=False,indent=2)+'\n')
existing={u['id']for u in json.loads((ROOT/'semantic_space/space_seed.json').read_text())['units']}
assert {r['unit_id']for r in results}<=existing|new|{REG}
assert len({(r['path'],r['test_name'],r['line'])for r in results})==len(results)
assert all(re.fullmatch(r'[^;]+:\d+(; [^;]+:\d+)*',r['oracle_source'])for r in results)
print(json.dumps({'count':len(results),'existing':sum(r['unit_id']not in new for r in results),'proposed':sum(r['unit_id']in new for r in results),'support':len(support),'unconfirmed':[(x['path'],x['test_name'])for x in missing],'files':len(reviews),'by_unit':{u:sum(r['unit_id']==u for r in results)for u in sorted({r['unit_id']for r in results})}},ensure_ascii=False,indent=2))
