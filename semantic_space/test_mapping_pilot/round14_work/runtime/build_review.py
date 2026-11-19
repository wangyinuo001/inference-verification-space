"""Explicit assignments after reading each entry and shared fixtures; AST exports evidence."""
import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集');PUB=ROOT/'github_preview/inference-verification-space';SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).resolve().parent
assign={};helper_links={}
def group(path,lines,unit,condition):
 for line in lines:
  assert (path,line) not in assign,(path,line)
  assign[(path,line)]=(unit,condition)
D='tests/v1/kv_connector/unit/test_decode_bench_connector.py'
c='真实 DecodeBenchConnector、CPU 缓存张量及 Scheduler；DecodeBenchTestRunner 用人工 token/结束标志构造 ModelRunnerOutput，不运行模型前向。填充值为 benchmark 人工状态。'
group(D,[162,206,243,293,682,800,896,914,924,959,995,1061,1098],'diagnostic.cache_probe_initialization',c)
group(D,[337,408,463,505,558,656,723,765],'cache.storage_layout_mapping',c+' 此项只断言填充元数据、块编号或 token 数量。')
group(D,[616,638],'diagnostic.cache_probe_initialization',c+' 此项仅断言不再填充/单 token 不填充的元数据。')
R='tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py'
c='真实 CPU Scheduler/缓存块池及 Nixl scheduler 逻辑；create_model_runner_output 人工提供 token 和 finished_recving，匹配数部分 patch；不运行远程传输或模型前向。'
group(R,[45,91,161,182,336,576,681,864,912,948,1025,1092],'schedule.progress',c)
group(R,[126],'cache.lifetime',c)
group(R,[211,502],'disagg.transfer_gate',c)
group(R,[434],'cache.state_invariants',c)
group(R,[770],'abort.cleanup',c)
group(R,[791],'schedule.segment_continuity',c)
S='tests/v1/kv_connector/unit/test_simple_cpu_offload_metrics.py'
group(S,[100,110,125,142,154,171,185,227,268,282,293,304,316,333,353,364,385,402,412,437],'transfer.operation_metrics','真实统计/调度器统计逻辑；Prometheus metric_types 使用 _FakeMetric 收集调用；人工增加 counter/填写 transfer 账本或发送完成标志，不执行 CPU/GPU 数据迁移。make_scheduler 创建真实小型 BlockPool。')
U='tests/v1/kv_connector/unit/test_umbp_events.py'
group(U,[111,145,169,196,209,275,295,325,376,413,436,457,477],'cache.event_payload_binding','UMBP 生产事件/worker/scheduler 逻辑；_WorkerHandle/_SchedulerHandle 为测试替身，驻留查询 Future/命中集合及完成数量人工注入；核查事件与账本条件，不代表真实后端驻留或跨 rank 传输。ROCm+MORI 收集门槛。')
group(U,[500],'transfer.operation_metrics','真实 UMBPStoreConnectorStats 累计与合并；人工统计输入。')
N='tests/v1/kv_connector/unit/test_nixl_heartbeat.py'
group(N,[45,71,82,102,122,138,156,172,215,234],'cache.lifetime','Nixl scheduler/worker 通过 __new__ 构造并填最小账本，heartbeat 更新、限频和回收运行生产函数；worker xfer_stats 为 MagicMock，时间部分 monkeypatch；不发网络心跳，也不实际释放设备页。')
I='tests/v1/kv_connector/unit/test_hisparse_stats.py'
group(I,[14,19,31,40,53,66,75,88],'transfer.operation_metrics','真实 HiSparse 统计累积与重建；输入为人工 snapshot/usage/eviction。Prometheus 类型替换为记录 inc/set/observe 的 _NamedFake；不运行缓存搬运。')
W='tests/v1/worker/test_gpu_worker_weight_transfer.py'
group(W,[94],'deployment.effective_configuration','Worker 的 reload_weights 真实包装，model_runner 为记录当前 VllmConfig 的替身；不加载权重。')
group(W,[123],'model.parameter_binding','CPU nn.Module/Parameter 与真实 LoRA wrapper，核查 named_parameters/get_parameter/get_parameter_for_reload 指向同一权重对象。')
group(W,[107,139,164,178,210,224,239,254,275,282,288,294,308,324],'model.weight_update_session','Worker __new__ 后设置最小字段；_RecordingEngine/_RecordingModelRunner 记录转发/会话标志，故障通过替身注入；断言更新顺序、rank 对应、异常与清理，不检查传输后的参数值。')
E='tests/v1/worker/test_extensible_kv_cache.py'
group(E,[66,81,373,408,437,480,533,567,635],'cache.virtual_allocation_contract','requires_cuda + VMM fixture，真实 ExtensibleKVCache 与 allocate_kv_cache；按层/块布局提交物理页、读写真实 CUDA 张量，核查地址、视图、清零和物理占用；本批仅源码审查未运行。')
group(E,[339],'cache.storage_layout_mapping','CPU 缓存规格与真实 pad_kv_cache_segments；核查配置的层偏移/步幅/大小，不分配 GPU。')
group(E,[118,182,230,611],'cache.capacity_plan','按具体入口：118 纯字节公式；182/230 accelerator 可用量/allocator 与硬件属性被 monkeypatch；611 requires_cuda+VMM 实际测粒度。各项断言对应预算到块数/限额。')
group(E,[294],'resource.allocator_scope','真实 ExtensibleKVCache set_torch_memory_limit/free；buffer.free 与 torch.cuda.set_per_process_memory_fraction 是替身，核查限额复位调用。')
G='tests/v1/worker/test_gpu_worker.py'
group(G,[23],'deployment.configuration_admissibility','HAS_TRITON=False 与 is_cpu=False 被注入，检查 Worker 初始化拒绝。')
group(G,[36,61],'model.sleep_parameter_retention','CPU Linear/Module 参数；Worker sleep 保存/恢复函数，区分 draft 私有和 target 共享参数；不执行真实休眠或设备卸载。')
group(G,[78],'model.execution_variant_relation','model_runner.load_model 是记录线程数替身；实际 CPU torch.compile 与 CompileCounter 检查线程切换后输出和编译次数。')
group(G,[121,159],'resource.allocator_scope','需 native CUDA/ROCm allocator；真实大/小设备分配或 allocator setting 快照，检查临时 split 设置和退出恢复；失败由测试显式 raise。')
group(G,[195],'resource.allocator_scope','平台/allocator backend 被 patch 为 cudaMallocAsync；检查不读取快照、不设置原生 allocator。')
group(G,[212,368,376,388,400],'cache.capacity_plan','内存快照/设备平台为人工输入，profile_run 为提前抛 Profiled 的替身；212 仍实际调用 decode logits 行数函数并检查配置上下文释放；其余测试只检查 fallback 数值。')
group(G,[283,297,317],'deployment.startup_plan_reuse','plan_env 隔离临时缓存目录；硬件身份与可用容量为 SimpleNamespace，真实 JSON 保存加载及指纹函数；不执行 GPU profiling。')
group(G,[422],'distributed.tensor_buffer_lifetime','PP send handle、通信组和 model_runner 为记录顺序替身；检查 wait 在 forward 前及保留 tensor handle，未运行 NCCL 或真实在途张量。')
group(G,[473,502],'deployment.effective_configuration','JIT monitor/runner/cache 依赖以记录函数替换；核对启用条件、warmup容量和初始化延后规则。')
group(G,[550],'distributed.resource_fallback_agreement','world group 与 all_reduce_int 为替身；核查 MAX 归约参数及根据人工归约结果返回的原因。')
group(G,[577],'cache.virtual_allocation_contract','真实 extend_kv_cache 包装；connector pool、commit 为记录调用的替身，commit 抛 Committed 停止；检查初始化顺序及不支持内存池拒绝。')
group(G,[700,718,740,753,773,787],'resource.allocator_scope','实际 Worker reload/discard 包装；cache release/recommit/allocator 限额和模型加载均记录或抛异常，依据人工 free/layer bytes 检查借用/恢复及失败前拒绝顺序。')
O='tests/v1/kv_connector/unit/offloading_connector/test_worker.py'
group(O,[205,224,240,258,278,293,334,356,377],'cache.offload_job_completion','真实 OffloadingConnectorWorker；spec/backend worker 为 MagicMock，完成/失败 TransferResult 人工输入；检查提交、等待、失败请求及完成计数，不核验搬运后的字节。_make_worker 分配当前设备上的占位张量。')
group(O,[387],'deployment.effective_configuration','外部 BareExternalOffloadingSpec 继承生产基类；构造 Worker 后断言默认 replicated_layout 为 False，未调用实际存储。')
group(O,[406,634,676],'cache.storage_layout_mapping','allocate_kv_cache 实际分配当前平台设备张量；spec.get_worker 为 mock，核查传入 CanonicalKVCaches 的组引用、页大小、形状及分片映射；未执行后端传输。')
F='tests/v1/kv_connector/unit/test_hf3fs_connector.py'
group(F,[48],'cache.storage_layout_mapping','仓库提供的 Hf3fsMockClient 使用本地文件；核查创建后文件大小，不是远程 HF3FS。')
group(F,[65,116],'cache.tier_transfer_content','仓库 Hf3fsMockClient 本地文件后端真实读写 CPU tensor，按 dtype 和偏移比较回读值；CUDA event 可为 MagicMock；不验证分布式 HF3FS。')
group(F,[87,101],'cache.storage_layout_mapping','Hf3fsMockClient 本地文件的边界错误返回 -1；不验证远程存储。')
group(F,[163,175,191,197,213,226],'transfer.operation_metrics','真实 HF3FSKVConnectorStats，人为提交时延与失败状态，核对统计/序列化/清空。')
PROPOSALS=[
 dict(id='diagnostic.cache_probe_initialization',name='性能探针缓存按声明范围与格式初始化',phase='cache',conditions=['numeric_format','execution_layout','cache_state'],property='用于性能探针的人工缓存按指定组、块和生命周期写入；常数/随机值按缓存格式编码，需清零的布局清零，未选范围保持原值；一次性初始化和逐请求填充按声明模式执行。',observable=['selected_blocks','fill_metadata','cache_values','numeric_encoding','unselected_values'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py',563),('vllm/distributed/kv_transfer/kv_connector/v1/decode_bench_connector.py',679)]),
 dict(id='model.weight_update_session',name='权重更新会话顺序与失败状态',phase='model_execution',conditions=['parameter_lifecycle','parameter_mapping','parallel_topology'],property='更新会话按开始、分块接收和完成推进；载荷按目标 rank 与目标模型转发，错误后活动状态、未完成标记和目标恢复符合声明策略，目标模型与草稿模型的完成状态分别维护。',observable=['active_session','incomplete_targets','forwarded_payload','target_rank','target_model','reset_calls'],sources=[('vllm/v1/worker/gpu_worker.py',1744),('vllm/v1/worker/gpu_worker.py',1790),('vllm/v1/worker/gpu_worker.py',1836)]),
 dict(id='cache.virtual_allocation_contract',name='缓存物理提交与视图保持对应',phase='cache',conditions=['execution_layout','cache_state','configuration_resolution_profile'],property='预留地址与实际提交页按块和层布局对应；扩展保留先前内容并清零新页，缩减、整理或释放后重建按声明规则重置内容；对外视图与传输描述符只覆盖实际提交范围。',observable=['committed_blocks','physical_bytes','segment_offsets','view_addresses','view_extents','cache_values'],sources=[('vllm/v1/worker/extensible_kv_cache.py',177),('vllm/v1/worker/extensible_kv_cache.py',206)]),
 dict(id='resource.allocator_scope',name='临时内存策略与资源借用按作用域恢复',phase='service',conditions=['deployment_configuration','configuration_resolution_profile','inflight_resource_profile'],property='初始化、profiling 或权重重载期间按后端能力临时调整分配器策略及缓存内存；正常和异常退出恢复先前限额与所有权状态，已经释放的资源不由非所有者恢复。',observable=['allocator_settings','memory_limit','release_calls','recommit_calls','operation_order','rejection'],sources=[('vllm/v1/worker/gpu_worker.py',465),('vllm/v1/worker/extensible_kv_cache.py',630),('vllm/v1/worker/extensible_kv_cache.py',347)]),
 dict(id='model.sleep_parameter_retention',name='休眠参数保留与共享关系',phase='model_execution',conditions=['parameter_lifecycle','parameter_mapping'],property='休眠保留项按名称匹配选择，草稿私有参数与共享目标参数区分；恢复将保存值复制回原参数而保持存储地址，清除保存记录，未匹配的普通模式报告错误。',observable=['selected_parameter_names','shared_parameter_identity','restored_value','storage_pointer','saved_parameter_state'],sources=[('vllm/v1/worker/gpu_worker.py',295),('vllm/v1/worker/gpu_worker.py',307),('vllm/v1/worker/gpu_worker.py',326)]),
 dict(id='deployment.startup_plan_reuse',name='启动容量计划的身份匹配与复用门禁',phase='preflight',conditions=['deployment_configuration','configuration_resolution_profile','parallel_topology'],property='持久化容量计划依据配置、软件、设备和 rank 身份匹配；格式版本及可用容量满足条件后应用，显式容量配置优先，身份或容量失配时保留重新测量路径。',observable=['plan_fingerprint','schema_version','free_memory','configured_kv_bytes','applied_plan'],sources=[('vllm/v1/worker/startup_plan.py',41),('vllm/v1/worker/startup_plan.py',87),('vllm/v1/worker/startup_plan.py',134)]),
]

# These entries invoke the shared cleanup oracle; export that helper too.
for line in [211,336,502,576,681]:helper_links[(R,line)]=[('tests/v1/kv_connector/unit/utils.py','assert_scheduler_empty')]
proposals=[]
for u in PROPOSALS:
 sources=u.pop('sources');u.update(kind='obligation',framework_scope=['vllm'],state_scope='shared',oracle='逐入口记录的实际值、结构、异常或调用断言；对应测试的替身、执行层次与输入条件见 evidence/crosswalk。',evidence={'vllm':[dict(path=p,line=l,kind='source') for p,l in sources]});proposals.append(u)
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
 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+decorators,rule_id='round14_source_review_runtime'))
 reviews.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,end_line=node.end_lineno,unit_id=unit,property=prop,source_sha256=hashes[path],definition_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),checks=checks,calls=calls,source_excerpt=ast.get_source_segment(source,node),helper_checks=helper_links.get(key,[]),review_method='逐项核对调用、fixture/辅助函数与断言；AST 仅导出证据；未执行测试'))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
file_reviews=[]
for p,h in hashes.items():
 s,t=trees[p];helpers=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(s,n)) for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('test')]
 file_reviews.append(dict(snapshot='vllm',path=p,source_sha256=h,helper_sources=helpers))
for helper_path in ['tests/v1/kv_connector/umbp_test_utils.py','tests/v1/kv_connector/unit/utils.py','tests/v1/simple_kv_offload/test_scheduler.py','tests/v1/kv_connector/unit/test_simple_cpu_offload_connector.py']:
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


excluded=[dict(snapshot='vllm',path=F,line=147,reason='只调用本地 mock client.flush/close，未显式检查状态或持久化效果；未确认更强资源清理性质。')]
(OUT/'excluded-reviewed.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
assigned_keys=set(assign);seen_keys={(r['path'],int(r['line'])) for r in rows}|set(skipped)
assert assigned_keys==seen_keys,sorted(assigned_keys-seen_keys)
assert len(rows)==len({(r['path'],r['line']) for r in rows})
