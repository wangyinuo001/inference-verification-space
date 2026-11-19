"""Source-reviewed semantic assignments; AST retains exact per-entry evidence."""
import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集');PUB=ROOT/'github_preview/inference-verification-space';SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).resolve().parent
assign={};helper_links={}
def group(path,lines,unit,condition):
 for line in lines:
  assert (path,line) not in assign,(path,line)
  assign[(path,line)]=(unit,condition)
P='tests/v1/worker/test_gpu_pcp_manager.py'
c='真实 PCPManager 局部布局计算；多数用 CPU tensor 和人工 rank 配置，跨 rank 通信由 monkeypatch 替身提供；标 skipif CUDA 的入口使用实际设备张量但未启动多进程模型。'
group(P,[78,87,210],'deployment.configuration_admissibility',c)
group(P,[103,153,192,239,316,363,398,412,428,466,511,567,593,607,621,649,724,788],'compute.batch_partition_relation',c)
group(P,[137],'model.forward_buffer_view',c)
group(P,[271,284],'model.forward_buffer_storage_sharing',c)
group(P,[305],'cache.storage_layout_mapping',c)
group(P,[834],'deployment.effective_configuration',c)
U='tests/v1/worker/test_gpu_ubatch_slicing.py'
c='真实微批切片与 DP 决策；切片张量在 CPU，_sync_dp 用预置矩阵替代 all-reduce。GPU 标记入口使用 toy-model 与真实 CUDA 运行器，未运行真实模型或多进程通信。'
group(U,[158,227,264,379,423,446,667,799,857,1027,1043,1052],'compute.batch_partition_relation',c)
group(U,[201,590,844],'model.forward_buffer_storage_sharing',c)
group(U,[339,351,359,367,419,929,985,1010],'compute.kernel_dispatch_binding',c)
group(U,[578],'deployment.configuration_admissibility',c)
group(U,[635,876],'compute.execution_relation',c)
group(U,[682],'compute.microbatch_failure_delivery',c)
A='tests/v1/worker/test_attn_utils.py'
c='attention 辅助函数局部检查：配置和 backend 大多使用 SimpleNamespace/替身；布局/块复制使用真实张量，host-pool profiling 入口有实际设备门槛；逐项断言区分配置、元数据和内容。'
group(A,[69,119,365],'deployment.effective_configuration',c)
group(A,[166,231,268,329,426],'compute.kernel_dispatch_binding',c)
group(A,[473,524],'shared_resource.lifetime',c)
group(A,[592,629,689,799,903,921,939,948],'cache.storage_layout_mapping',c)
group(A,[663,714,759,956],'cache.block_copy_relation',c)
N='tests/v1/kv_connector/unit/test_nixl_desc_geometry.py'
c='真实 NIXL worker 几何计算；_RecordingNixl 记录描述符与 transfer 调用，transfer 本身为空操作；人工远端元数据；CPU 张量，非真实 RDMA。'
group(N,[199,218,235,255,272,334,485,540,702,740,787,968,1233,1327,1513,1524,1547,1582,1619,1671,1791,1909],'cache.storage_layout_mapping',c)
group(N,[908,1201,1215,1261,1312],'cache.transfer_request_byte_ranges',c+' _run_hetero_case 对描述符 token 对齐、范围与完整覆盖作断言，并模拟 DMA 写零检查接收清理。')
group(N,[2238],'cache.transfer_request_byte_ranges',c+' 本项以 ctypes.memmove 实际执行记录的 READ 描述符，逐页比较 CPU 字节并检查请求外存储未变化。')
helper_links[(N,908)]=['_assert_local_writes_within']
for line in [1201,1215,1261,1312]:helper_links[(N,line)]=['_run_hetero_case','_assert_local_writes_within']
helper_links[(N,2238)]=['_run_recorded_copies']
C='tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py'
c='真实 coordinator 与 ExternalCachedBlockPool；预设哈希集合、混合 Attention/Mamba block spec；仅计算命中长度及 load/store 掩码，没有外部存储或 KV 数据传输。'
group(C,[61,71,81,87],'cache.storage_existence_accounting',c)
group(C,[124,134,144,154,171,185,201,224,244,447,570,753],'cache.hit_accounting',c)
group(C,[266,273,283,299,313,327,363,376,398,407,417,428,474,511,519,535,546,584,611,637,660,686,727],'cache.storage_retention_mask',c)
S='tests/v1/kv_connector/unit/test_mooncake_store_connector.py'
c='真实 connector/scheduler/client 调用，按入口以 MagicMock 替代 store、socket 或 worker；异步查询使用真实 Python executor 和人工 gated_recv；无远端存储和网络传输。'
group(S,[82,243,286],'deployment.effective_configuration',c)
group(S,[152,224],'deployment.configuration_admissibility',c)
group(S,[315,1114,1632],'cache.hit_accounting',c)
group(S,[338,376,427,666,687,707,726,745,1181],'transfer.connector_delegation',c)
group(S,[408],'cache.event_configuration',c)
group(S,[451],'transfer.operation_metrics',c)
group(S,[470,497,541,553,564,581,593,610,639],'cache.event_payload_binding',c)
group(S,[763,787,1157,1167],'transfer.control_message_contract',c)
group(S,[831,876,935,972,1007,1033,1041,1054,1081,1218],'cache.async_lookup_lifecycle',c)
group(S,[1352,1373],'shared_resource.lifetime',c)
group(S,[1395],'deployment.runtime_mode_binding',c)
group(S,[1421,1501],'cache.lifetime',c+' 真实 BlockPool ref_cnt 随分块 store job 的完成更新；完成通知人工构造。')
group(S,[1670],'cache.async_lookup_lifecycle',c+' 真实调度器与请求 progress；store worker.lookup 用替身。')
M='tests/v1/kv_connector/unit/test_mooncake_stats.py'
c='真实 stats 聚合、锁和 Prometheus registry；成功/失败样本人工注入，worker.engine 是 MagicMock，检查指标而非吞吐实测或真实传输。'
group(M,[27,33,58,74,89,99,117,174,191,197,225,244,255,261,274,355,371,388,394,405],'transfer.operation_metrics',c)
X='tests/v1/kv_connector/unit/test_multi_connector.py'
c='MultiConnector 包装真实逻辑，子 connector 通常是 MagicMock；只检查参数、选择和汇总，不据此声称真实数据传输。真实 LLM 集成入口另注明。'
group(X,[189,239,252,256,282,554,1150],'transfer.connector_delegation',c)
group(X,[266],'deployment.configuration_admissibility',c)
group(X,[546],'deployment.effective_configuration',c)
group(X,[1104,1296,1328,1349,1372],'deployment.effective_configuration',c)
group(X,[636,645,654,681,712,734,766,799,829,840,856,904,937,965,998,1027],'transfer.operation_metrics',c)
group(X,[310],'cache.tier_transfer_content','实际 LLM(Llama-3.2-1B-Instruct) 与 ExampleConnector 目录存储集成；两组本地目录内容逐文件比较，检查 scheduler/worker 回调顺序；不是远端网络传输实验。')
helper_links[(X,310)]=['_compare_directories']
PROPOSALS=[
 dict(id='compute.batch_partition_relation',name='批次拆分与重组对应',phase='model_execution',conditions=['request_batch_profile','parallel_topology','execution_variant','request_phase'],property='批次按请求与 token 边界划分后，各切片的长度、位置、缓存槽位及填充标记与原批次对应；按约定重排或合并后保留原始数据次序。',observable=['request_indices','query_start_loc','seq_lens','slot_mapping','padding_mask','merged_hidden_states'],sources=[('vllm/v1/worker/gpu/ubatch_utils.py',1),('vllm/v1/worker/gpu/pcp_manager.py',1)]),
 dict(id='compute.microbatch_failure_delivery',name='微批执行错误归属与返回',phase='model_execution',conditions=['execution_variant','request_batch_profile'],property='微批执行中出现异常时，将错误与对应微批标识一起交付调用方；在被检验的失败时序下，调用线程和失败微批线程能够退出。',observable=['exception_type','cause','microbatch_id','thread_alive'],sources=[('vllm/v1/worker/gpu/ubatch_utils.py',1)]),
 dict(id='cache.transfer_request_byte_ranges',name='缓存传输的请求范围与字节对应',phase='transfer',conditions=['transfer_topology','parallel_topology','execution_layout','cache_state'],property='传输描述符把同一请求的逻辑 token 和缓存组件对应到本地与远端页；写入落在该请求拥有的区间内，所需页被完整覆盖，接收后的尾部按声明规则处理。',observable=['local_descriptors','remote_descriptors','request_owned_ranges','covered_tokens','destination_values','untouched_bytes'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py',1),('vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py',1)]),
 dict(id='cache.storage_retention_mask',name='缓存保留与装载范围',phase='cache',conditions=['operator_profile','cache_state','execution_layout','speculative_mode'],property='缓存保存和装载掩码按 Attention 窗口、块大小、保留间隔与回放需求确定；命中范围内需要的块保持可恢复，掩码在不同长度和缓存组之间遵循声明的对应关系。',observable=['store_mask','lookup_mask','load_mask','cache_group','block_boundary'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py',1)]),
 dict(id='transfer.connector_delegation',name='传输组件的操作转发与结果归属',phase='transfer',conditions=['transfer_topology','deployment_configuration','request_phase'],property='组合或包装传输组件将操作参数转发到按角色、命中和资源规则选定的子组件，返回对应结果；子组件元数据按位置和类型保留，需广播的通知传递到所有相应组件。',observable=['selected_connector','call_arguments','returned_result','child_metadata','operation_notifications'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py',1),('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py',1)]),
 dict(id='transfer.operation_metrics',name='传输操作的统计与报告',phase='transfer',conditions=['transfer_topology','concurrent_request_profile','diagnostic_dump_pair'],property='传输成功、失败、过期和耗时样本按引擎及组件归属汇总；聚合、序列化、快照与重置保持计数和样本对应，报告类型符合指标接口。',observable=['transfer_count','failure_count','bytes','duration','engine_label','snapshot','prometheus_sample'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/stats.py',1),('vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py',1)]),
 dict(id='transfer.control_message_contract',name='传输控制消息编码与解析',phase='transfer',conditions=['message_transport_profile','transfer_topology'],property='查询、重置与响应消息具有可区分的操作标记；序列化和解析保留命中长度、尾部键及声明字段，调用方按响应状态得到对应结果。',observable=['message_tag','encoded_frames','decoded_response','success_status'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/protocol.py',1)]),
 dict(id='cache.async_lookup_lifecycle',name='异步缓存查询的复用与失效',phase='cache',conditions=['cache_state','request_phase','inflight_resource_profile','concurrent_request_profile'],property='异步查询按请求和查询内容保存 pending/完成状态；相同有效查询复用结果，内容变化或过期按规则刷新，取消、重置和关闭清除关联状态并限制新查询；预取数量遵守容量限制。',observable=['lookup_future','request_id','query_content','pending_count','cached_result','lookup_count','discarded_entries'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py',1),('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py',1)]),
]
SOURCE_LINES={'vllm/v1/worker/gpu/ubatch_utils.py':54,'vllm/v1/worker/gpu/pcp_manager.py':330,'vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py':2193,'vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py':138,'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py':264,'vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py':469,'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/connector.py':382,'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/stats.py':93,'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/protocol.py':53,'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py':2810,'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py':244}
proposals=[]
for u in PROPOSALS:
 sources=u.pop('sources');u.update(kind='obligation',framework_scope=['vllm'],state_scope='shared',oracle='逐入口记录的实际值、结构、异常或调用断言；对应测试的替身、执行层次与输入条件见 evidence/crosswalk。',evidence={'vllm':[dict(path=p,line=(553 if u['id']=='compute.microbatch_failure_delivery' else SOURCE_LINES[p]),kind='source') for p,l in sources]});proposals.append(u)
existing=json.loads((PUB/'semantic_space/space_seed.json').read_text());labels={u['id']:u['name'] for u in existing['units']+proposals};valid=set(labels)
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
 for helper in helper_links.get(key,[]):
  hn=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==helper];assert len(hn)==1,(key,helper)
  checks+=extract_checks(hn[0],helper)
 assert checks,key
 prop=labels[unit]+'。本项断言：'+'；'.join(dict.fromkeys(c['expression'] for c in checks))
 oracle=f"{path}:{checks[0]['line']}";decorators='; '.join(ast.unparse(n) for n in node.decorator_list)
 calls=[dict(line=n.lineno,expression=ast.unparse(n)) for n in ast.walk(node) if isinstance(n,ast.Call)]
 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+decorators,rule_id='round12_source_review_runtime'))
 reviews.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,end_line=node.end_lineno,unit_id=unit,property=prop,source_sha256=hashes[path],definition_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),checks=checks,calls=calls,source_excerpt=ast.get_source_segment(source,node),helper_checks=helper_links.get(key,[]),review_method='逐项核对调用、fixture/辅助函数与断言；AST 仅导出证据；未执行测试'))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
file_reviews=[]
for p,h in hashes.items():
 s,t=trees[p];helpers=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(s,n)) for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and not n.name.startswith('test')]
 file_reviews.append(dict(snapshot='vllm',path=p,source_sha256=h,helper_sources=helpers))
(OUT/'evidence.json').write_text(json.dumps(dict(file_reviews=file_reviews,entry_reviews=reviews),ensure_ascii=False,indent=2))
source_reviews=[]
for u in proposals:
 for v in u['evidence']['vllm']:
  data=(SRC/v['path']).read_bytes();s=data.decode();t=ast.parse(s);candidates=[n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.lineno==v['line']]
  excerpt=ast.get_source_segment(s,candidates[0]) if candidates else '\n'.join(s.splitlines()[max(0,v['line']-1):v['line']+60])
  source_reviews.append(dict(**v,snapshot='vllm',source_sha256=hashlib.sha256(data).hexdigest(),source_excerpt=excerpt))
(OUT/'product-source-evidence.json').write_text(json.dumps(source_reviews,ensure_ascii=False,indent=2))
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2))
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
summary=dict(mapped_entries=len(rows),reviewed_files=len(hashes),already_reviewed_excluded=len(skipped),by_file={p:sum(r['path']==p for r in rows) for p in hashes},by_unit={u:sum(r['unit_id']==u for r in rows) for u in sorted({r['unit_id'] for r in rows})},runtime_execution=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
excluded=[dict(snapshot='vllm',path=S,line=n,reason='测试中直接复制 RESET handler 分支，未调用生产 process_request；mock 顺序断言不能作为生产路径证据。') for n in [1261,1317]]
(OUT/'excluded-reviewed.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
