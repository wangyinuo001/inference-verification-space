"""Explicit source-reviewed mappings. AST exports evidence, not semantic labels."""
import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集');PUB=ROOT/'github_preview/inference-verification-space';SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).resolve().parent
assign={}
def group(path,lines,unit,condition):
 for line in lines:
  assert (path,line) not in assign,(path,line)
  assign[(path,line)]=(unit,condition)
S='tests/v1/ec_connector/unit/test_session.py'
sc='真实 Producer/ConsumerSession 与 Xfer 状态机；data、transport、ZMQ连接用 MagicMock；人工 ACK、句柄99、deadline 驱动；仅验证控制状态和调用参数，不声明远端字节传输。'
group(S,[79,84,127,133,216,241,248,262,266,276,288,295,302,313,321,331,339,359,409,435,447,462,474],'transfer.session_progress',sc)
group(S,[143,163,223,231,392,398],'transfer.session_binding',sc)
group(S,[112,178,196,280,364,371],'shared_resource.lifetime',sc)
C='tests/v1/ec_connector/unit/cpu/scheduler/test_embedding_cache.py'
cc='EmbeddingCache 真实整数账本；字符串键与固定块容量；没有 mmap、CUDA 或 embedding 数值计算；并发测试使用 Python 线程。'
group(C,[24,35,42,48,55,61,68,79,84,91,102,113,121,174,179,187,347,357,365,383,388],'cache.state_invariants',cc)
group(C,[130,137,149,163,287,374],'cache.lifetime',cc)
group(C,[201,214,229,242,256,268,315],'cache.eviction_outcome',cc)
M='tests/v1/kv_connector/unit/offloading_connector/test_metrics.py'
mc='真实 OffloadingConnectorStats 聚合/归约；Prometheus 部分用 _FakeMetric 收集 labels/increments/observed/set_values；给定人工指标和 spec 配置。仅检查指标计算与传递。'
group(M,[45,178,187,196,260,312,345,369,384,404,441,461,489,515,563,610,649,681,696,720,749,775,816,837,860,888,900,923,937,950,971,1001,1018,1033,1045,1060,1073,1089,1111],'cache.offload_metrics',mc)
O='tests/v1/kv_offload/tiering/test_obj_tier.py'
oc='真实 ObjectStoreSecondaryTierManager；MockNixlAgent 只记录对象键集合和句柄状态，没有连接 S3，也不复制张量字节；人工异常与 PROC/DONE 返回驱动。'
group(O,[305],'deployment.configuration_admissibility',oc)
group(O,[314,317,324,334,346,377,449,458,470],'cache.storage_existence_accounting',oc)
group(O,[351,363,370,403,420,477,486,495,504,524,553,572],'cache.offload_job_completion',oc)
group(O,[592,645],'cache.lifetime',oc)
group(O,[670,690,707,725,738,746],'cache.event_payload_binding',oc)
group(O,[754,764],'cache.event_configuration',oc)
group(O,[780,791,801,814],'deployment.effective_configuration',oc)
group(O,[825,861],'cache.namespace',oc)
F='tests/v1/kv_offload/tiering/test_fs_tier.py'
fc='真实 FileSystemTierManager、临时磁盘文件和线程池；spec 为配置替身；部分测试注入 open/rename/O_DIRECT 失败。数据完整性仅由含张量/字节比较的入口确认。'
group(F,[216,222,234,292,526,573,614,666,694,748],'cache.storage_existence_accounting',fc)
group(F,[246,429,469,780,1016],'cache.tier_transfer_content',fc)
group(F,[320,335],'deployment.configuration_admissibility',fc)
group(F,[348],'deployment.effective_configuration',fc)
group(F,[368,378,393,590],'cache.offload_job_completion',fc)
group(F,[411,506],'distributed.queue_wait_shutdown',fc)
group(F,[826,848,870,885,916,972],'cache.event_payload_binding',fc)
group(F,[939,950],'cache.event_configuration',fc)
W='tests/v1/worker/test_utils.py'
wc='worker 辅助函数的局部调用；多数 worker/stream/event/region 通过替身构造，另有实际 CPU 张量视图与复制，少量设备条件测试；以逐入口源码区分。'
group(W,[60,661],'cache.tier_transfer_content',wc)
group(W,[501],'cache.block_copy_relation',wc)
group(W,[89,294],'deployment.effective_configuration',wc)
group(W,[147],'cache.offload_metrics',wc)
group(W,[184,195,209,223,588,613,749,789],'cache.storage_layout_mapping',wc)
group(W,[229,260,561,640,730,827,880,912,956,986,1010,1048,1081,1095,1117,1176,1201],'cache.host_mirror_ordering',wc)
group(W,[317,450],'resource.shared_host_region',wc)
group(W,[386,483],'resource.host_registration',wc)
group(W,[1341],'shared_resource.lifetime',wc)
group(W,[522,1245,1389,1468,1524,1556,1584,1607],'cache.layer_storage_binding',wc)
group(W,[1651],'cache.capacity_plan',wc)
T='tests/v1/kv_offload/tiering/test_tiering_offloading.py'
tc='真实 TieringOffloadingManager 和 CPU primary 账本；ExampleSecondaryTierManager 在进程内跟踪键与完成记录，mmap region 是基于 CPU tensor 的替身；hook 用 patch/wrap 记录；不验证远端 KV 内容。'
group(T,[161,196,262,808,848,894,1809,1840,1855,1875,1905,1976],'cache.offload_metrics',tc)
group(T,[1773,1786,1790,1955],'deployment.effective_configuration',tc)
group(T,[237,422,520,938,1042,1054,1659,1689,1707],'cache.storage_existence_accounting',tc)
group(T,[299,346,376,438,536,618,1021,1065,1102,1136,1155,1225,1262,1286,1319,1359,1389,1474,1502,1522,1545,1566],'cache.cascade_orchestration',tc)
group(T,[401],'cache.event_payload_binding',tc)
group(T,[471,703,742,761,775,783,795,1626],'cache.lifetime',tc)
group(T,[950],'cache.eviction_outcome',tc)
group(T,[973,992],'cache.eviction_candidate_order',tc)

PROPOSALS=[
 dict(id='cache.block_copy_relation',name='缓存逻辑块复制',phase='cache',conditions=['cache_state','execution_layout','numeric_format'],property='按逻辑块索引复制对应缓存内容，保留未指定目标与物理填充区域，复制后的目标值等于声明源值。',observable=['source_blocks','destination_blocks','copied_values','unchanged_padding'],sources=[('vllm/v1/worker/utils.py',933)]),
 dict(id='transfer.session_progress',name='传输会话进展与失败恢复',phase='transfer',conditions=['request_phase','inflight_resource_profile','transfer_topology'],property='传输会话依据确认、就绪、完成、失败和截止时间进入对应状态；可重试、终止和隔离结果分别交付，尚未终止的在途访问按状态保留。',observable=['ack_status','transfer_state','deadline','completed_keys','retryable_keys','quarantined_handles'],sources=[('vllm/distributed/ec_transfer/ec_connector/cpu/session.py',121),('vllm/distributed/ec_transfer/ec_connector/cpu/session.py',145)]),
 dict(id='transfer.session_binding',name='传输会话身份与兼容条件',phase='transfer',conditions=['transfer_topology','input_form','execution_layout'],property='传输请求、确认与完成通知携带一致的会话和对象标识；版本与兼容信息匹配后才能授权，读操作的源和目标块按请求对应。',observable=['session_id','object_hash','compatibility_hash','ack_status','source_blocks','destination_blocks'],sources=[('vllm/distributed/ec_transfer/ec_connector/cpu/session.py',294)]),
 dict(id='cache.offload_job_completion',name='缓存卸载任务完成与失败结果',phase='cache',conditions=['cache_storage_transition','storage_backend_profile','inflight_resource_profile'],property='异步卸载任务按身份汇报完成或失败，pending 任务继续等待；提交、内存注册、轮询或清理异常按约定处理，结果不被遗漏或重复交付。',observable=['job_id','pending_jobs','success','successful_keys','completion_results','registered_handles'],sources=[('vllm/v1/kv_offload/tiering/obj/manager.py',384),('vllm/v1/kv_offload/tiering/fs/manager.py',278)]),
 dict(id='cache.cascade_orchestration',name='多层缓存迁移与请求生命周期',phase='cache',conditions=['cache_storage_transition','cache_state','concurrent_request_profile','storage_backend_profile'],property='多层缓存根据各层命中与写入状态安排下沉和晋升，保留请求上下文；共享键不重复晋升，未完成写入延迟级联和请求结束，失败或重置清理相应编排状态。',observable=['lookup_result','submitted_jobs','request_context','pending_primary_stores','request_finalization','load_source'],sources=[('vllm/v1/kv_offload/tiering/manager.py',911)]),
 dict(id='cache.host_mirror_ordering',name='主机缓存镜像的执行次序与写入角色',phase='cache',conditions=['execution_variant','storage_writer_role_profile','parallel_topology','cache_storage_transition'],property='主机镜像按计算完成、设备传输和后续使用的依赖顺序提交；不同执行路径不重复或漏掉应镜像的层，共享区域只由指定写入者更新和确认。',observable=['event_waits','mirror_submissions','active_layers','transfer_completion','writer_rank'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/hisparse/worker.py',895)]),
 dict(id='cache.layer_storage_binding',name='模型层与缓存存储对应',phase='model_execution',conditions=['execution_layout','operator_profile','cache_state','forward_buffer_context'],property='模型层、执行器和缓存组绑定对应存储视图与跟踪状态；共享组复用同一状态，不同组隔离；重新绑定、复制与清理遵守地址和布局约束。',observable=['layer_cache','runner_cache_order','group_state_identity','storage_addresses','tracker_shape','copied_values'],sources=[('vllm/v1/worker/utils.py',844),('vllm/v1/worker/utils.py',896)]),
]
proposals=[]
for u in PROPOSALS:
 sources=u.pop('sources');u.update(kind='obligation',state_scope='shared',oracle='按逐入口显式断言检查状态、参数或实际数据；mock 控制测试和真实数据比较分别记录，见 crosswalk/evidence。',evidence={'vllm':[dict(path=p,line=l,kind='source') for p,l in sources]});proposals.append(u)
existing=json.loads((PUB/'semantic_space/space_seed.json').read_text());labels={u['id']:u['name'] for u in existing['units']+proposals};valid=set(labels)
rows=[];reviews=[];hashes={};skipped=[];missing=[]
index=list(csv.DictReader((PUB/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
for entry in index:
 if entry['snapshot']!='vllm':continue
 path=entry['path'];line=int(entry['line']);key=(path,line)
 if key not in assign:continue
 if entry['review_status']=='property_reviewed':skipped.append(key);continue
 unit,cond=assign[key];assert unit in valid,unit
 data=(SRC/path).read_bytes();source=data.decode();tree=ast.parse(source)
 nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno==line];assert len(nodes)==1,key
 node=nodes[0];checks=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Assert):checks.append(dict(line=n.lineno,expression=ast.unparse(n)))
  elif isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and (n.func.attr.startswith('assert') or n.func.attr=='raises'):checks.append(dict(line=n.lineno,expression=ast.unparse(n)))
 assert checks,key
 hashes[path]=hashlib.sha256(data).hexdigest()
 prop=labels[unit]+'。本项断言：'+'；'.join(dict.fromkeys(c['expression'] for c in checks))
 oracle=f"{path}:{checks[0]['line']}"
 decorators='; '.join(ast.unparse(n) for n in node.decorator_list)
 calls=[dict(line=n.lineno,expression=ast.unparse(n)) for n in ast.walk(node) if isinstance(n,ast.Call)]
 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+decorators,rule_id='round11_source_review_runtime'))
 reviews.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,end_line=node.end_lineno,unit_id=unit,property=prop,source_sha256=hashes[path],definition_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),checks=checks,calls=calls,source_excerpt=ast.get_source_segment(source,node)))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
file_reviews=[]
for p,h in hashes.items():
 s=(SRC/p).read_text();t=ast.parse(s);helpers=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(s,n)) for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and not n.name.startswith('test')]
 file_reviews.append(dict(snapshot='vllm',path=p,source_sha256=h,helper_sources=helpers))
(OUT/'evidence.json').write_text(json.dumps(dict(file_reviews=file_reviews,entry_reviews=reviews),ensure_ascii=False,indent=2))
source_reviews=[]
for u in proposals:
 for v in u['evidence']['vllm']:
  data=(SRC/v['path']).read_bytes();source_reviews.append(dict(**v,snapshot='vllm',source_sha256=hashlib.sha256(data).hexdigest()))
(OUT/'product-source-evidence.json').write_text(json.dumps(source_reviews,ensure_ascii=False,indent=2))
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2))
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
summary=dict(mapped_entries=len(rows),reviewed_files=len(hashes),already_reviewed_excluded=len(skipped),by_file={p:sum(r['path']==p for r in rows) for p in hashes},by_unit={u:sum(r['unit_id']==u for r in rows) for u in sorted({r['unit_id'] for r in rows})},runtime_execution=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
