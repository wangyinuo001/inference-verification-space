"""Source-reviewed assignments; AST only records evidence, never chooses semantics."""
import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集')
PUB=ROOT/'github_preview/inference-verification-space'; SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'; OUT=Path(__file__).resolve().parent
assign={}
def group(path,lines,unit,prop,conditions):
 for line in lines:
  key=(path,line)
  assert key not in assign,key
  assign[key]=(unit,prop,conditions)
C='tests/v1/kv_offload/cpu/test_manager.py'
cc='真实 CPUOffloadingManager 及 LRU/ARC 管理状态；输入为人工构造 OffloadKey 和 ReqContext；验证元数据/策略，不执行实际 KV 数据搬运。'
group(C,[154,168],'cache.event_payload_binding','存储事件携带写请求上下文；腾出容量的移除事件先于新存储事件，介质字段正确。',cc)
group(C,[236,261,273,308,331,350,366,390],'cache.offload_metrics','准备、完成、失败、跳过和读取操作后，缓存使用量、分配直方图和跳过计数与实际管理状态对应；收集后按约定清零。',cc)
group(C,[188,400,688,994,1022,1674,1706],'cache.eviction_outcome','容量不足与并发加载条件下保留受保护块，返回指定淘汰键、分配块和命中状态；失败写入清理新块。',cc)
group(C,[517],'cache.storage_layout_mapping','prepare_load 返回的 chunk_ids 与调用者 keys 顺序逐项对应，置换输入键同步置换块编号。',cc)
group(C,[554,589,628,927,943,1265,1612,1763],'cache.state_invariants','失败淘汰保持原缓存、重置清除旧请求分类；pending/ready、ARC 队列和可淘汰计数符合操作序列。',cc)
group(C,[571,667,728,755,788,834,883,962,1358,1380,1404,1426,1457,1571,1632,1650,1731,1746,1785,1828],'cache.eviction_candidate_order','根据请求访问、token 位置及 LRU/ARC 策略确定候选和顺序；重试或乱序完成不重复计算访问热度；批量扫描遵守断言中的上界。',cc)
group(C,[1061],'cache.event_metadata_lifetime','失败写入清理事件在重试重新拥有键后被裁剪；部分消费事件迭代器不重播已取事件。',cc)
group(C,[1082,1130,1152,1195,1231],'cache.offload_admission','仅在每请求去重后的使用次数达到阈值时接收新键；重试不增加次数，已就绪块仍可复用，超长 offer 能继续进展。',cc)
group(C,[1485,1525],'cache.offload_admission','请求完成按 KV group 转交去重后的访问和新写集合；写入失败调用 store-miss hook，单请求仅记录一次访问。',cc+' 其中 hook 用截获调用验证参数。')
group(C,[1808],'cache.offload_admission','增量写入按 token 位置维护次序，位置查询次数不超过测试声明的线性上界。',cc)
B='tests/v1/kv_offload/tiering/test_backpressure.py'
bc='真实背压探测器/决策策略；直接设置 EMA、完成数、时钟或失败次数；部分测试使用 DelayedSecondaryTierManager 模拟完成和丢弃，不验证真实远端写入。'
group(B,[139,144,149,154,159,172],'cache.event_configuration','文件/对象/P2P 介质与 locality 选择对应高低水位和失败阈值。',bc)
group(B,[189,201,213,225,242,333,368,391,417,526,587,632,640,658,669,858,862,873,891],'cache.offload_backpressure','完成延迟按单位字节归一并更新 EMA；预热、空闲衰减、迟滞、失败连续计数和重置决定压力及健康状态。',bc)
group(B,[296,455,469,536,566,651,707,712,719,727,734,746,760,770,783,831,885,911,928,933,939],'cache.offload_backpressure','依据压力与冷却期选择提交或丢弃存储，累计丢弃比例与设定对应；读取继续允许；完成/丢弃后释放相应 pin。',bc)
group(B,[485,502,817,825],'cache.offload_metrics','丢弃的存储数和块数准确累加、按 tier 输出，收集或 reset 后清零。',bc)
# 813 only asserts inheritance, not system behavior: explicitly do not promote.
F='tests/v1/kv_offload/test_factory.py'
fc='OffloadingSpecFactory 与 CPU/Tiering spec；make_offloading_config 提供配置；worker/shared-region 构造部分用替身记录参数。'
group(F,[57,63,68,73,78,85,491,662,683,695],'deployment.effective_configuration','注册名、默认值、动态模块和配置决定 offloading spec 类型与生效 replicated-layout 标志；外部构造接收原配置。',fc)
group(F,[91,111,329,348,434,458,667],'cache.capacity_plan','按 worker 字节、共享/分片复制数和对齐确定 chunk 字节数与容量；零字节产生空缓存；标准化 group 几何保持。',fc)
group(F,[121,144,166,193,219,231,249,258,267,297],'cache.offload_metrics','容量信息反映 spec 的字节、块及 token 容量，计入 TP 副本和多 group；缺少推导条件时报告未知，零容量准确报告零。',fc+' 297 使用实际共享区，其余按各测试构造路径。')
group(F,[365,403,519,558,590,630],'cache.storage_layout_mapping','worker 构造按复制/分片布局选择 rank 与共享区域参数；禁用共享区或容量为零时不构造 mmap。',fc+' 这里是参数传递断言，不是跨进程数据一致性。')
group(F,[706,715,720,780],'deployment.configuration_admissibility','拒绝无注册 spec、缺失容量、重复注册及不支持跨节点 secondary-tier 配置；拒绝时不构造 worker/region。',fc)
group(F,[727,738,760],'cache.offload_metrics','根据 store threshold 注册跳过计数器，分配直方图保留声明的桶边界。',fc)
P='tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py'
pc='通过真实 KVCacheManager/混合 attention-Mamba 元数据管理器和人工请求检查 block/hash/refcount；调度分割或 connector 接口部分用 SimpleNamespace/MagicMock；不执行 GPU block copy。'
group(P,[44,61],'cache.hit_accounting','connector 是否支持 divergent local hybrid hits 决定 lookup 入口及返回的块、命中长度和 partial 标志。',pc)
group(P,[166,371,783,1685,1846,1921,2020,2086,2155,2227,2292,2348,2503,2533,2769],'cache.hit_accounting','混合缓存命中长度受 hash 对齐、实际各组块覆盖、checkpoint retention 与 speculative 回退边界共同约束，返回块数和 hash token 长度对应。',pc)
group(P,[237,291,330,1982],'schedule.segment_continuity','prefill 分段满足预算和 checkpoint/hash 边界，预算小于块长时允许受限进展，Eagle 末端停在可复用边界。',pc)
group(P,[446,643,688,751,818,857,884,1602,2820],'cache.state_invariants','checkpoint 保留、覆盖和暂存策略决定 hash 发布与 block 预留，覆盖旧状态撤销旧 hash，CoW 新块初始不携带旧 hash。',pc)
group(P,[520,593,911,1389,1466,1528,2412,2457],'cache.lifetime','部分命中后继续执行为需写块安排独立 CoW 目标，保存原前缀；同一步未完成拷贝的命中延期，复制状态和独立快照按各组规则保留。',pc)
group(P,[983,1061,1216,2588,2629,2680,2724],'disagg.kv_mapping','边界状态 handoff 携带准确的 group、block ID 和 token 边界，包含 CoW 目标和恢复 prefill 后边界，按接口规定排空一次。',pc)
group(P,[1127,1168,1757,1810],'cache.lifetime','请求完成前登记边界块；释放请求后不再排出悬空 handoff，CoW 源/目标保留至对应执行步处理后释放。',pc)
group(P,[1269,1325],'cache.storage_layout_mapping','截取命中块按各组 block size 保留稀疏占位和原数组；仅允许可由外部状态补足的 Mamba 短组。',pc)
E='tests/distributed/test_events.py'
ec='ZMQ event publisher 与 conftest.MockSubscriber（该辅助类实际创建 SUB/DEALER socket）；限定本机 TCP/inproc、topic、buffer_steps 和测试 timeout。'
group(E,[43,62,80,103,125,165,191,230],'distributed.message_delivery','事件发送与回放保留 payload、顺序和 topic；有限回放缓冲保留最近窗口。高流量测试只要求 10 秒内收到至少 90%，不是无损保证。',ec)
# assign event functions by exact names below to avoid guessing source lines.
EVENT_NAMES={
 'test_basic_publishing':('distributed.message_delivery','发送事件的序号、时间戳、数量及各项值与原事件一致。'),
 'test_multiple_events':('distributed.message_delivery','多批事件按连续序号完整接收。'),
 'test_replay_mechanism':('distributed.message_delivery','从请求序号开始回放保留窗口内全部事件且顺序正确。'),
 'test_replay_includes_topic':('distributed.message_delivery','回放消息含正确 topic、连续序号及规定数量。'),
 'test_full_replay_with_slow_subscriber':('distributed.message_delivery','慢订阅者重复回放超过高水位的批次仍收到全部序号和结束标记。'),
 'test_buffer_limit':('distributed.message_delivery','回放只保留最近 buffer_steps 个批次且序号对应窗口。'),
 'test_topic_filtering':('distributed.message_delivery','匹配 topic 的订阅者接收消息，其他 topic 在指定超时内不收到。'),
 'test_high_volume':('distributed.message_delivery','高流量条件下十秒内接收至少 90% 批次且序号有序。'),
 'test_data_parallel_rank_tagging':('distributed.message_delivery','不同 rank 的端点发送各自数量的事件，并标记来源 data_parallel_rank。'),
 'test_event_publisher_factory':('deployment.effective_configuration','事件开关与 publisher 名决定实例；rank 端口偏移生效且原配置不改写。'),
 'test_offset_endpoint_port_keeps_port_zero':('distributed.remote_endpoint_binding','rank 端口偏移保留端口零，使操作系统分配独立端口。'),
 'test_ephemeral_publishers_report_bound_endpoints':('distributed.remote_endpoint_binding','两个 rank 的发布和回放端点均为实际节点地址与不同非零端口，订阅与回放可用。'),
 'test_ephemeral_bind_needs_a_node_ip':('deployment.configuration_admissibility','缺少可公布节点 IP 时拒绝把 wildcard 作为临时端口端点。'),
}
# Replace provisional event lines with the exact reviewed symbol mapping.
assign={k:v for k,v in assign.items() if k[0]!=E}
etree=ast.parse((SRC/E).read_text())
for n in ast.walk(etree):
 if isinstance(n,ast.FunctionDef) and n.name in EVENT_NAMES:
  u,p=EVENT_NAMES[n.name];group(E,[n.lineno],u,p,ec)
K='tests/distributed/test_packed_tensor.py'
kc='生产 pack/unpack 函数与测试张量；广播 roundtrip 使用内存复制的 mock 通信组；IPC roundtrip 使用独立子进程和真实 CUDA IPC；此轮仅源码审查。'
group(K,[77,101,143,191,233,460,485,505,521,530,548,578,605,622,637,655,883,912],'distributed.tensor_pack_contract','打包/解包保留名字、shape、dtype、值和声明视图别名；空输入、容量阈值和 GPU UUID 按各断言处理。',kc)
group(K,[343,387],'distributed.tensor_buffer_lifetime','跨 CUDA stream 延迟读取须等待接收完成，pending replay 的旧缓冲不被下一次接收覆盖；检查实际地址和 sentinel 值。',kc+' 使用人工广播组和 GPU sleep 控制并发窗口。')
group(K,[776,805,830,855],'distributed.message_delivery','独立进程通过 CUDA IPC 逐块接收，确认拷到 CPU 后再复用生产缓冲；最终名字、shape、dtype 与数值对应原张量。',kc)

units=[
 ('cache.offload_metrics','缓存卸载统计与操作对应','cache','存储、加载、失败、跳过与收集操作产生的计数、容量标签、使用量与直方图符合实际操作及配置；重置遵守统计周期。','vllm/v1/kv_offload/cpu/manager.py',494),
 ('cache.offload_admission','缓存卸载接收与请求去重','cache','按每请求去重访问、阈值和容量决定接收哪些存储键；失败重试不重复计数，已就绪缓存仍可访问。','vllm/v1/kv_offload/cpu/manager.py',268),
 ('cache.offload_backpressure','缓存卸载背压与恢复','cache','根据完成延迟、失败历史和声明水位更新压力状态，按策略限流存储；健康、冷却、恢复和重置保持对应决策及资源引用。','vllm/v1/kv_offload/tiering/backpressure.py',341),
 ('distributed.tensor_pack_contract','张量打包与解包对应','transfer','传输前后的张量表示保留逻辑名字、顺序、shape、dtype、值及声明的共享视图关系；块容量与设备身份决定打包边界与接收可用性。','vllm/distributed/weight_transfer/packed_tensor.py',27),
 ('distributed.tensor_buffer_lifetime','异步张量传输缓冲生命周期','transfer','缓冲在接收完成后才能读取，在所有待执行消费者完成前不能覆盖或复用；设备 stream 的依赖与存储生命周期保持一致。','vllm/distributed/weight_transfer/packed_tensor.py',196),
]
proposals=[dict(id=i,name=n,kind='obligation',phase=ph,state_scope='shared',conditions=[],property=p,observable=[],oracle='按具体测试输入、操作序列、直接断言与辅助断言检查；参见 crosswalk.tsv 和 evidence.json。',evidence={'vllm':[{'path':sp,'line':ln,'kind':'source'}]}) for i,n,ph,p,sp,ln in units]
details={
'cache.offload_metrics':(['cache_state','cache_storage_transition','deployment_configuration'],['cache_usage','allocation_histogram','skipped_stores','capacity_labels']),
'cache.offload_admission':(['cache_state','cache_eviction_policy_profile','concurrent_request_profile'],['admitted_keys','request_access_count','store_miss_callbacks']),
'cache.offload_backpressure':(['cache_storage_transition','storage_backend_profile','cache_event_profile'],['pressure_state','latency_ema','admit_or_drop','pinned_blocks']),
'distributed.tensor_pack_contract':(['tensor_transport_profile','numeric_format','shared_buffer_profile'],['tensor_names','shape_dtype','values','storage_alias','chunk_metadata']),
'distributed.tensor_buffer_lifetime':(['tensor_transport_profile','shared_resource_lifecycle_profile','execution_variant'],['receive_events','pending_replay','buffer_addresses','observed_values'])}
for u in proposals:u['conditions'],u['observable']=details[u['id']]
# Specialize packaging checks: do not claim value roundtrip for metadata-only tests.
for line,prop in {
77:'空迭代器不调用 mock 广播。',460:'unpack 返回 packed 存储的视图，清零 packed 后结果同步清零。',485:'pack 保留名字顺序、shape、dtype，并使用 uint8 承载 packed 字节。',505:'容量阈值限制单次打包的张量个数。',521:'空迭代器打包返回 None。',530:'单张量超过容量时仍保留该张量并发出指定警告。',578:'跨多次 pack 消费迭代器后名称顺序完整。',605:'指定容量下 IPC producer 输出多个 chunk。',622:'IPC handle 以声明的 GPU UUID 为键。',637:'IPC dtype_names 为对应字符串。',655:'空输入的 IPC producer 不输出 chunk。',883:'接收端找不到设备 UUID 的 IPC handle 时抛出 ValueError。'}.items():
 old=assign[(K,line)];assign[(K,line)]=(old[0],prop,old[2])
index=list(csv.DictReader((PUB/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
existing=json.loads((PUB/'semantic_space/space_seed.json').read_text()); valid={u['id'] for u in existing['units']}|{u['id'] for u in proposals}
labels={u['id']:u['name'] for u in existing['units']+proposals}
rows=[];evidence=[];hashes={}; skipped=[]
for entry in index:
 if entry['snapshot']!='vllm':continue
 path=entry['path'];line=int(entry['line']); key=(path,line)
 if key not in assign:continue
 if entry['review_status']=='property_reviewed':skipped.append(entry);continue
 unit,prop,cond=assign[key];assert unit in valid
 data=(SRC/path).read_bytes();source=data.decode();tree=ast.parse(source)
 nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno==line];assert len(nodes)==1,key
 node=nodes[0]; checks=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Assert):checks.append({'line':n.lineno,'expression':ast.unparse(n)})
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and (n.func.attr.startswith('assert') or n.func.attr=='raises'):checks.append({'line':n.lineno,'expression':ast.unparse(n)})
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and (n.func.id.startswith('verify_') or n.func.id.startswith('check_')):checks.append({'line':n.lineno,'expression':ast.unparse(n),'helper':n.func.id})
 assert checks,key
 hashes[path]=hashlib.sha256(data).hexdigest()
 decorators=[ast.unparse(n) for n in node.decorator_list]
 oracle=f"{path}:{checks[0]['line']}"
 family_context=prop
 # Entry-level promises are restricted to checks actually present here.
 # Broad reviewed family descriptions remain context only, not claims.
 unique_checks=list(dict.fromkeys(c['expression'] for c in checks))
 prop=labels[unit]+'。本项断言：'+'；'.join(unique_checks)
 if key==(C,154):
  prop='prepare_store 返回非空；take_events 返回的存储事件保留同一个 req_context 对象。'
 if key==(C,168):
  prop='腾出容量产生的 removed 事件先于新 stored 事件，所有事件 medium 与 manager.medium 一致。'
 helper_names={c['helper'] for c in checks if 'helper' in c}
 helper_sources=[{'name':h.name,'line':h.lineno,'end_line':h.end_lineno,'source':ast.get_source_segment(source,h)} for h in ast.walk(tree) if isinstance(h,ast.FunctionDef) and h.name in helper_names]

 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+'; '.join(decorators),rule_id='round10_source_review_runtime'))
 evidence.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,end_line=node.end_lineno,source_sha256=hashes[path],definition_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),unit_id=unit,property=prop,checks=checks,helper_sources=helper_sources,family_context=family_context,review_basis='逐函数读取调用、断言、辅助检查及 mock/真实传输边界；line 分组为审查结果，不是运行时关键词推断。',source_excerpt=ast.get_source_segment(source,node)))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':[{'snapshot':'vllm','path':p,'source_sha256':h} for p,h in hashes.items()], 'entry_reviews':evidence},ensure_ascii=False,indent=2))
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2))
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
summary={'mapped_entries':len(rows),'reviewed_files':len(hashes),'already_reviewed_excluded':len(skipped),'by_file':{p:sum(r['path']==p for r in rows) for p in hashes},'by_unit':{u:sum(r['unit_id']==u for r in rows) for u in sorted({r['unit_id'] for r in rows})},'not_promoted':['test_backpressure.py::test_shared_base 仅继承关系断言','test_events.py::test_null_publisher 仅执行不报错，无事件交付断言'],'runtime_execution':False}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
