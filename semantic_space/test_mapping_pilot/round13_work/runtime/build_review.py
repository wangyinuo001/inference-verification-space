"""Per-entry source review of connector semantics. AST is evidence export only."""
import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集');PUB=ROOT/'github_preview/inference-verification-space';SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).resolve().parent
assign={};helper_links={}
def group(path,lines,unit,condition):
 for line in lines:
  assert (path,line) not in assign,(path,line)
  assign[(path,line)]=(unit,condition)
C='tests/v1/kv_connector/unit/test_umbp_core.py'
c='UMBP connector/data/scheduler/worker 真实局部逻辑；_SchedulerHandle/_WorkerHandle 及延时子类是测试替身，_EmbeddedRuntime 仅保存键集合；MemoryRuntime 另用 ctypes 复制真实 CPU 字节。文件有 ROCm+MORI 收集门槛，不代表每项运行了 GPU。'
group(C,[123,141,162,178,201,295,307,346,379,1643,1685,1729],'cache.namespace',c)
group(C,[132,190,224,397,410,428,445,479,486,509,700,995,1033,1092,1425],'cache.storage_layout_mapping',c)
group(C,[233,1455],'cache.tier_transfer_content',c+' 本项使用 MemoryRuntime 真实 CPU 字节复制，断言目标块与源块相等且邻居未被写入。')
group(C,[329,543,678,1129,1154,1183,1205,1335,1355,1381,1818,1840],'cache.offload_job_completion',c)
group(C,[558,568,619,638,662,1540,1603],'cache.hit_accounting',c)
group(C,[574],'cache.event_payload_binding',c)
group(C,[593,600],'deployment.effective_configuration',c)
group(C,[717,766,807,838,866,1069],'cache.storage_retention_mask',c)
group(C,[895,948,1236,1263],'cache.lifetime',c)
group(C,[1276,1301],'abort.other_requests_progress',c)
H='tests/v1/kv_connector/unit/test_umbp_hybrid.py'
c='UMBP 混合缓存的真实 scheduler/worker 元数据逻辑；主要使用 _SchedulerHandle、_WorkerHandle 替身和人工命中；未将命中/保存计划当成实际模型输出对照。'
group(H,[79,127,153,193,244,290,361,451,563,700,738,776,1038],'cache.storage_retention_mask',c)
group(H,[416,818],'cache.hit_accounting',c)
group(H,[521],'deployment.configuration_admissibility',c)
group(H,[540,673,1122],'deployment.effective_configuration',c)
group(H,[625],'cache.state_invariants',c)
group(H,[841],'cache.lifetime',c)
group(H,[912],'cache.tier_transfer_content','CPU 参数使用 MemoryRuntime(ctypes)；GPU 参数使用生产 EmbeddedRuntime/MORI 后端；真实 BlockPool 与完整字节比较，检查 store/load 结束、所有权释放和目标外内容。')
group(H,[1138],'cache.offload_job_completion',c)
E='tests/v1/kv_connector/unit/test_umbp_embedded.py'
c='生产 EmbeddedRuntime 包装逻辑；部分 backend 是 SimpleNamespace 替身，lookup server 使用实际 Unix socket；逐入口区分元数据、任务状态与真实字节复制。'
group(E,[81,249,290],'cache.storage_layout_mapping',c+' bulk_worker client 仅记录参数并返回完成标志，未复制数据。')
group(E,[157],'cache.tier_transfer_content',c+' 本项 client.get/put 使用 ctypes.string_at/memmove 复制 CPU 内存，对全部 backing storage 作比较。')
group(E,[316],'cache.storage_layout_mapping',c+' 混合布局触发整批退回 range 路径，断言 materialize 后的范围参数相同；未核验传输内容。')
group(E,[355,410],'deployment.configuration_admissibility',c)
group(E,[362,399],'deployment.effective_configuration',c)
group(E,[422],'deployment.network_endpoint_representation',c)
group(E,[437,454,681],'transfer.peer_registration',c+' 测试启动真实 lookup socket 并检查冲突、陈旧路径及替换生命周期。')
group(E,[473,531],'cache.storage_existence_accounting',c+' client 键集合人工更改，真实 lookup RPC 检查聚合结果；不验证 KV 内容。')
group(E,[560,591,668],'cache.offload_job_completion',c)
group(E,[655],'cache.lifetime',c+' blocked_transfer 真实线程阻塞配人工 monotonic；超时仍保持任务 RUNNING，release 后完成。')
group(E,[755],'disagg.transfer_gate',c+' 独立调度器使用测试 MemoryRuntime，lookup 后清空对象模拟淘汰；检验失败终止或重算策略。')
M='tests/v1/kv_connector/unit/test_mooncake_connector.py'
c='生产 Mooncake worker/scheduler 局部逻辑；FakeMooncakeWrapper 的注册和传输返回成功但不复制数据，socket 多为 AsyncMock/_ScriptedSock；真实 HTTP bootstrap 入口单独注明。'
group(M,[83,114,142,260,343,465,1349,2224,2301,2326,2356,2398,2428,2451,2527,2538,2611,2632,2650,2683,2718,2725,3038,3071,3123,3207,3239,3267],'cache.storage_layout_mapping',c)
group(M,[547,596],'disagg.kv_mapping',c+' 只检查 scheduler 下发的 recv 元数据与块编号，不比较接收字节。')
group(M,[644,655,671,696,726,772],'transfer.peer_registration',c+' 本项 bootstrap 使用真实本机 socket/HTTP 服务；失败时序部分用 monkeypatch 注入。')
group(M,[933,952],'transfer.peer_registration',c+' httpx.AsyncClient 替身注入超时次数，检查重试上限。')
group(M,[1034],'deployment.runtime_mode_binding',c)
group(M,[1071,1091],'deployment.network_endpoint_representation',c)
group(M,[1097],'deployment.configuration_admissibility',c)
group(M,[1103,1330,1861,1948,1978,2011,2066],'cache.lifetime',c+' 检验持有/续租/过期账本及通知，不执行真实传输。')
group(M,[1274,1372,1439,1615,1643,1698,1788,2147],'disagg.transfer_gate',c)
group(M,[2755,2845,2925,3003,3025],'transfer.layout_handshake_reuse',c)
group(M,[3417],'cache.transfer_address_registration',c+' 注册/注销 engine 方法为 mock，核查调用的指针与容量。')
R='tests/v1/kv_connector/unit/test_moriio_connector.py'
c='真实 MoRIIO connector/scheduler，parallel groups 为单 rank MagicMock，底层引擎替换为 FakeMoRIIOWrapper；声明 ROCm/MORI 条件。多数只核查元数据，handshake 单独使用实际 ZMQ。'
group(R,[203],'cache.hit_accounting',c)
group(R,[359,414,471,1128],'disagg.kv_mapping',c)
group(R,[549],'cache.lifetime',c)
group(R,[574],'transfer.control_message_contract',c)
group(R,[597,1258],'cache.storage_layout_mapping',c)
group(R,[696],'transfer.peer_registration',c+' 本项真实本机 DEALER 连接 handshake 线程，解码响应为 MoRIIOAgentMetadata；未比较所有元数据字段。')
group(R,[759],'deployment.network_endpoint_representation',c)
group(R,[770,801,855,885,950,1072,1117],'cache.storage_retention_mask',c)
group(R,[1041,1058,1176,1219,1231],'deployment.effective_configuration',c)
group(R,[1083,1203,1244],'deployment.configuration_admissibility',c)
L='tests/v1/kv_connector/unit/test_lmcache_connector.py'
c='LMCache engine 返回人工事件；mock connector 绑定生产 get/update/take_events；LMCacheKVEvents 与 KVOutputAggregator 是真实聚合逻辑，未启动 LMCache 数据存储或多进程。'
group(L,[79,88,96,119,149,179,205,213,226,246,287,315,367,399,407,443,483,526,567,590,629,647,679],'cache.event_payload_binding',c)
group(L,[792],'cache.offload_job_completion',c+' LegacyLayerwiseEngine 是测试 stub，真实 connector 清空已耗尽的生成器列表；断言检查空步骤后列表为空。')
PROPOSALS=[
 dict(id='transfer.peer_registration',name='传输端点注册与生命周期',phase='transfer',conditions=['remote_endpoint_profile','transfer_topology','parallel_topology','inflight_resource_profile'],property='传输端点按引擎与 rank 身份注册并提供查询，重复注册、端点冲突、启动失败与超时按协议返回；关闭和重新启动正确处理所持端口或 socket，注册重试有界。',observable=['registered_engine','rank_endpoints','registration_response','retry_count','socket_ownership','startup_status'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_utils.py',127),('vllm/distributed/kv_transfer/kv_connector/v1/umbp/runtime/embedded.py',117)]),
 dict(id='transfer.layout_handshake_reuse',name='传输布局握手复用与失效',phase='transfer',conditions=['transfer_topology','execution_layout','cache_state'],property='传输双方依据布局身份复用已确认的地址关系；未知或失效布局触发重新发送，未支持布局确认的对端继续接收完整信息；缓存淘汰及重新注册后不复用旧布局。',observable=['layout_id','peer_layout_cache','acknowledged_peers','layout_miss_response','resend_payload','registration_generation'],sources=[('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py',3187)]),
]

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
 for helper in helper_links.get(key,[]):
  hn=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==helper];assert len(hn)==1,(key,helper)
  checks+=extract_checks(hn[0],helper)
 assert checks,key
 prop=labels[unit]+'。本项断言：'+'；'.join(dict.fromkeys(c['expression'] for c in checks))
 oracle=f"{path}:{checks[0]['line']}";decorators='; '.join(ast.unparse(n) for n in node.decorator_list)
 calls=[dict(line=n.lineno,expression=ast.unparse(n)) for n in ast.walk(node) if isinstance(n,ast.Call)]
 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+decorators,rule_id='round13_source_review_runtime'))
 reviews.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,end_line=node.end_lineno,unit_id=unit,property=prop,source_sha256=hashes[path],definition_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),checks=checks,calls=calls,source_excerpt=ast.get_source_segment(source,node),helper_checks=helper_links.get(key,[]),review_method='逐项核对调用、fixture/辅助函数与断言；AST 仅导出证据；未执行测试'))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
file_reviews=[]
for p,h in hashes.items():
 s,t=trees[p];helpers=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(s,n)) for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('test')]
 file_reviews.append(dict(snapshot='vllm',path=p,source_sha256=h,helper_sources=helpers))
helper_path='tests/v1/kv_connector/umbp_test_utils.py'
helper_data=(SRC/helper_path).read_bytes();helper_source=helper_data.decode();helper_tree=ast.parse(helper_source)
file_reviews.append(dict(snapshot='vllm',path=helper_path,source_sha256=hashlib.sha256(helper_data).hexdigest(),review_role='shared_fixture',helper_sources=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(helper_source,n)) for n in helper_tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]))
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

excluded=[dict(snapshot='vllm',path=M,line=972,reason='本项自行创建 asyncio loop、线程和 failing_listener，再检查 run_coroutine_threadsafe 的异常传播；未调用生产 Mooncake listener，不能确认生产传输路径。')]
(OUT/'excluded-reviewed.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
(OUT/'dependencies.json').write_text(json.dumps(dict(previous_batch='round12_work/runtime',unit_ids=['cache.storage_retention_mask','transfer.control_message_contract']),ensure_ascii=False,indent=2))
assigned_keys=set(assign);seen_keys={(r['path'],int(r['line'])) for r in rows}|set(skipped)
assert assigned_keys==seen_keys,sorted(assigned_keys-seen_keys)
assert len(rows)==len({(r['path'],r['line']) for r in rows})
