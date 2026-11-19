import ast,csv,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;BASE=OUT.parents[1];PUB=BASE.parents[1]
SRC=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集/experiments/third_party/reference_repos/vllm-20261105')
def anchor(p,l):
 b=(SRC/p).read_bytes();s=b.decode();ns=[n for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.lineno==l];assert ns,(p,l)
 return dict(path=p,line=l,kind='source',snapshot='vllm',source_sha256=hashlib.sha256(b).hexdigest(),source_excerpt=ast.get_source_segment(s,ns[0]))
exist_sources=[anchor('vllm/v1/kv_offload/tiering/fs/manager.py',216),anchor('vllm/v1/kv_offload/tiering/obj/manager.py',290),anchor('vllm/v1/kv_offload/tiering/async_lookup.py',146),anchor('vllm/v1/kv_offload/tiering/async_lookup.py',234),anchor('vllm/v1/kv_offload/tiering/manager.py',699),anchor('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/coordinator.py',59),anchor('vllm/distributed/kv_transfer/kv_connector/v1/umbp/runtime/embedded.py',211)]
reg_sources=[anchor('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py',2518),anchor('vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py',2526)]
new=[dict(id='cache.storage_presence_verdict',name='存储存在性与查询状态对应',kind='obligation',framework_scope=['vllm'],phase='cache',state_scope='shared',conditions=['storage_backend_profile','cache_state'],property='根据查询键及所属分组报告存在、缺失或待查询状态；写入完成、读取失败和后端状态变化按声明规则更新判断，过期异步查询不得覆盖更新后的判断。多层路由及故障返回服从各自的查询策略。',observable=['query_keys','group_ids','lookup_verdict','query_generation','backend_state','lookup_diagnostics'],oracle='按测试注入的后端存在集合、传输完成和故障时序逐项核对查询返回值；各项只承诺已记录的布尔值、状态或元数据断言。',evidence={'vllm':[{k:a[k] for k in ['path','line','kind']} for a in exist_sources]}),dict(id='cache.memory_registration_lifecycle',name='缓存内存注册的注销与恢复',kind='obligation',framework_scope=['vllm'],phase='cache',state_scope='shared',conditions=['storage_backend_profile','host_transfer_profile','cache_state'],property='缓存注册记录保留底层地址与长度；注销和恢复按当前注册状态执行，重复操作不重复调用后端，后端错误沿声明路径报告。',observable=['base_addresses','registered_lengths','unregistered_flag','registration_calls','unregistration_calls'],oracle='对注册后端记录调用，核查注销与恢复次数、地址和长度，以及失败后的状态；传输内容另由数据检查承担。',evidence={'vllm':[{k:a[k] for k in ['path','line','kind']} for a in reg_sources]})]
corrections=[];extensions=[]
for rd in [11,12,13]:
 for r in csv.DictReader((BASE/f'round{rd}_work/runtime/crosswalk.tsv').open(),delimiter='\t'):
  old=r['unit_id'];dest=None;reason=''
  if old=='cache.storage_existence_accounting':dest='cache.storage_presence_verdict';reason='原单元要求物理键到逻辑页数量换算；本项检查存在布尔值、命中/缺失/待查询或后端变化，不检查页计数分母。'
  elif old=='cache.transfer_address_registration':dest='cache.memory_registration_lifecycle';reason='本项检查休眠注销/恢复的地址长度和幂等调用；未比较传输描述符范围与有效注册区间。'
  elif old=='model.forward_buffer_view':dest='model.forward_buffer_storage_sharing';reason='本项只核对公开输入缓冲区与内部来源是同一对象及其形状，没有抽取前向批次、补齐或字段保留操作。'
  if dest:
   prop=r['property'];pos=prop.find('本项断言')
   if pos>=0:prop=prop[pos:]
   corrections.append(dict(snapshot=r['snapshot'],path=r['path'],test_name=r['test_name'],line=int(r['line']),old_unit_id=old,new_unit_id=dest,reason=reason,property=prop,conditions=r['conditions'],source_batch=f'round{rd}_work/runtime'))
sharing_sources=[anchor('vllm/v1/worker/gpu/pcp_manager.py',447),anchor('vllm/v1/worker/gpu/pcp_manager.py',723),anchor('vllm/v1/worker/gpu/pcp_manager.py',827),anchor('vllm/v1/worker/gpu/pcp_manager.py',836),anchor('vllm/v1/worker/gpu/ubatch_utils.py',312)]
extensions=[dict(unit_id='model.forward_buffer_storage_sharing',framework_scope_add=['vllm'],evidence_add={'vllm':[{k:a[k] for k in ['path','line','kind']} for a in sharing_sources]},source_reviews=sharing_sources,semantic_correspondence='共同要求是前向缓冲区与声明来源/并发使用者之间的存储对应。vLLM PCP 使用持久输入来源，microbatch 分配独立重定位缓冲区；SGLang registry 通过注册共享键控制复用。',semantic_differences='vLLM 没有同一注册键算法；不能用这些测试支持 SGLang 的注册顺序无关或同键复用判断。这里保留来源绑定、并发使用者隔离及 data_ptr/值保持的单项判定。',suggested_property='前向缓冲区按声明的来源与并发使用者绑定存储；允许共享的缓冲区复用指定来源，同时存活且会独立写入的缓冲区保持隔离。注册键、注册顺序、批次视图及持久输入采用各实现的条件。')]
(OUT/'prior-batch-corrections.json').write_text(json.dumps(corrections,ensure_ascii=False,indent=2))
(OUT/'correction-proposed-units.json').write_text(json.dumps(new,ensure_ascii=False,indent=2))
(OUT/'unit-evidence-extensions.json').write_text(json.dumps(extensions,ensure_ascii=False,indent=2))
(OUT/'correction-product-source-evidence.json').write_text(json.dumps(exist_sources+reg_sources,ensure_ascii=False,indent=2))
print('corrections',len(corrections),'new_units',len(new),'extensions',len(extensions))
