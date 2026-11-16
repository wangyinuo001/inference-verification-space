"""Append narrow cross-framework proposals after source/body review."""
import ast,csv,hashlib,json,subprocess
from pathlib import Path
import build_runtime_review as review
H=Path(__file__).resolve().parent
V=review.ROOT
G=V.parent/'sglang-v0.5.14-complete'
SG_SHA='49e384ce9d304648e9959666ecb8ce8cd98d0deb'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==SG_SHA
path='tests/v1/attention/test_sparse_mla_backends.py'

def cite(snapshot,path,line,kind='source'):
    p=(V if snapshot=='vllm' else G)/path
    b=p.read_bytes(); lines=b.decode().splitlines(); assert lines[line-1].strip(),(path,line)
    result=dict(path=path,line=line,kind=kind,source=lines[line-1].strip(),source_sha256=hashlib.sha256(b).hexdigest())
    if snapshot=='sglang':result['snapshot_commit']=SG_SHA
    return result

def unit(id,name,phase,prop,observable,oracle,vrefs,grefs):
    return dict(id=id,name=name,kind='obligation',phase='model_execution',state_scope='local' if id=='compute.attention_scale_binding' else 'shared',
                conditions=['operator_profile','numeric_format','parallel_topology','execution_layout'],
                property=prop,observable=observable,oracle=oracle,
                framework_scope=['vllm','sglang'] if grefs else ['vllm'],
                evidence={'vllm':[cite('vllm',*r) for r in vrefs],'sglang':[cite('sglang',*r) for r in grefs]})
units=[
unit('compute.attention_scale_binding','注意力得分缩放绑定','compute',
     '注意力内核接收到的得分缩放对应当前层或模型的声明缩放；表示变换和权重吸收不得静默改用内部潜在维度的默认缩放。量化附加缩放由具体格式条件规定。',
     ['kernel_score_scale','layer_score_scale','latent_dimension'],
     '拦截实际内核调用的缩放实参，与当前层声明的值及不应采用的内部维度默认值比较。',
     [('vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py',523),('vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py',671),(path,299,'test')],
     [('python/sglang/srt/layers/attention/flashmla_backend.py',355),('python/sglang/srt/layers/attention/flashmla_backend.py',371)]),
unit('execution.stream_dependency','设备流之间的生产消费依赖','compute',
     '异步生产者在读取输入和复用缓冲区前等待所需的先行操作，完成后发布就绪事件，消费者等待该事件后使用结果；临时切换流结束后恢复调用者的流上下文。',
     ['stream_identity','wait_dependencies','ready_event','consumer_visible_values'],
     '核对实际等待与事件发布调用及上下文恢复，并用明确依赖的写入和读取检查可见值。调用记录仅证明被记录的依赖边，不单独证明完整无竞争。',
     [('vllm/v1/attention/backends/mla/index_group.py',327),('vllm/v1/attention/backends/mla/index_group.py',340),('vllm/v1/hisparse/backend.py',114),(path,2225,'test'),(path,2228,'test'),(path,5509,'test')],
     [('python/sglang/srt/managers/cache_controller.py',769),('python/sglang/srt/managers/cache_controller.py',772),('python/sglang/srt/managers/cache_controller.py',789),('python/sglang/srt/managers/cache_controller.py',64),('python/sglang/srt/managers/cache_controller.py',67),('python/sglang/srt/mem_cache/memory_pool.py',158),('python/sglang/srt/mem_cache/memory_pool.py',162)]),
unit('distributed.empty_attention_partition_identity','空注意力分片的归约单位元','compute',
     '上下文分片没有本地有效键时，进入跨 rank 在线 softmax 合并的局部状态为零输出和负无穷 log-sum-exp；非空分片的有效结果保留，空分片不引入 NaN 或非零权重。',
     ['local_valid_count','partial_output','partial_lse'],
     '构造空与非空本地行，并为内核空行设置 NaN 或正无穷哨兵，检查合并前输出和 LSE 的单位元及非空行保持。',
     [('vllm/v1/attention/backends/mla/flashmla_sparse.py',957),('vllm/v1/attention/backends/mla/flashmla_sparse.py',1307),(path,5314,'test'),(path,5315,'test'),(path,5410,'test')],
     [])
]
u=json.loads((H/'units-ready.json').read_text()); existing={x['id'] for x in u['units']}
for x in units:
    if x['id'] not in existing:u['units'].append(x)
    else:u['units']=[x if old['id']==x['id'] else old for old in u['units']]
(H/'units-ready.json').write_text(json.dumps(u,ensure_ascii=False,indent=2)+'\n')
proposals=list(csv.DictReader((H/'proposed-crosswalk.tsv').open(),delimiter='\t'))
oldkeys={(x['path'],x['test_name']) for x in proposals}
source=(V/path).read_text();lines=source.splitlines();tree=ast.parse(source)
records=[]
for name,uid,prop,conditions,oracle in [
('test_nope_flashinfer_sparse_mla_uses_model_scale','compute.attention_scale_binding','NoPE sparse MLA passes model scale 256**-0.5 to bmm1_scale instead of latent rank 512**-0.5','CPU synthetic one-row tensors; auto KV dtype; spy FlashInfer kernel only; numerical output is mocked',[299,300]),
('test_hisparse_generic_stream_orders_ipc_event','execution.stream_dependency','Backend IPC event works with generic copy stream; context restores compute stream and waited copy increment is visible','Test explicitly records/waits input and output events; CUDAHiSparseBackend supplies event; checks event interoperability and context/value, not full runtime automatic scheduling',[2225,2228]),
('test_fp8_mixed_batch_dcp_neutralizes_empty_rows','distributed.empty_attention_partition_identity','FP8 DCP wrapper replaces empty-local-row NaN output/LSE with zero/-inf and preserves populated rows','Three token rows, middle all -1; DCP filter and kernel mocked; check occurs before any collective, so does not assert actual distributed reduce execution',[5314,5315,5317,5318,5319,5320,5321]),
('test_bf16_dcp_filters_topk_and_neutralizes_empty_rows','distributed.empty_attention_partition_identity','BF16 DCP wrapper makes empty-local-row LSE -inf, keeps kernel-provided zero output and preserves populated rows','Mock DCP filter reports valid counts [2,0,2]; fake kernel already supplies zero output/+inf LSE on empty row; wrapper normalizes LSE and contiguous layout, not arbitrary NaN output',[5407,5408,5409,5410,5412,5413,5414])
]:
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    checks=review.direct_checks(fn,lines)
    assert all(any(c['line']==l for c in checks) for l in oracle)
    record=dict(snapshot='vllm',path=path,test_name=name,line=str(fn.lineno),unit_id=uid,property=prop,oracle_source=';'.join(f'{path}:{l}' for l in oracle),conditions=conditions,rule_id='runtime_narrow_property_proposal_20261116')
    if (path,name) not in oldkeys:proposals.append(record)
    records.append({**record,'source_sha256':hashlib.sha256((V/path).read_bytes()).hexdigest(),'checks':[c for c in checks if c['line'] in oracle]})
with (H/'proposed-crosswalk.tsv').open('w',newline='') as f:
    w=csv.DictWriter(f,review.FIELDS,delimiter='\t');w.writeheader();w.writerows(proposals)
pe=json.loads((H/'proposed-evidence.json').read_text())
pe['continuation_20261116']={'reviewed_entries':records,'new_units':units,
 'remaining_boundary':{
   'test_unified_mla_prepares_batch_without_slot_mapping':{
      'finding':'Actual context returns None slot mapping; prepare called exactly once, update not called. This is absent mapping, not an empty tensor mapping and not negative slot filtering.',
      'vllm_evidence':[cite('vllm',path,422,'test'),cite('vllm',path,423,'test'),cite('vllm','vllm/model_executor/layers/attention/mla_attention.py',1495),cite('vllm','vllm/model_executor/layers/attention/mla_attention.py',1496)],
      'sglang_related_but_not_equivalent':[cite('sglang','python/sglang/srt/layers/attention/flashmla_backend.py',316),cite('sglang','python/sglang/srt/layers/attention/flashmla_backend.py',318)],
      'decision':'Keep pending: SGLang save_kv_cache/k-is-None gates are related write admission but do not prove the same absent-slot-mapping contract.'},
   'test_hisparse_expanded_hash_keeps_union_bound':{
      'finding':'Subprocess directly launches hisparse_resolve_residency with four selected host rows but union bound three; checks nonzero process exit and unspecified launch failure. Do not map this device launch assertion to startup configuration admissibility.',
      'evidence':[cite('vllm',path,2876,'test'),cite('vllm',path,2877,'test')],
      'decision':'Keep pending for runtime bounded-residency input contract; failure string alone does not show which internal assertion fired.'}
 }}
(H/'proposed-evidence.json').write_text(json.dumps(pe,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(proposals),'units-ready',len(u['units']))
