"""Re-audit SGLang evidence against the only admitted fixed snapshot."""
import hashlib,json,subprocess
from pathlib import Path
import build_runtime_review as review
H=Path(__file__).resolve().parent
G=review.ROOT.parent/'sglang-v0.5.14-complete'
SHA='49e384ce9d304648e9959666ecb8ce8cd98d0deb'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()==SHA

def evidence(path,lines):
 b=(G/path).read_bytes();src=b.decode().splitlines();sha=hashlib.sha256(b).hexdigest()
 result=[]
 for line in lines:
  assert src[line-1].strip(),(path,line)
  result.append(dict(path=path,line=line,kind='source',source=src[line-1].strip(),source_sha256=sha,snapshot_commit=SHA))
 return result

specs={
'cache.capacity_plan':([
 ('python/sglang/srt/model_executor/pool_configurator.py',[225,226,227,313,314,330,331])],
 'Fixed snapshot computes page-aligned token capacity from available bytes and per-token cell size, and splits hybrid full/SWA capacity. It does not establish identical vLLM layer-group representation or null-block accounting.'),
'resource.shared_host_region':([
 ('python/sglang/srt/utils/host_shared_memory.py',[35,37,40,42,43,53])],
 'Rank0 creates one named shared segment, peers attach after barrier, tensor views share shm.buf, records retain handles. TODO dispose remains: no SGLang cleanup-on-last-consumer guarantee is inferred.'),
'observability.profile_session':([
 ('python/sglang/srt/utils/profile_utils.py',[199,200,201,204,209,298,310,315,318,335])],
 'Stage trigger starts/stops according to configured count/stage; Torch profiler records configured activities and exports rank-named trace. These anchors do not establish exception recovery or vLLM Proton-specific behavior.'),
'resource.host_registration':([
 ('python/sglang/srt/utils/host_shared_memory.py',[45,46,47]),
 ('python/sglang/srt/mem_cache/pool_host/common.py',[49,50,51,52,62,63])],
 'Registration passes actual pointer and byte count; common allocator raises on registration failure and best-effort unregister warns on error. host_shared_memory has no explicit rollback; do not transfer vLLM rollback/fallback oracle to SGLang.'),
'resource.shared_region_population':([],
 'Kept vLLM-only. SGLang mmap allocator has MAP_POPULATE but no matching parallel page-range pre-touch fallback preserving previous bytes; this related allocation mechanism is not counted as the same property.'),
'compute.sparse_index_binding':([
 ('python/sglang/srt/layers/attention/dsa_backend.py',[258,265,271,278]),
 ('python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py',[99,103,106,117,120,138,140,143]),
 ('python/sglang/srt/managers/hisparse_coordinator.py',[815,816,825,826,832,833,843])],
 'Fixed snapshot supplies per-query ragged offsets or batch-selected page tables to top-k conversion; HiSparse initializes output to -1, binds request indices/top-k to swap-in and returns device locations. There is no claim that SGLang has vLLM SparseMLAIndexGroup cross-layer-sharing implementation.'),
'compute.kernel_dispatch_binding':([
 ('python/sglang/srt/layers/quantization/fp8.py',[262,266,267,272,273,782,783]),
 ('python/sglang/srt/layers/attention/hybrid_attn_backend.py',[43,44,48,49,52]),
 ('python/sglang/srt/layers/attention/attention_registry.py',[38,48,61,62])],
 'Linear-layer type/skipped-layer/platform config selects quant method; apply selects Marlin; HybridAttnBackend dispatches decode/prefill/target-verify using explicit mode. Registry rejects TRTLLM MLA for non-MLA. Old 255/776/28/26 were weak definition anchors; these replacements cite actual predicates and returns.'),
'compute.attention_scale_binding':([
 ('python/sglang/srt/layers/attention/flashmla_backend.py',[355,371])],
 'Both fixed-snapshot FlashMLA decode branches pass layer.scaling explicitly; FP8 additionally passes descale_q/descale_k. Only vLLM NoPE test checks against the latent-rank default; no SGLang test coverage is implied.'),
'execution.stream_dependency':([
 ('python/sglang/srt/managers/cache_controller.py',[769,771,772,774,789,64,67]),
 ('python/sglang/srt/mem_cache/memory_pool.py',[157,158,160,162])],
 'Admitted snapshot starts load stream after producer start_event, records per-layer completion and consumer waits on it. Split KV-store capture path waits before/after temporary alt stream. This replaces inadmissible main page_interleave_pool references; no IPC-event interchangeability claim is made for SGLang.'),
'distributed.empty_attention_partition_identity':([],
 'Keep vLLM-only for the currently worded cross-rank DCP identity. Fixed SGLang snapshot has the same (0,-inf) fixup for zero-KV PREFIX CHUNKS merged by merge_state, not the cited newer DCP/XPU/CuteDSL behavior. A future generalized chunk-or-rank partial-attention identity could reuse the related evidence, but this audit does not silently broaden the unit.')
}
unitsdoc=json.loads((H/'units-ready.json').read_text())
replacements=[]
for uid,(groups,note) in specs.items():
 refs=[r for path,lines in groups for r in evidence(path,lines)]
 replacements.append(dict(id=uid,framework_scope=['vllm','sglang'] if refs else ['vllm'],evidence={'sglang':refs},scope_note=note))
 for unit in unitsdoc['units']:
  if unit['id']==uid:
   unit['evidence']['sglang']=refs;unit['framework_scope']=['vllm','sglang'] if refs else ['vllm'];unit['snapshot_evidence_note']=note
(H/'units-ready.json').write_text(json.dumps(unitsdoc,ensure_ascii=False,indent=2)+'\n')
(H/'seed-evidence-replacements.json').write_text(json.dumps(dict(snapshot='sglang-v0.5.14-complete',snapshot_commit=SHA,units=replacements),ensure_ascii=False,indent=2)+'\n')
ns=json.loads((H/'needs-space.json').read_text());byid={x['id']:x for x in replacements}
for p in ns['proposals']:
 uid='cache.capacity_plan' if p['candidate_id']=='cache.capacity_and_layout_plan' else p['candidate_id']
 if uid in byid:
  p['sglang_evidence']=[f'{x["path"]}:{x["line"]}' for x in byid[uid]['evidence']['sglang']]
  p['sglang_source_evidence']=byid[uid]['evidence']['sglang'];p['sglang_snapshot_commit']=SHA
  p['snapshot_scope_note']=byid[uid]['scope_note']
(H/'needs-space.json').write_text(json.dumps(ns,ensure_ascii=False,indent=2)+'\n')
pe=json.loads((H/'proposed-evidence.json').read_text())
pe['continuation_20261116']['new_units']=[x for x in unitsdoc['units'] if x['id'] in ['compute.attention_scale_binding','execution.stream_dependency','distributed.empty_attention_partition_identity']]
related=evidence('python/sglang/srt/layers/attention/trtllm_mla_backend.py',[1014,1023,1030,1032])+evidence('python/sglang/jit_kernel/fixup_zero_kv.py',[43,44])+evidence('python/sglang/jit_kernel/csrc/attention/fixup_zero_kv.cuh',[65,77,78])
pe['sglang_snapshot_audit']=dict(snapshot_commit=SHA,source_root=str(G),replacements=replacements,
 rejected_prior_source_root='sglang-main-20261105',
 correction='The prior new-unit SGLang references were taken from another snapshot and are invalid for this corpus. Older unit anchors were re-read and replaced with predicate/call lines; no test mapping count is increased by this correction.',
 related_not_equivalent_empty_prefix_chunk_evidence=related,
 related_not_equivalent_note='TRTLLM prefix chunks, not cross-rank DCP. Do not attach to the current distributed.empty_attention_partition_identity without an explicit generic-property review.')
(H/'proposed-evidence.json').write_text(json.dumps(pe,ensure_ascii=False,indent=2)+'\n')
audit=dict(snapshot_commit=SHA,unit_count=len(replacements),source_references=sum(len(x['evidence']['sglang']) for x in replacements),source_files=len({r['path'] for x in replacements for r in x['evidence']['sglang']}),vllm_only_units=[x['id'] for x in replacements if x['framework_scope']==['vllm']],replacements_file='seed-evidence-replacements.json',related_empty_prefix_chunk_evidence=related,checks=['git HEAD exactly matches admitted SHA','Every selected anchor is nonempty','All selected source files rehashed after body review','Property notes distinguish partial evidence and framework differences','No newer-main references remain in active unit evidence'])
(H/'snapshot-correction-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(audit,ensure_ascii=False,indent=2))
