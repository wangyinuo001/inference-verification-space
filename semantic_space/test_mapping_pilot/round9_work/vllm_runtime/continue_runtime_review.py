"""Explicit source-reviewed continuation. Does not execute vLLM or infer properties."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import build_runtime_review as review

HERE=Path(__file__).resolve().parent
ROOT=review.ROOT
ledger=json.loads((HERE/'reviewed-decisions.json').read_text())
prior={(d['path'],d['test_name']) for d in ledger['decisions']}
allrows=list(csv.DictReader(review.ALL.open(),delimiter='\t'))
bykey={(r['path'],r['test_name']):r for r in allrows if r['snapshot']=='vllm'}
added=[]

def add(path,name,unit,prop,conditions,helper_names=(),oracle_lines=None):
    matches=[k for k in bykey if k[0]==path and k[1].split('::')[-1]==name]
    assert len(matches)==1,(path,name)
    key=matches[0]
    if key in prior:return
    row=bykey[key]; data=(ROOT/path).read_bytes(); src=data.decode(); lines=src.splitlines(); tree=ast.parse(src)
    funcs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
    fn=next(n for n in funcs if n.lineno==int(row['definition_line']))
    checks=review.direct_checks(fn,lines)
    decision=dict(path=path,test_name=key[1],unit_id=unit,property=prop,conditions=conditions,
                  oracle_lines=oracle_lines if oracle_lines is not None else [c['line'] for c in checks],
                  rule_id='runtime_source_review_continuation_20261116',source_sha256=hashlib.sha256(data).hexdigest())
    decision['input_and_call_source']=[{'line':n.lineno,'source':lines[n.lineno-1].strip(),'expression':ast.unparse(n)} for n in ast.walk(fn) if isinstance(n,ast.Call) and not ast.unparse(n.func).startswith(('pytest.','monkeypatch.','patch','torch.empty','torch.zeros','torch.ones'))]
    if helper_names:
        decision['helper_oracles']=[]
        # Explicitly reviewed bindings: helper is either called directly or passed
        # as the worker argument to distributed_run/multi_process_parallel.
        for helper_name in helper_names:
            hf=next(n for n in funcs if n.name==helper_name)
            call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and (ast.unparse(n.func)==helper_name or ast.unparse(n.func) in ('distributed_run','multi_process_parallel')))
            hc=review.direct_checks(hf,lines)
            chosen=[c['line'] for c in hc]
            if helper_name=='test_sparse_backend_decode_correctness':chosen=[1039,1040,1041,1046,1050]
            decision['helper_oracles'].append(dict(path=path,function=helper_name,oracle_lines=chosen,call_line=call.lineno,call_expression=ast.unparse(call),binding='Explicit worker argument or direct helper call; inputs and reference assertions read. Failure propagation is supported only by the attached transport source or reviewed same-file call graph. Parametrized worker alternatives are recorded separately.'))
    assert decision['oracle_lines'] or decision.get('helper_oracles')
    ledger['decisions'].append(decision);added.append(key)

D='compute.kernel_dispatch_binding'
E='deployment.effective_configuration'
A='deployment.configuration_admissibility'
O='compute.operator_reference_relation'
S='compute.sparse_index_binding'
R='distributed.reduce_relation'
M='distributed.message_delivery'
p='tests/v1/attention/test_mla_prefill_selector.py'
props={
 'test_priorities_no_capability':'Unknown device capability tries only FlashAttention',
 'test_priorities_blackwell_glm_dimensions':'Blackwell GLM dimensions try TRTLLM, FlashAttention, FlashInfer then TokenSpeed in that order',
 'test_priorities_blackwell_deepseek_dimensions':'Blackwell DeepSeek dimensions try FlashAttention before TRTLLM, FlashInfer and TokenSpeed',
 'test_priorities_hopper':'Hopper prefill hook tries only FlashAttention',
 'test_skips_import_error_backends':'Platform hook skips backend ImportError and returns the first valid imported backend',
 'test_all_invalid_raises_error':'Platform hook raises No valid MLA when every candidate reports invalid configuration',
 'test_cpu_platform_returns_fixed_backend':'CPU hook returns available Zen backend or CPU fallback',
 'test_cpu_uses_sdpa_prefill':'Public selector preserves the CPU backend returned by mocked platform hook',
 'test_zen_cpu_prefers_zentorch_prefill':'CPU platform selects Zen prefill when Zen availability reports true',
 'test_zen_cpu_falls_back_when_zentorch_unavailable':'CPU platform returns CPU prefill when Zen availability reports false',
 'test_no_device_capability_returns_flash_attn':'Public selector returns FlashAttention provided by platform hook; capability fallback itself is mocked',
 'test_explicit_flash_attn_selection':'Explicit FlashAttention with successful validation returns FlashAttention backend',
 'test_explicit_backend_invalid_raises_error':'Explicit unsupported FlashInfer selection raises configuration error',
 'test_explicit_backend_import_error_raises':'Explicit TRTLLM import failure becomes invalid-backend ValueError',
 'test_auto_selection_on_hopper':'Public selector preserves mocked FlashAttention choice across three head-dimension pairs',
 'test_blackwell_glm_dimensions_use_trtllm':'Automatic selector preserves mocked platform TRTLLM choice for GLM dimensions',
 'test_platform_hook_raises_propagates':'Automatic selector propagates platform No valid MLA ValueError',
 'test_backend_supported_dimension_validation':'FlashInfer accepts (128,64,128), rejects (64,64,128), and TRTLLM accepts GLM (192,64,256)',
 'test_flash_attn_accepts_glm53_flash_nope_dimensions':'FlashAttention validation accepts (256,0,256) and rejects (256,64,256)',
 'test_rocm_priorities_prefer_aiter_fa':'ROCm hook attempts AITER FA before FlashAttention when candidates fail validation',
 'test_supported_dtypes_are_fp16_bf16_only':'AITER FA dtype gate accepts fp16/bf16 and rejects e4m3 fp8',
 'test_supports_compute_capability_on_rocm':'AITER FA capability gate requires ROCm and MI3xx predicate',
 'test_is_available_delegates_to_rocm_aiter_ops':'AITER FA availability follows mocked AITER enabled flag',
 'test_auto_select_prefers_aiter_fa_on_rocm':'Automatic selector preserves AITER FA choice supplied by mocked platform hook',
 'test_auto_select_falls_back_to_flash_attn_when_aiter_invalid':'Automatic selector preserves FlashAttention fallback supplied by mocked platform hook; fallback decision is not exercised',
 'test_supports_compute_capability_requires_gfx950':'FlyDSL fp8 capability gate rejects non-ROCm and follows gfx950 predicate on ROCm',
 'test_is_available_requires_flydsl_fp8_attention':'FlyDSL fp8 availability requires enabled AITER and both imported function and capability-check symbols',
 'test_validate_configuration_rejects_unsupported_head_dims':'Available FlyDSL fp8 backend rejects v_head_dim=100 configuration',
}
for n,prop in props.items():add(p,n,D,prop,'Read _make_mock_model_config/_make_vllm_config and platform-hook recorder; synthetic MLA dimensions and mocked backend availability/capability; assertions establish selection/validation only, not kernel numerics')
for n,unit,prop in [
 ('test_valid_string_parses_to_enum',E,'FLASH_ATTN string normalizes to MLAPrefillBackendEnum.FLASH_ATTN'),
 ('test_invalid_string_raises_error',A,'Unknown MLA prefill backend string raises ValueError'),
 ('test_default_backend_is_none',E,'AttentionConfig default prefill backend is None'),
 ('test_explicit_flash_attn_backend',E,'AttentionConfig retains explicitly provided FLASH_ATTN enum'),
 ('test_explicit_trtllm_ragged_backend',E,'AttentionConfig retains explicitly provided TRTLLM_RAGGED enum')]:add(p,n,unit,prop,'AttentionConfig constructed directly; asserted parsed/default field or expected ValueError')

p='tests/v1/attention/test_rocm_attention_backends_selection.py'
props={
 'test_deepseek_rocm_backend_registry_paths_are_model_specific':'DSV4 and DSV41 registry enums resolve to their respective AMD backend module paths',
 'test_deepseek_v41_rocm_accepts_legacy_backend_alias':'Legacy and canonical DSV41 backend names select DSV41 attention class; only legacy alias warns',
 'test_aiter_unified_attention_uses_dedicated_metadata_builder':'ROCm and AITER unified backends return their respective metadata builder classes',
 'test_rocm_backends_do_not_support_mm_prefix':'Both ROCm attention backend support predicates reject multimodal prefix attention',
 'test_segmented_attention_uses_dedicated_backend_components':'Segmented backend exposes its dedicated builder/implementation and declared feature gates',
 'test_segmented_attention_admits_unified_fallback_features':'Segmented attention constructor accepts sliding window, soft-cap or sink in mocked RDNA environment',
 'test_segmented_attention_requires_rocm_rdna':'Segmented support gate and constructor reject non-ROCm or non-RDNA and accept ROCm RDNA',
 'test_segmented_attention_is_opt_in':'Segmented backend is absent from automatic priorities but explicit selection resolves on supported RDNA',
 'test_segmented_attention_rejects_mm_prefix_at_selection':'Explicit segmented backend rejects multimodal full-prefix attention',
 'test_segmented_attention_forward_uses_dedicated_dispatch':'Segmented forward invokes dedicated dispatcher once and returns provided output buffer',
 'test_standard_attention_backend_selection':'ROCm standard attention selection matches parameterized explicit/environment priorities and rejects DCP',
 'test_rdna_prefers_triton_attn':'gfx1201/gfx1100 default picks Triton except fp8_e5m2; explicit backend and other RDNA devices retain their prescribed choice',
 'test_mla_backend_selection':'Dense MLA backend selection follows explicit/AITER choice and block-size support with or without DCP',
 'test_sparse_mla_backend_rejects_dcp':'Sparse ROCm MLA selects AITER without DCP and rejects DCP',
 'test_specialized_attention_backends_reject_dcp':'DiffKV and TurboQuant supported cache configurations select without DCP and reject DCP',
 'test_aiter_fa_requires_mi3xx':'Explicit AITER FA is rejected on mocked CDNA1',
 'test_turboquant_boundary_selection_is_not_cached_from_ordinary_run':'Sequential ordinary/TurboQuant runs resolve quantized backend and compatible boundary layout without stale ordinary selection',
 'test_turboquant_run_does_not_mask_unsupported_selected_backend':'TurboQuant run retains compute-capability rejection for explicit AITER FA',
 'test_turboquant_layout_check_respects_backend_override':'Layout compatibility lookup uses overridden TurboQuant backend class',
 'test_turboquant_boundary_layers_share_a_layout':'Native boundary and quantized layer backends have a nonempty common KV layout under parameterized explicit selection',
 'test_sparse_not_supported':'Sparse selection without MLA rejects when no valid backend exists',
 'test_unified_attn_declares_kv_connector_support':'AITER unified backend declares connector support while ROCm attention does not',
 'test_unified_attn_supports_kv_connector':'Explicit AITER unified selection accepts KV connector configuration',
 'test_rocm_attn_rejects_kv_connector':'Explicit ROCm attention selection rejects KV connector configuration',
 'test_auto_selection_for_kv_connector':'Connector automatic selection returns AITER unified when available or Triton fallback',
 'test_unified_attn_prefers_block_contiguous_layout':'Unified backend preferred layout is block contiguous while ROCm attention preferred layout is not',
 'test_unified_attn_drops_lhbnc_with_kv_connector':'Connector-active unified backend filters noncompact layouts while ordinary support includes LHBNC',
}
for n,prop in props.items():add(p,n,D,prop,'ROCm-only test module; explicit mocked platform/environment/topology fixtures and AttentionSelectorConfig; no runtime attention numerical claim')
add(p,'test_segmented_triton_cache_survives_standalone_compilation',E,'Backend initialization and later standalone compilation retain explicit or default Triton cache directory across target/draft/restart','Temporary VLLM_CACHE_ROOT; explicit_cache toggles; environment value and directory existence checked each iteration')

p='tests/distributed/test_pynccl.py'
workers={
 'test_pynccl':('worker_fn',R,'All-reduce of ones equals participating world size'),
 'test_pynccl_multiple_allreduce':('multiple_allreduce_worker_fn',R,'Independent two-rank groups sum ones once or twice to two or four'),
 'test_pynccl_multiple_allreduce_with_vllm':('multiple_allreduce_with_vllm_worker_fn',R,'vLLM TP groups under graph_capture context independently reduce ones to two or four'),
 'test_pynccl_all_gather':('all_gather_worker_fn',M,'All-gather concatenates per-rank arange payloads in rank order'),
 'test_cuda_communicator_all_gather_dim_not_zero':('cuda_communicator_all_gather_dim_worker_fn',M,'CUDA communicator gathers 2x3x4 payloads in rank order along dimensions 1 and -1'),
 'test_pynccl_all_to_all_single':('all_to_all_single_worker_fn',M,'Uneven all-to-all including empty splits delivers source/destination-coded chunks and graph replay reads updated inputs'),
 'test_pynccl_all_gatherv':('all_gatherv_worker_fn',M,'Variable-sized all-gather concatenates each rank payload at prescribed size offsets'),
 'test_pynccl_reduce_scatter':('reduce_scatter_worker_fn',R,'Reduce-scatter returns this rank contiguous slice of summed rank inputs'),
 'test_pynccl_reduce_scatterv':('reduce_scatterv_worker_fn',R,'Variable reduce-scatter returns summed rank inputs at cumulative variable-size offsets'),
 'test_pynccl_with_cudagraph':('worker_fn_with_cudagraph',R,'Captured all-reduce replay of ones equals communicator world size'),
 'test_pynccl_send_recv':('send_recv_worker_fn',M,'Rank zero ones are received unchanged by peer'),
 'test_pynccl_multiple_send_recv':('multiple_send_recv_worker_fn',M,'Independent rank pairs preserve their distinct one/two payloads'),
 'test_pynccl_broadcast':('broadcast_worker_fn',M,'Broadcast from each root delivers that root rank value to all participants'),
}
for n,(helper,unit,prop) in workers.items():add(p,n,unit,prop,'GPU worker wrapper sets rank environment and initializes distributed group; distributed_run spawns 2 or 4 ranks, joins all and asserts zero exit codes; numerical worker assertions determine property, no execution performed in this audit',[helper])

p='tests/distributed/test_custom_all_reduce.py'
props={
 'test_custom_allreduce_filters_dtype':'Custom AR eligibility accepts fp32/fp16/bf16 and rejects int8/fp8',
 'test_cross_node_custom_allreduce_declines_without_launching_ipc':'Cross-node-only communicator declines custom AR without launching IPC in capture and eager modes',
 'test_cross_node_mnnvl_all_gather_and_reduce_scatter_remain_eligible':'Cross-node MNNVL with multicast pointer remains eligible for gather and reduce-scatter',
 'test_custom_allreduce_size_gate_ignored_under_batch_invariance':'Oversized custom AR remains eligible exactly when batch invariance is enabled',
 'test_custom_reduce_scatter_disabled_under_batch_invariance':'Custom reduce-scatter in-range input is disabled under batch invariance',
 'test_cross_node_mnnvl_gate_checks_generation_and_multicast':'MNNVL group gate requires SM100 capability on chosen device and local multicast',
 'test_cross_node_mnnvl_gate_requires_support_on_every_rank':'MNNVL group attempt is rejected when reduced peer support is false',
 'test_local_multicast_support_rejects_non_cuda':'Local multicast predicate rejects non-CUDA platform',
 'test_mnnvl_multimem_reduce_scatter_platform_gate':'Multimem reduce-scatter gate follows explicit SM10x, supported world size and multicast matrix',
 'test_mnnvl_reduce_scatter_backend_gate':'Reduce-scatter selects Lamport/multimem/none according to message bytes, initialization, pointers and batch invariance',
}
for n,prop in props.items():add(p,n,D,prop,'CustomAllreduce constructed without process groups; mocked CUDA/platform/peer predicates and explicit dtype/byte/topology matrix; eligibility assertions only')
for n,prop in [
 ('test_mnnvl_multimem_reduce_scatter_skips_rendezvous_after_peer_alloc_failure','Peer allocation failure prevents rendezvous and leaves initialized but unavailable multimem state for NCCL fallback'),
 ('test_mnnvl_multimem_reduce_scatter_warns_on_rendezvous_failure','Rendezvous failure is reduced across peer status and leaves no usable multimem buffer or pointer')]:add(p,n,'distributed.resource_fallback_agreement',prop,'Fake symmetric-memory allocator and collective status; recorded event order and final no-buffer/no-pointer state; no actual cross-rank run')
for n,helpers,unit,prop in [
 ('test_all_reduce_mhc_matches_unfused_path',['_all_reduce_mhc'],O,'Fused all-reduce/mHC output matches unfused BF16 reference exactly and RMSNorm within one ULP; finalize and LL/BT protocol paths preserve declared equalities'),
 ('test_finalize_all_reduce_gemma_norm_mxfp8_matches_unfused_path',['_finalize_all_reduce_gemma_norm_mxfp8'],O,'Fused finalize/reduce/Gemma RMSNorm/MXFP8 matches explicit residual and quantization reference thresholds including zero padded scales'),
 ('test_custom_allreduce_chunked',['chunked_allreduce'],R,'Chunked all-reduce above max size matches fixed-order FP32-summed BF16 reference bitwise for contiguous/transposed and graph modes'),
 ('test_custom_allreduce',['eager_allreduce','graph_allreduce'],R,'Parameterized eager and graph all-reduce match repeated-sum reference or NCCL across independent TP/PP groups'),
 ('test_one_and_two_stage_allreduce_bitwise_identical',['one_and_two_stage_allreduce'],R,'One-stage and two-stage reduce integer inputs exactly and preserve identical bits on order-sensitive rank contributions')]:add(p,n,unit,prop,'Explicit multi_process_parallel worker binding; helpers initialize rank groups; tests/utils.py ray.get propagates worker assertion failures; GPU/hardware skip gates retained; numerical tolerances and graph input mutations read',helpers)

p='tests/v1/attention/test_sparse_mla_backends.py'
for n,prop,conditions in [
 ('test_rocm_long_bf16_prefill_avoids_nonpersistent_aiter','Long ROCm BF16 sparse MQA output agrees with sparse SDPA reference while forbidden AITER callback raises on use','ROCm gfx942 routing mocked while physical CDNA retained; three batch/TP cases; BF16 reference tolerance 0.01; shared helper forces sparse MQA, simulated head partition only'),
 ('test_sparse_mla_packed_stride','Sparse MLA output for padded physical block stride matches sparse SDPA reference','mixed_small batch; explicit backend/cache/block-size/physical-stride matrix; FP8 rtol 0.065 atol 0.05 and other dtype rtol/atol 0.01; helper verifies rewritten query row')]:add(p,n,O,prop,conditions,['test_sparse_backend_decode_correctness'])
add(p,'test_hisparse_gather_zeroes_unaligned_destination',O,'Out-of-range host row gather zeros the selected unaligned destination and preserves adjacent sentinel byte','uint8 host one row, strided destination with byte offset 17; hisparse_gather_plan; CUDA synchronize before exact output/sentinel assertions')
add(p,'test_hisparse_newest_write_and_recycled_slot_invalidation','cache.safe_reuse','Rewritten host row invalidation prevents reuse of stale hot copy and next swap_in returns updated host content','HiSparse host/hot synthetic cache; row8 incremented by 1000 then invalidated; map solely stale/new row checks, not manually copied host newest-write assertion',oracle_lines=[3681,3682])
add(p,'test_hisparse_invalidation_covers_every_index_group','cache.state_invariants','Slot invalidation replaces matching row ownership with -1 in each listed group while preserving other row/group entries','Three synthetic ownership tables, only first two listed; state index maps request0 to row1; invalid request -1 ignored; expected table equality')
add(p,'test_hisparse_fp8_decode_resolves_rows_once_then_runs_batched_attention',S,'All speculative decode rows pass request IDs and top-k to one swap_in and preserve decode/query grouping into batched kernel','Two decode requests each three verification rows; mock swap_in returns indices+10; asserts input identity/value and kernel/output shapes only')
add(p,'test_sparse_mqa_hisparse_lanes',D,'HiSparse MQA selects hot single-call path for resident or decode-only batches and ordered decode/staged-prefill split otherwise','Five explicit decode/residency/split-query cases; mocked converter/kernel records actual query rows, cache identity by block count and index-source markers; sentinel outputs validate assembly')
add(p,'test_fa4_sparse_hook_routes_lanes',D,'FA4 sparse hook routes all-decode batch to FA4 and other batch to TRTLLM with expected LSE request/layout','Mocked FA4/TRTLLM kernels; two DCP ranks; seven tokens; asserts sole chosen call and sentinel output; LSE conversion is separate numeric assertion, no real kernel executed')
add(p,'test_hisparse_fp8_prefill_gather_uses_dedicated_stream',S,'FP8 prefill gather binds resident/host cache, destination, staging block table and host/device row IDs to same staging plan','Mock gather records arguments; mapping cites argument checks only; wait_stream/record calls are separate event-order proposal',oracle_lines=[5514,5515,5516,5517,5518,5519,5520,5521])

for decision in ledger['decisions']:
    if not decision.get('helper_oracles'):continue
    refs=[]
    for helper in decision['helper_oracles']:
        helper['binding']=helper['binding'].replace('Explicit worker argument or direct helper call; worker inputs and reference assertions read. distributed_run joins and asserts each exitcode at test_pynccl.py:47; multi_process_parallel invokes target.remote and ray.get at tests/utils.py:1468,1476. Parametrized test_target alternatives are listed separately.', 'Explicit worker argument or direct helper call; inputs and reference assertions read. Failure propagation is supported only by the attached transport source or reviewed same-file call graph. Parametrized worker alternatives are recorded separately.')
        if helper['call_expression'].startswith('distributed_run('):
            refs.extend([('tests/distributed/test_pynccl.py',38),('tests/distributed/test_pynccl.py',44),('tests/distributed/test_pynccl.py',47)])
        elif helper['call_expression'].startswith('multi_process_parallel('):
            refs.extend([('tests/utils.py',1468),('tests/utils.py',1476)])
    refs.extend((v['path'],v['line']) for v in decision.get('helper_transport_evidence',[]))
    decision['helper_transport_evidence']=[]
    for refpath,line in sorted(set(refs)):
        b=(ROOT/refpath).read_bytes(); text=b.decode().splitlines()[line-1].strip()
        assert text,(refpath,line)
        decision['helper_transport_evidence'].append(dict(path=refpath,line=line,source=text,source_sha256=hashlib.sha256(b).hexdigest()))
(HERE/'reviewed-decisions.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n')
print('added',len(added))
print(json.dumps(added,ensure_ascii=False))
