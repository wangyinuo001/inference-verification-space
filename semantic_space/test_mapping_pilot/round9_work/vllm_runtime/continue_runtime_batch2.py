"""Second explicit runtime cohort; source-only review, no imports of vLLM."""
import ast,json,hashlib
import continue_runtime_review as r
H=r.HERE
R=r.R;M=r.M;O=r.O;D=r.D;E=r.E;A=r.A
L='shared_resource.lifetime'
base_count=len(r.ledger['decisions'])

def add(path,name,unit,prop,conditions,helpers=()):
 r.add(path,name,unit,prop,conditions,helpers)

def indirect(path,name,unit,prop,conditions,entry,oracle_helpers,edges):
 key=next(k for k in r.bykey if k[0]==path and k[1].split('::')[-1]==name)
 if key in r.prior:return
 add(path,name,unit,prop,conditions,[entry])
 d=r.ledger['decisions'][-1];b=(r.ROOT/path).read_bytes();src=b.decode();lines=src.splitlines();tree=ast.parse(src)
 funcs=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
 template=d['helper_oracles'][0]
 for helper in oracle_helpers:
  fn=next(n for n in funcs if n.name==helper);checks=r.review.direct_checks(fn,lines)
  assert checks,helper
  d['helper_oracles'].append({**template,'function':helper,'oracle_lines':[c['line'] for c in checks],
     'binding':'Explicit same-file call graph attached below links entry/worker/reference oracle; wrapper exit-code/status checks propagate worker failures. Property applies to recorded body assertions, not exit-code alone.'})
 d['helper_call_graph']=[]
 for caller_name,callee in edges:
  caller=next(n for n in funcs if n.name==caller_name)
  call=next(n for n in ast.walk(caller) if isinstance(n,ast.Call) and any(isinstance(v,ast.Name) and v.id==callee for v in ast.walk(n)))
  d['helper_call_graph'].append(dict(path=path,caller=caller_name,caller_definition_line=caller.lineno,callee=callee,call_line=call.lineno,call_expression=ast.unparse(call),source=lines[call.lineno-1].strip(),source_sha256=hashlib.sha256(b).hexdigest()))

p='tests/distributed/test_quick_all_reduce.py'
props={
 'test_flydsl_int4_dispatch_falls_back_to_hip':'BF16 contiguous aligned INT4 inputs use FlyDSL; fp16, strided, misaligned or below-quantization-threshold inputs dispatch HIP',
 'test_should_quick_allreduce_uses_builtin_min_size_when_unset':'Absent minimum override rejects half-threshold input and accepts builtin-threshold fp16 input',
 'test_should_quick_allreduce_uses_min_size_override':'Zero minimum override admits 8-element fp16 input below builtin threshold',
 'test_quick_allreduce_quantization_min_size_unset_uses_configured_codec':'Unset quantization threshold retains configured INT4 codec',
 'test_quick_allreduce_quantization_min_size_uses_fp_below_threshold':'Input bytes below configured quantization threshold select FP codec',
 'test_quick_allreduce_quantization_min_size_uses_configured_codec_at_threshold':'Input bytes equal to quantization threshold select configured INT4 codec',
 'test_quick_allreduce_quantization_min_size_does_not_change_eligibility':'Quantization threshold does not alter builtin QuickReduce input-size admission',
 'test_quick_allreduce_passes_dynamic_quant_level':'QuickReduce forwards dynamically selected FP codec to native qr_all_reduce for below-threshold input',
 'test_rocm_aiter_fused_rmsnorm_uses_aiter_qr_rmsnorm_for_prefill':'81-token prefill selects fused QR+RMSNorm with original input/residual/weight and configured INT4/cast flags',
 'test_rocm_aiter_fused_rmsnorm_keeps_1stage_decode_path':'Four-token decode preserves one-stage fused AR+RMSNorm dispatch and returns its sentinel outputs',
}
for n,prop in props.items():add(p,n,D,prop,'Synthetic QuickAllReduce state or mocked AITER/native calls; fp16/bf16 shape, byte alignment, quantization and size conditions read from each body; no numerical all-reduce inference from fake outputs')
add(p,'test_close_closes_flydsl_int4',L,'Closing QuickAllReduce closes FlyDSL handle and clears retained handle reference','Fake FlyDSL resource; HIP teardown disabled; assert close side effect and reference None')
for n,unit,prop in [
 ('test_quick_allreduce_min_size_env_unset',E,'Unset minimum-MB environment value resolves to None'),
 ('test_quick_allreduce_min_size_env_converts_mb_to_bytes',E,'Minimum-MB environment value four resolves to 4 MiB'),
 ('test_quick_allreduce_min_size_env_rejects_negative',A,'Negative minimum-MB setting is rejected'),
 ('test_quick_allreduce_min_size_env_allows_equal_to_max',E,'Minimum equal to configured maximum resolves to the maximum byte count'),
 ('test_quick_allreduce_min_size_env_rejects_larger_than_max',A,'Minimum above effective maximum is rejected'),
 ('test_quick_allreduce_quantization_min_size_env_unset',E,'Unset codec-threshold-KB environment value resolves to None'),
 ('test_quick_allreduce_quantization_min_size_env_converts_kb_to_bytes',E,'Quantization threshold 2048 KiB converts to 2048 times KB bytes'),
 ('test_quick_allreduce_quantization_min_size_env_rejects_negative',A,'Negative quantization-threshold-KB setting is rejected')]:add(p,n,unit,prop,'Environment cache disabled around direct configuration getter; explicit unset/string value and max=16 MiB; expected returned field or ValueError')
add(p,'test_custom_quick_allreduce',R,'Eager and captured QuickReduce outputs match repeated-sum or NCCL reference within configured codec tolerance','ROCm TP2, PP1/2; FP/INT8/INT6/INT4/INT3; INT3 atol5.5 rtol0.1, others atol2.5 rtol0.1; eight aligned random sizes and fp16/bf16; gfx942 graph branch xfailed; ray.get propagates worker errors',['graph_quickreduce','eager_quickreduce'])
add(p,'test_custom_quick_allreduce_bf16_cast',R,'BF16 input reduced through FP16-kernel mode agrees with TP sum','ROCm TP2, FP codec, CAST_BF16_TO_FP16=1, atol2.5 rtol0.1; worker verifies cast flag',['bf16_cast_quickreduce'])

p='tests/distributed/test_rocm_quick_reduce.py'
for n,prop in {
 'test_quick_reduce_quantization_env_var':'Explicit quantization environment string is preserved by environment reload',
 'test_quick_reduce_quantization_default':'Absent quantization setting defaults to NONE',
 'test_quick_reduce_cast_bf16_to_fp16_env_var':'Boolean cast setting resolves from explicit 1/0 environment string',
 'test_quick_reduce_cast_bf16_to_fp16_default':'Absent cast setting defaults to true',
 'test_quick_reduce_max_size_env_var':'Maximum size environment is parsed to supplied integer MB or None when absent',
 'test_quick_reduce_max_size_default':'Absent maximum-size setting resolves to None',
}.items():add(p,n,E,prop,'Module environment reloaded after monkeypatch set/delete; assertions inspect effective environment field')
for n,prop in {
 'test_quick_allreduce_rocm_arch_available':'ROCm architecture support accepts gfx942/gfx950 and rejects gfx90a/empty architecture',
 'test_quick_allreduce_rocm_arch_available_handles_probe_failure':'Device-property probing failure reports QuickReduce unavailable',
 'test_quick_allreduce_rejects_disabled':'Disabled communicator rejects input eligibility',
 'test_quick_allreduce_rejects_unsupported_dtype':'QuickReduce rejects fp32 input',
 'test_quick_allreduce_rejects_non_aligned_input':'QuickReduce rejects five-element fp16 input',
 'test_quick_allreduce_rejects_non_contiguous_input':'QuickReduce rejects strided fp16 input',
 'test_quick_allreduce_rejects_input_smaller_than_threshold':'QuickReduce rejects fp16 input just below builtin byte threshold',
 'test_quick_allreduce_accepts_input_at_threshold':'QuickReduce accepts fp16 input at builtin byte threshold',
 'test_quick_allreduce_rejects_input_larger_than_max_size':'QuickReduce rejects input exceeding configured byte maximum',
 'test_quick_allreduce_bf16_uses_fp16_threshold_when_cast_enabled':'Same BF16 input is eligible with FP16 cast threshold and ineligible without cast',
 'test_quick_allreduce_supported_world_sizes':'QuickReduce advertises only world sizes 2,4,8',
 'test_quick_allreduce_supported_dtypes':'QuickReduce advertises fp16 and bf16 dtypes',
}.items():add(p,n,D,prop,'ROCm-only file; _make_quick_allreduce sets state directly, platform/device properties mocked where applicable; support predicates or declared supported set only')
# Numeric constants and positive table cardinalities are not promoted to runtime
# selection claims; the corresponding entries remain pending.
for n,entry,worker,prop,cond in [
 ('test_quick_allreduce_two_gpu_correctness','_run_two_gpu_quick_allreduce_test','_quick_allreduce_worker','Two-rank QuickReduce of ones matches sum within tolerance','FP/INT8/INT6/INT4; fp16; no cast; atol2.5 rtol0.1'),
 ('test_quick_allreduce_bf16_cast_mode','_run_two_gpu_quick_allreduce_test','_quick_allreduce_worker','BF16 cast QuickReduce of ones matches two-rank sum','FP codec; BF16 with cast flag verified; atol2.5 rtol0.1'),
 ('test_quick_allreduce_cudagraph_replay','_run_cudagraph_replay_test','_quick_allreduce_cudagraph_worker','Single-operation QuickReduce graph replay reads each round new inputs and equals rank sum','Two ranks, ten rounds, fp16 FP codec, 2M elements, inputs updated in place; isclose mismatch count zero')
]:indirect(p,n,R,prop,cond+'; spawn wrapper joins with deadline and asserts exitcode zero',entry,[worker],[(n,entry),(entry,worker)])
for n in ['test_quick_allreduce_llm_e2e','test_quick_allreduce_llm_e2e_ray']:
 indirect(p,n,'quality.sample_semantic_expectation','Both baseline and QuickReduce Qwen answers retain required fact words for two provided long-memo prompts','Qwen2.5-0.5B-Instruct TP2 eager half; greedy max20; backend mp or ray; required words oslo/otter and tuesday/apricot are hard assertions. Exact recorded response and baseline/variant differences cause xfail, so not mapped as hard model output equality.',
 'run_quick_reduce_llm_e2e',['_assert_required_words'],[(n,'run_quick_reduce_llm_e2e'),('run_quick_reduce_llm_e2e','_quick_reduce_llm_e2e_worker'),('_quick_reduce_llm_e2e_worker','_run_quick_reduce_llm_e2e_in_subprocess'),('_run_quick_reduce_llm_e2e_in_subprocess','_assert_required_words')])

p='tests/distributed/test_dcp_a2a.py'
for n,u,prop in [
 ('test_default_is_ag_rs',E,'Unspecified DCP communication mode resolves to ag_rs'),
 ('test_a2a_with_dcp_valid',E,'Explicit a2a communication mode retained for TP4/DCP4'),
 ('test_invalid_backend_rejected',A,'Invalid DCP communication backend rejected by ParallelConfig'),
 ('test_ag_rs_with_dcp_1_valid',E,'Explicit ag_rs communication mode retained with DCP1')]:add(p,n,u,prop,'Direct ParallelConfig construction and set_dcp_defaults where specified; no collective execution')
for n,prop in {
 'test_single_rank':'Single partial attention state combines to unchanged output',
 'test_equal_lse':'Equal local LSE values produce arithmetic average of partial outputs',
 'test_dominant_rank':'LSE difference -100 versus zero makes dominant-rank output agree within 1e-5',
 'test_empty_shard_ignores_undefined_output':'LSE-weighted combine ignores NaN output of rank with -inf LSE',
 'test_mathematically_correct':'Two partial outputs combine according to explicit exp-normalized LSE weights',
 'test_return_lse':'Returned global LSE equals log(exp(1)+exp(2)) and expected shapes',
 'test_base2_return_lse':'Base2 mode returns explicit exp2-normalized weighted output and log2-sum-exp2 LSE',
}.items():add(p,n,O,prop,'CPU tensor arithmetic representing partial-rank attention; explicit small reference values; no transport; tolerances read from body')
add(p,'test_ag_rs_masks_empty_shard_and_padded_lse','distributed.empty_attention_partition_identity','AG/RS LSE preparation marks empty-sequence and padded query rows -inf while preserving active rows','Fake two-rank gather and correct_attn_out; seq_lens=[0,2], query_start=[0,1,5], 7 padded rows; assertions only on LSE')
add(p,'test_empty_seq_lens_ignore_undefined_output','distributed.empty_attention_partition_identity','Packed A2A preparation/combination returns zero output and -inf LSE for zero-KV rows and finite active outputs','GPU BF16 5 rows/2 fake packed ranks; first output row NaN, seq_lens=[0,2]; packing and unpack combine without actual interrank collective')
add(p,'test_pack_unpack_combine_matches_reference',O,'Packed attention output/LSE roundtrip and combine match reference state reduction','GPU synthetic 4-rank tensor layout, fp16/bf16/fp32, e/base2 and return_lse modes; output tolerance 0.03 except fp32 1e-5, LSE 1e-4',['_assert_packed_a2a_close'])
for n in ['test_distributed_packed_a2a_matches_reference','test_distributed_packed_a2a_with_workspace_matches_reference']:
 indirect(p,n,R,'Four-rank packed A2A combines local heads to all-gather plus LSE-weighted reference','Four GPUs; fp16/bf16/fp32 or BF16 with workspace manager; output dtype tolerance and LSE1e-4; spawn exitcode propagated','_distributed_run',['_distributed_packed_a2a_worker','_assert_packed_a2a_close'],[(n,'_distributed_packed_a2a_worker'),('_distributed_packed_a2a_worker','_assert_packed_a2a_close')])

p='tests/distributed/test_dcp_direct_a2a_lse_reduce.py'
props={
 'test_env_disabled_returns_none':'Explicit disabled direct-A2A flag returns no workspace',
 'test_forced_with_unsupported_dtype_raises':'Forced direct-A2A rejects fp32 unsupported output dtype',
 'test_auto_with_unsupported_dtype_returns_none':'Automatic direct-A2A falls back on unsupported fp32 dtype',
 'test_q_gather_env_disabled_returns_none':'Q gather disabled flag overrides enabled A2A and returns no Q workspace',
 'test_q_gather_flag_is_independent':'Q gather enabled flag constructs requested fp32 workspace independently of disabled A2A',
 'test_kv_gather_env_disabled_returns_none':'KV gather disabled flag overrides enabled A2A',
 'test_kv_gather_flag_is_independent':'KV gather enabled flag constructs workspace independently of disabled A2A',
 'test_gather_requires_multicast':'Q/KV direct gather declines without group-spanning symmetric-memory multicast',
 'test_mla_dcp_manager_selects_direct_backends':'MLA DCP manager binds direct Q/KV gather and LSE reduce callbacks with sequence metadata',
 'test_mla_dcp_manager_selects_fallback_backends':'Unavailable direct workspaces select torch/group gather and packed A2A combine callbacks',
 'test_mla_dcp_manager_selects_pcp_combine':'PCP ag_rs manager selects cp_lse_ag_out_ar and disables separate query gather',
 'test_sparse_mla_builder_initializes_dcp_manager':'Sparse MLA builder reuses layer DCP manager and initializes KV gather with its chunked workspace',
}
for n,prop in props.items():add(p,n,D,prop,'Synthetic manager/group/config and mocked workspace factories; direct versus fallback function identity/calls verified, not numerical collective correctness')
for n,prop in {
 'test_zero_ubatches_raises':'Direct A2A workspace rejects zero ubatches',
 'test_kv_gather_rejects_invalid_workspace_geometry':'Direct KV workspace rejects zero ubatches, indivisible token capacity and non-16-byte row width',
 'test_q_gather_rejects_invalid_workspace_geometry':'Direct Q workspace rejects zero ubatches, too few padded heads and non-16-byte row width',
}.items():add(p,n,A,prop,'CPU mock process group; constructor validation raises ValueError before real allocation/communication')
for n,prop in {
 'test_dcp_workspace_covers_decode_width':'Derived DCP token capacity accounts for V1 parallel draft width versus V2/diffusion width',
 'test_dcp_chunk_workspace_alignment_covers_interleave':'Chunked context workspace rounds to lcm(block_size,DCP*interleave) boundary',
 'test_sparse_mla_workspace_preserves_non_dcp_size':'DCP1 sparse workspace remains index_topk times max_num_seqs rather than DCP-aligned size',
}.items():add(p,n,E,prop,'Explicit synthetic config fields; expected derived workspace-size integers, no capacity execution claim')
for n,worker,u,prop,helpers,edges in [
 ('test_distributed_direct_q_gather_cuda_graph_replay','_distributed_direct_q_gather_worker',M,'Direct Q gather preserves rank-ordered bytes, dtype/stride/storage and fresh graph-replay inputs across ubatches',['_assert_q_gather_matches_reference'],[('_distributed_direct_q_gather_worker','_assert_q_gather_matches_reference')]),
 ('test_distributed_direct_kv_gather_matches_reference','_distributed_direct_kv_gather_worker',M,'Direct KV gather matches all_gather_into_tensor bytes across bf16/fp16 sizes and ubatches',[],[]),
 ('test_distributed_direct_a2a_matches_reference','_distributed_direct_a2a_worker',R,'Direct symmetric-memory A2A LSE reduction matches gathered weighted reference including empty shards, noncontiguous layouts and graph replay',['_assert_close'],[('_distributed_direct_a2a_worker','_assert_close')]),
]:indirect(p,n,u,prop,'GPU hardware guards; two/four ranks according to decorator; spawn helper kills overdue processes and asserts exitcode; q-gather bitwise, A2A numerical rtol/atol0.03; input layout assertions retained','_distributed_run',[worker]+helpers,[(n,worker)]+edges)

p='tests/distributed/test_comm_ops.py'
for n,unit,prop,cond in [
 ('test_irecv_tensor_dict_send_allgather_postprocess_binds_keys',M,'Deferred receive postprocessing reconstructs both dictionary keys to full independently gathered tensors','Fake irecv fills two int32 slices; waits and per-key closures invoked; both outputs equal ones(4)'),
 ('test_cuda_communicator_checkpoints_flashinfer_workspaces',L,'Communicator checkpoints/restores each unique normal/quantized workspace once even when aliased','Mock normal/quant workspace; alias switch; prepare and group-bound restore call assertions'),
 ('test_cuda_communicator_suspend_detaches_flashinfer_workspaces',L,'Suspend/resume delegates once to each FlashInfer workspace and PyNCCL while tolerating unsupported symmetric-memory checkpoint','Mock workspaces; optional NotImplementedError; lifecycle calls checked only'),
 ('test_cuda_communicator_resume_retries_failed_restore',L,'Failed workspace restore is retryable on the next resume','First mock restore raises RuntimeError then succeeds; count exactly two'),
 ('test_flashinfer_standalone_size_tuning',E,'Derived standalone allreduce maximum size follows backend/GPU/world/node matrix','Explicit ten-case tuning matrix; mocked platform capability/node count'),
 ('test_flashinfer_standalone_workspace_size',E,'Workspace token capacity is derived from byte maximum and hidden dimension only when standalone FlashInfer enabled','Mock128 MiB cap; hidden7168/4096; factory max-token argument equals9362/16384/128'),
 ('test_flashinfer_auto_backend_uses_group_topology',D,'Automatic FlashInfer backend choice follows node count, Blackwell capability and HiSparse activation','Six-case matrix; mocked configuration and topology'),
 ('test_flashinfer_workspace_failure_is_not_retried','distributed.resource_fallback_agreement','Failed MNNVL/TRTLLM workspace attempts are cached until explicit destruction, avoiding repeated collective creation','Mock creation returns None; two calls attempt only once per backend; destroy resets retry state; no multi-rank run'),
 ('test_flashinfer_all_reduce_precedes_nccl',D,'Eligible FlashInfer all-reduce returns its output without invoking NCCL selector','Mock communicator eligible and enabled'),
 ('test_aiter_all_gather_precedes_pynccl',D,'Eligible AITER all-gather returns its output even with PyNCCL disabled','Mock AITER collective and capability gate'),
 ('test_isend_object_posts_size_then_object_and_releases_on_wait',L,'Object-send composite retains both size/payload tensors until first wait and releases them without duplicate waits','Fake Work records waits; two posted tensors; repeated handle.wait is idempotent'),
 ('test_isend_tensor_dict_includes_metadata_handle',L,'Tensor-dictionary send exposes metadata completion handle and releases its retained storage after wait','Fake isend records size/object/tensor order; each returned handle waited'),
 ('test_isend_tensor_dict_self_retains_for_fire_and_forget',L,'Fire-and-forget tensor sends retain pending tensors/handles until completion and reap prior metadata handles','Fake Work completion toggled; metadata-only entry explicitly best-effort reaped; do not infer reliable metadata-only completion signal'),
 ('test_send_tensor_dict_sync_path_does_not_self_retain',L,'Synchronous tensor-dictionary send leaves no pending retention entry','Mock send/isend initialized group; final deque length zero'),
]:add(p,n,unit,prop,cond)
add(p,'test_multi_process_pipeline_parallel',M,'Pipeline tensor and tensor-dictionary send/receive preserve payloads on downstream ranks','PP2, TP1; ray worker target parameterization includes plain float tensor and mixed CPU/GPU dictionary; ray.get propagates assertion errors',['send_recv_test_worker','send_recv_tensor_dict_test_worker'])
# Unsupported checkpoint exception and lazy wait are retained pending rather than
# invented as deployment admissibility or GPU stream dependency.

(H/'reviewed-decisions.json').write_text(json.dumps(r.ledger,ensure_ascii=False,indent=2)+'\n')
summary={'new_entries':len(r.ledger['decisions'])-base_count,'total_decisions':len(r.ledger['decisions']),
 'pending_findings':[
 {'path':'tests/distributed/test_quick_all_reduce.py','test_name':'test_custom_quick_allreduce_variable_input','reason':'Worker asserts numerical values but parent only join/is_alive; no exitcode assertion. Cannot promote worker numerical oracle to pytest entry. The test detects timeout/hang only.'},
 {'path':'tests/distributed/test_rocm_quick_reduce.py','tests':['test_quick_reduce_regime_values','test_quick_reduce_regime_names','test_quick_allreduce_min_size_table','test_qr_max_size'],'reason':'Only codec enum constants/table positivity/size-positive checks; no exact runtime selection or parsed configuration relation established.'},
 {'path':'tests/distributed/test_dcp_a2a.py','tests':['TestLSEWeightedCombine::test_importable','TestLSEWeightedCombine::test_lse_pack_dim'],'reason':'Callable existence and packing-lane count utility checks kept distinct from full reduce/output relation.'},
 {'path':'tests/distributed/test_comm_ops.py','tests':['test_multi_process_tensor_parallel','test_multi_process_tensor_parallel_pipeline_parallel'],'reason':'Worker alternatives mix message delivery and arithmetic reduction; no single-property promotion of the entire parameterized entry.'},
 {'path':'tests/distributed/test_comm_ops.py','tests':['test_cuda_communicator_checkpoint_rejects_symm_mem_workspace','test_async_intermediate_tensors_lazy_wait'],'reason':'Checkpoint operation capability and lazy wait lifecycle need suitable specific property; not startup config or GPU stream dependency.'}
 ],'execution':'Pinned-source review only; no tests/GPU/server execution'}
(H/'batch2-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(summary['new_entries'],summary['total_decisions'])
