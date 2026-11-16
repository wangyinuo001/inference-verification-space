"""Full-body review of four compact routing/sampling files, no runtime claim."""
import ast,csv,hashlib,json
from extract_review import HERE,SOURCE
from review_cpu_gdn_batch6 import exact_entry
REF='compute.operator_reference_relation';EXEC='compute.execution_relation';OUT='compute.output_constraints';POL='sampling.policy'
SH='tests/kernels/test_shuffle_rows.py';GR='tests/kernels/moe/test_grouped_topk.py';RO='tests/kernels/moe/test_rocm_aiter_topk.py';SA='tests/kernels/test_cpu_fused_sampling.py'
D={
SH:{
 'test_shuffle_rows_basic':(REF,'Identity dst2src map reproduces input exactly; shape, dtype and device are also checked.',[]),
 'test_shuffle_rows_permutation':(REF,'Reverse dst2src map equals torch.flip(input,0) at atol=1e-6, rtol=1e-5; shape/dtype/device preserved.',[]),
 'test_shuffle_rows_expansion':(REF,'Random 2x expansion returns each source row named by dst2src at 1e-6/1e-5; output shape/dtype/device checked.',[]),
 'test_shuffle_rows_invalid_indices_are_zeroed':(REF,'For dst2src [-1,1,3] and 3 source rows, invalid rows are zero and row1 matches input row1.',[]),
 'test_shuffle_rows_random_permutation':(REF,'Seed42 random permutation returns each mapped input row at 1e-6/1e-5 with shape/dtype/device preservation.',[]),
 'test_shuffle_rows_edge_cases':(REF,'Single-token and single-feature identity maps reproduce the input exactly.',[]),
 'test_shuffle_rows_moe_like_scenario':(REF,'32x1024 FP16 rows duplicated for topk=2 yield 64 mapped rows at 1e-6/1e-5; shape/dtype/device checked.',[]),
 'test_shuffle_rows_dtype_consistency':(REF,'64x512 identity mapping preserves FP16/BF16/FP32 dtype/device and values at 1e-6/1e-5.',[]),
 'test_shuffle_rows_device_consistency':(OUT,'32x256 FP16 input returns output on the same CUDA device; no numerical assertion.',[]),
 'test_shuffle_rows_contiguous_output':(OUT,'64x512 identity mapping produces a contiguous output tensor; no numerical assertion.',[]),
},
GR:{
 'test_kimi_grouped_topk_masks_padding_when_enabled':(REF,'Nonpadding row exactly matches unmasked baseline; when skip_padding=True padded row has IDs -1 and weights0, otherwise IDs>=0 and weights>0.',[]),
 'test_grouped_topk_padding_changes_on_graph_replay':(REF,'On graph replay, current alternating/all/none padding mask produces exact masked baseline weights0 and IDs-1, across grouped/single-group kernel tiers.',['fused_grouped_topk']),
 'test_grouped_topk_padding_opcheck':(None,'custom_operator_opcheck_only_requires_registration_contract',[]),
 'test_grouped_topk':(EXEC,'Explicit unfused forward_cuda baseline and fused grouped topk agree on IDs exactly and weights at atol=2e-2, rtol0; routing dispatch method also checked.',[]),
 'test_grouped_topk_single_group_large_batch':(REF,'1536x896 BF16 logits select top16 IDs matching stable reference; renormalized sigmoid weights agree at atol2e-5, rtol0.',['_single_group_reference','_run_single_group_topk']),
 'test_grouped_topk_single_group_tiers':(REF,'17-row size/dtype tiers select reference IDs and sigmoid/softmax weights after bias selection, optional renormalization and scaling, atol2e-5 rtol0.',['_single_group_reference','_run_single_group_topk']),
 'test_grouped_topk_single_group_capacity_tiers':(REF,'3-row capacity-boundary cases select reference IDs and renormalized weights scaled2.5, atol2e-5 rtol0.',['_single_group_reference','_run_single_group_topk']),
 'test_grouped_topk_single_group_stable_ties':(REF,'All-zero logits/bias select stable IDs0..15 and uniform weights2.5/16 across 512/896/1024 experts.',['_run_single_group_topk']),
 'test_grouped_topk_single_group_nonfinite_scores':(REF,'All-nonfinite or15finite logits produce explicitly constructed IDs and weights, including zero-filled invalid selections and uniform fallback on renormalization.',['_run_single_group_topk']),
},
RO:{
 'test_rocm_aiter_biased_grouped_topk_custom_op_registration':(None,'custom_operator_registration_only_existence_and_callable',[]),
 'test_rocm_aiter_grouped_topk_custom_op_registration':(None,'custom_operator_registration_only_existence_and_callable',[]),
 'test_rocm_aiter_topk_gating_custom_op_registration':(None,'custom_operator_registration_only_callable',[]),
 'test_rocm_aiter_biased_grouped_topk_torch_compile_compatibility':(EXEC,'Eager and torch.compile(fullgraph,inductor) biased-grouped outputs agree after sorting IDs and reordering weights: weights1e-2/1e-2 and IDs allclose.',[]),
 'test_rocm_aiter_grouped_topk_torch_compile_compatibility':(EXEC,'Eager and torch.compile grouped-softmax outputs agree after sorting IDs and reordering weights: weights1e-2/1e-2 and IDs allclose.',[]),
 'test_rocm_aiter_topk_gating_matches_softmax_reference':(REF,'Selected expert IDs are unique/in-range and at or above softmax top-k cutoff; selected weights match gathered softmax scores with optional renormalization at2e-2/2e-2.',[]),
 'test_rocm_aiter_topk_gating_torch_compile_compatibility':(EXEC,'For4x512 logits/top10, compiled and eager IDs match exactly after sorting and weights match at1e-2/1e-2; fake-tensor opcheck is additional.',['_sort_routing']),
 'test_flat_topk_matches_aiter':(EXEC,'Flat sigmoid+bias topk agrees with AITER on sorted expert-ID sets exactly and raw weight arrays at2e-3/2e-3.',['_assert_matches_aiter','_aiter_reference','_routing_inputs']),
 'test_flat_topk_matches_aiter_with_uneliminating_groups':(EXEC,'When all1/2/4/8 groups survive, flat topk equals AITER sorted IDs and raw weights at2e-3/2e-3 for64x64/top6.',['_assert_matches_aiter','_aiter_reference','_routing_inputs']),
 'test_flat_topk_casts_bias_like_aiter':(EXEC,'BF16 gating with FP32 bias agrees with AITER after reference casts bias to gating dtype; sorted IDs exact, raw weights2e-3/2e-3.',['_assert_matches_aiter','_aiter_reference','_routing_inputs']),
 'test_flat_topk_honours_routed_scaling':(EXEC,'Scaling1/2.446/2.827 in8x64/top6 flat topk agrees with AITER sorted IDs and raw weights2e-3/2e-3.',['_assert_matches_aiter','_aiter_reference','_routing_inputs']),
 'test_biased_grouped_topk_substitutes_only_when_expected':('compute.kernel_dispatch_binding','Counted flat-topk substitution occurs only under declared capture/group-selection cases; weights must be finite, with no reference numerical comparison.',['_routing_inputs']),
},
SA:{
 'TestGreedyArgmax::test_single_dominant':(POL,'Greedy argmax chooses42 from50000 logits when only token42 has dominant100 versus -1e9.',[]),
 'TestFusedGumbelArgmax::test_distribution_chi_squared':(None,'needs_sampling_distribution_relation:100000 seeds,100-way softmax reference, chi-square on expected-count>5 bins, p>0.001',[]),
 'TestFusedGumbelArgmax::test_noise_not_sliding_window':(None,'needs_sampling_randomness_relation:2048 consecutive seeds on128 uniform logits must have one-token-shift winner fraction<0.05',[]),
 'TestFusedGumbelArgmax::test_zipf_head_matches_softmax':(None,'needs_sampling_distribution_relation:8192-way Zipf head frequency over16384 seeds is within8 standard errors of softmax probability',[]),
 'TestFusedGumbelArgmax::test_deterministic_same_seed':(EXEC,'Repeated Gumbel sampling with identical4x32000 logits and seeds[42,123,456,789] returns equal tokens.',[]),
 'TestFusedGumbelArgmax::test_different_seeds_differ':(None,'needs_sampling_randomness_relation:uniform16x50000 logits with seed vectors differing by1000000 produce unequal output vectors; does not prove independence',[]),
 'TestFusedGumbelArgmax::test_output_in_range':(POL,'Each sampled token ID lies in[0,vocab_size) for declared vocabulary and batch sizes.',[]),
 'TestFusedGumbelArgmax::test_samples_only_from_finite_logits':(POL,'With -inf outside IDs[3,17,99,1024,8191], all64 sampled tokens belong to the finite allowed set.',[]),
 'TestFusedGumbelArgmax::test_single_unmasked_token':(POL,'Only token123 unmasked in4096-way logits: all8 seeds including signed-int64 boundaries return123.',[]),
 'TestFusedGumbelArgmax::test_tokens_beyond_gumbel_max_cannot_win':(POL,'For256 rows with logits[0,-80,-80,-80], the bounded-Gumbel implementation returns token0 for every declared seed.',[]),
 'TestFusedGumbelArgmax::test_tied_huge_logits_not_index_biased':(None,'needs_sampling_distribution_relation:4096 seeds with2 logits at FP32 max must select both IDs and have index0 frequency between0.35 and0.65',[]),
 'TestFusedGumbelArgmax::test_serial_batch1_matches_batched':(EXEC,'Eight2048-way rows with matching per-row seeds yield identical token vectors in batched OpenMP and serial batch1 execution.',[]),
 'TestFusedGumbelArgmax::test_empty_batch':(OUT,'Empty[0,7] logits with empty seed vector returns shape(0,) and dtypeint64.',[]),
 'TestFusedGumbelArgmax::test_masked_support_matches_softmax':(None,'needs_sampling_distribution_relation:20000 samples never select masked sixth token and five-bin softmax chi-square p>1e-7',[]),
 'TestFusedGumbelArgmax::test_zipf_tail_matches_softmax':(None,'needs_sampling_distribution_relation:8192-way Zipf upper-half token frequency over16384 seeds within8 standard errors of softmax tail mass',[]),
 'TestMemory::test_fused_no_intermediate_allocs':(None,'needs_resource_observation_contract:positive net Python tracemalloc snapshot allocation over50 calls is less than16*128256*4 bytes; does not measure all native tensor allocations',[]),
}}


# Accepted ontology refinements: these are active mappings, not candidate rules.
for _name in ('test_distribution_chi_squared','test_zipf_head_matches_softmax','test_tied_huge_logits_not_index_biased','test_masked_support_matches_softmax','test_zipf_tail_matches_softmax'):
    _key='TestFusedGumbelArgmax::'+_name
    _old=D[SA][_key]
    D[SA][_key]=('sampling.distribution_relation',_old[1].split(':',1)[1],_old[2])
for _name in ('test_noise_not_sliding_window','test_different_seeds_differ'):
    _key='TestFusedGumbelArgmax::'+_name
    _old=D[SA][_key]
    D[SA][_key]=('sampling.randomness_relation',_old[1].split(':',1)[1],_old[2])

def main():
 packets=[p for p in (json.loads(l) for l in (HERE/'source-packets.jsonl').open()) if p['key'][1] in D]
 assert {(p['key'][1],p['key'][2]) for p in packets}=={(p,n) for p,ds in D.items() for n in ds}
 rows=[];pending=[];entries=[]
 for p in packets:
  snapshot,path,name,line=p['key'];unit,prop,helper_names=D[path][name];e=exact_entry(path,name)
  # The product function is traced below, not mistakenly looked up as a test helper.
  helper_names=[h for h in helper_names if h!='fused_grouped_topk']
  helpers=[exact_entry(path,h) for h in helper_names]
  checks=[{'path':path,**c} for c in e['checks']]+[{'path':path,**c} for h in helpers for c in h['checks']]
  refs=';'.join(f"{c['path']}:{c['line']}" for c in checks)
  cond='; '.join(d['source'].replace('\n',' ') for d in e['decorators'])+'; exact input state, setup, skips and helper sources in evidence-batch8.json entry '+name
  entries.append({'key':p['key'],'source_sha256':p['sha256'],'unit_id':unit,'property':prop,**e,'shared_helper_sources':helpers,'all_checks':checks})
  if unit:rows.append([snapshot,path,name,line,unit,prop,refs,cond,'vllm_routing_sampling_per_entry_review_20261116'])
  else:pending.append([snapshot,path,name,line,prop,refs or f'{path}:{line}'])
 for file,fields,data in [('crosswalk-batch8.tsv',['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],rows),('pending-batch8.tsv',['snapshot','path','test_name','line','reason','evidence_source'],pending)]:
  with (HERE/file).open('w',newline='') as f:w=csv.writer(f,delimiter='\t');w.writerow(fields);w.writerows(data)
 prod={'vllm/_custom_ops.py':[948], 'vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py':[31,92,263], 'vllm/_aiter_ops.py':[539,582,606,3727], 'vllm/model_executor/layers/fused_moe/router/rocm_sigmoid_bias_flat_topk.py':[169], 'csrc/cpu/sampling_kernels.cpp':[39,73,90,144,166]}
 ev={'batch':'routing_sampling','counts':{'scope':len(packets),'mapped':len(rows),'pending':len(pending)},'review_method':'All assigned test bodies and named shared helpers read, preserving exact numerical tolerances, input generation and checked observable. No test execution claimed. Statistics, schema-only opcheck and allocation observations remain distinct.', 'file_reviews':[{'snapshot':'vllm','path':p,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest(),'review':'Complete file read and each assigned test individually classified against concrete assertions and shared helper reference.'} for p in D], 'product_sources':[{'path':p,'lines':lines,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p,lines in prod.items()], 'entries':entries}
 (HERE/'evidence-batch8.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n');print(ev['counts'])
if __name__=='__main__':main()
