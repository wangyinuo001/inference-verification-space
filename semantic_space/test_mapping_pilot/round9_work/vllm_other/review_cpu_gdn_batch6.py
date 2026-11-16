"""Manual per-test CPU GDN review, with exact source and input evidence.
No tests run: this records assertion semantics, not execution verdicts.
"""
import ast
import csv
import hashlib
import json
from extract_review import HERE, SOURCE

PATH = 'tests/kernels/mamba/cpu/test_cpu_gdn_ops.py'
# These decisions follow a complete read of the test, local reference functions,
# and the CPU convolution/GDN adapter implementations.
DECISIONS = {
 'test_fused_gdn_gating_cpu': ('compute.operator_reference_relation', 'g matches -exp(A_log)*softplus(a+dt_bias) at 1e-4; sigmoid(b) beta matches at 5e-3.', [108], ''),
 'test_fused_sigmoid_gating_delta_rule_update_cpu': ('compute.operator_reference_relation', 'One-token decode output agrees with the explicit FP32 gated-delta recurrence at atol=rtol=1e-2; selected transposed final states are also compared.', [121,201], 'The final-state write is recorded separately as recurrent-state candidate evidence.'),
 'test_chunk_gated_delta_rule_cpu': ('compute.operator_reference_relation', 'Variable-length prefill output agrees with the explicit recurrence at atol=rtol=1e-2, including Q/K normalization and state dtype rounding every 64 tokens; final-state tensor is also compared.', [121,201], 'State continuation requires the separate recurrent-state relation.'),
 'test_chunk_gated_delta_rule_cpu_two_call_split': (None, 'needs_compute_recurrent_state_transition: split-prefill final state within 1e-3 and concatenated output within 2e-2 of one-call execution, with the first call state carried into the second.', [201], ''),
 'test_causal_conv1d_update_cpu_preserves_sd_state': (None, 'needs_compute_recurrent_state_binding: monkeypatched native operator receives the identical SD state and index objects and returns the same x object.', [477], 'Mocked forwarding, not native numerical correctness.'),
 'test_spec_aware_mixed_routing_preserves_token_order': (None, 'needs_compute_recurrent_state_binding: speculative/non-speculative input selections are contiguous and preserve [0,2]/[1,3] token selection; scatter restores mixed_qkv.', [], 'Both computational branches are recording mocks.'),
 'test_spec_aware_nonspec_materializes_state_indices': (None, 'needs_compute_recurrent_state_binding: strided block-table first-column indices become contiguous [0,4] before the convolution call.', [], 'Native kernels are replaced with recording/identity mocks.'),
 'test_spec_forward_prepares_native_conv_metadata': (None, 'needs_compute_recurrent_state_binding: AMX or Arm BF16 path forwards contiguous int32 indices [0,4], accepted counts [1,4], and packed-weight flag.', [], 'Forwarding metadata only; kernels are mocked.'),
 'test_causal_conv1d_torch_two_call_split': (None, 'needs_compute_recurrent_state_transition: fallback convolution output for two calls with carried state matches one call at 1e-2; no direct final-state assertion.', [465], ''),
 'test_causal_conv1d_update_cpu_accepts_wide_state': ('compute.operator_reference_relation', 'Native decode output with 5 extra state columns matches narrow-state output at 1e-2 under indices [2,0,1]; active state matches exactly and unused tail remains unchanged.', [465,486,494], 'Active-state/tail integrity retained as recurrent-state candidate; primary mapping covers numerical layout equivalence.'),
 'test_causal_conv1d_update_cpu_rejects_invalid_accepted_count': (None, 'needs_compute_recurrent_state_input_validation: RuntimeError matching num_accepted_tokens must be in.*seqlen for accepted count 0 or 17 with sequence length 16.', [486], ''),
 'test_causal_conv1d_fwd_cpu_two_call_split': (None, 'needs_compute_recurrent_state_transition: native convolution output split across initial-state False/True calls matches one-call output at 1e-2; final state not directly asserted.', [465], ''),
 'test_causal_conv1d_fwd_cpu_accepts_wide_state': ('compute.operator_reference_relation', 'Native prefill outputs for narrow and padded state agree at 1e-2 for lengths [63,69], indices [2,0], mixed initial-state [True,False]; active state and untouched tail agree exactly.', [465,486,494], 'Recurrent-state integrity recorded separately.'),
 'test_batch_memcpy_cpu_fallback': (None, 'needs_memory_copy_relation: ctypes fallback copies four source buffers into destinations for byte sizes [256,1024,68,4096]; dispatch registration type is also checked.', [], 'Memory-copy result, not a numerical inference operator.'),
 'test_causal_conv1d_fwd_cpu_ds_adapter_varlen_mixed_initial_state': ('compute.operator_reference_relation', 'DS-layout adapter and SD-layout path produce matching convolution output and state for lengths [2,3], state slots [2,0], initial-state [False,True].', [465,477,1298], 'Native convolution is replaced by Python reference; scratch shape/strides and remapped indices are separately asserted.'),
 'test_causal_conv1d_update_cpu_ds_adapter_wide_multi_token_copyback': (None, 'needs_compute_recurrent_state_binding: mocked mutation is copied from scratch into only slots [3,1,0], including wide tail; forwarded scratch indices are contiguous identity and accepted counts retained.', [477], 'No convolution numerical oracle; native update is a state-mutating mock.'),
 'test_conv_cpp_matches_torch': ('compute.operator_reference_relation', 'Native prefill output and final state agree with Torch fallback at atol=rtol=1e-2 for each declared variable-length batch.', [465,1260,1284], 'State tensor comparison retained separately.'),
 'test_conv_cpp_no_worse_than_torch_vs_fp32': ('compute.operator_reference_relation', 'Mean absolute error of native prefill versus explicit FP32 grouped-convolution+SiLU oracle is no greater than Torch fallback error plus 1e-3.', [465,1241,1260,1284], 'Relative error-budget check, not elementwise closeness.'),
 'test_conv_cpp_weight_packed_matches_torch': ('compute.operator_reference_relation', 'Packed-weight native prefill using noncontiguous SD state yields the same output as unpacked Torch fallback within 1e-2.', [465,1260,1284], 'No final-state assertion in this test.'),
 'test_conv_update_cpp_matches_torch': ('compute.operator_reference_relation', 'Across three signed-input decode steps, native output agrees with Torch convolution at 1e-2; complete indexed/padded state storage agrees exactly.', [494], 'State persistence and untouched slots recorded separately.'),
 'test_conv_weight_pack_roundtrip_unpacked_matches': ('compute.operator_reference_relation', 'Packed and unpacked native prefill produce bitwise-equal output and state for lengths [7,64,65].', [465,1284], 'Checks execution equivalence after packing, not pack/unpack inversion.'),
 'test_spec_decode_unpacked_conv_weight_stash': ('compute.operator_reference_relation', 'Torch prefill using stashed original weights agrees at 1e-2 with native packed-weight execution for lengths [7,65]; stash existence and exact equality to original weights are separately checked.', [1260], 'Does not call speculative forward; mirrors its weight selection.'),
}


ACCEPTED_UNIT_REFINEMENTS = {'test_chunk_gated_delta_rule_cpu_two_call_split': ('compute.execution_relation', 'Concatenated two-call prefill output matches one-call output at 2e-2; final state also checked at 1e-3.', 'Independent carried-state correctness remains recurrent-state candidate evidence.'), 'test_causal_conv1d_update_cpu_preserves_sd_state': ('compute.kernel_call_binding', 'SD-layout adapter forwards the identical state and index objects to the mocked native operator and returns its x object.', 'Mocked call-binding oracle; no native numerical execution.'), 'test_spec_aware_mixed_routing_preserves_token_order': ('compute.kernel_call_binding', 'Wrapper forwards contiguous speculative/non-speculative rows selected by [0,2]/[1,3] and scatters mocked outputs back to original token order.', 'The underlying numerical branches are mocked.'), 'test_spec_aware_nonspec_materializes_state_indices': ('compute.kernel_call_binding', 'Wrapper forwards contiguous int32 state-slot indices [0,4] from strided block-table first-column input.', 'Recording/identity kernel mocks only.'), 'test_spec_forward_prepares_native_conv_metadata': ('compute.kernel_call_binding', 'AMX or Arm BF16 wrapper forwards contiguous int32 slot indices [0,4], accepted counts [1,4], and packed-weight flag.', 'The native kernels are mocked.'), 'test_causal_conv1d_torch_two_call_split': ('compute.execution_relation', 'Torch convolution two-call output with carried state matches one-call output at atol=rtol=1e-2.', 'Final state is not directly asserted.'), 'test_causal_conv1d_fwd_cpu_two_call_split': ('compute.execution_relation', 'Native convolution split output with has_initial_state False then True matches one-call output at atol=rtol=1e-2.', 'Final state is not directly asserted.'), 'test_causal_conv1d_update_cpu_rejects_invalid_accepted_count': ('compute.input_admissibility', 'RuntimeError matching num_accepted_tokens must be in.*seqlen for accepted count 0 or 17 and sequence length 16.', 'Native kernel feature-gated to AMX or Arm BF16.')}
for _name, (_unit, _prop, _limit) in ACCEPTED_UNIT_REFINEMENTS.items():
    DECISIONS[_name] = (_unit, _prop, DECISIONS[_name][2], _limit)


# Seed accepted these recurrent-state obligations after concrete source review.
_name = 'test_causal_conv1d_update_cpu_ds_adapter_wide_multi_token_copyback'
DECISIONS[_name] = ('compute.recurrent_state_transition', 'Mocked native mutation is copied from scratch into only slots [3,1,0], including wide tail; forwarded scratch indices are contiguous identity and accepted counts retained.', DECISIONS[_name][2], 'State copyback oracle with mocked arithmetic; no native convolution numerical claim.')
_name = 'test_chunk_gated_delta_rule_cpu_two_call_split'
DECISIONS[_name] = ('compute.recurrent_state_transition', DECISIONS[_name][1], DECISIONS[_name][2], 'First-call final state seeds second call; both final state and concatenated output are checked against full-call execution.')


def exact_entry(path, name):
    text=(SOURCE/path).read_text();tree=ast.parse(text)
    target=name.split('::')[-1]
    matches=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==target]
    if '::' in name:
        cls=name.split('::')[0]
        matches=[n for c in ast.walk(tree) if isinstance(c,ast.ClassDef) and c.name==cls for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==target]
    assert len(matches)==1,(path,name)
    n=matches[0]
    checks=[]
    for node in ast.walk(n):
        if isinstance(node,ast.Assert) or (isinstance(node,ast.Call) and any(t in ast.unparse(node.func) for t in ['assert_close','assert_equal','assert_array','pytest.raises'])):
            checks.append({'line':node.lineno,'source':ast.get_source_segment(text,node)})
    return {'definition_line':n.lineno,'end_line':n.end_lineno,'decorators':[{'line':d.lineno,'source':ast.get_source_segment(text,d)} for d in n.decorator_list], 'checks':sorted(checks,key=lambda x:x['line']), 'full_test_source':ast.get_source_segment(text,n)}


def main():
    packets=[json.loads(l) for l in (HERE/'source-packets.jsonl').open()]
    packets=[p for p in packets if p['key'][1]==PATH]
    assert {p['key'][2] for p in packets}==set(DECISIONS)
    rows=[];pending=[];entries=[]
    for p in packets:
        snapshot,path,name,line=p['key'];unit,prop,helpers,limit=DECISIONS[name]
        e=exact_entry(path,name)
        refs=';'.join(f"{path}:{c['line']}" for c in e['checks'])
        conditions='; '.join(d['source'].replace('\n',' ') for d in e['decorators'])
        conditions+='; exact input construction and complete source in evidence-batch6.json entry '+name
        entries.append({'key':p['key'],'source_sha256':p['sha256'],'unit_id':unit,'property':prop,'scope_limit':limit,'helper_lines':helpers,**e})
        if unit: rows.append([snapshot,path,name,line,unit,prop,refs,conditions,'vllm_cpu_gdn_per_test_review_20261116'])
        else:pending.append([snapshot,path,name,line,prop,refs])
    for filename,fields,data in [('crosswalk-batch6.tsv',['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],rows),('pending-batch6.tsv',['snapshot','path','test_name','line','reason','evidence_source'],pending)]:
        with (HERE/filename).open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t');w.writerow(fields);w.writerows(data)
    product_paths=['vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py','vllm/model_executor/layers/mamba/ops/cpu/causal_conv1d.py','vllm/_custom_ops.py']
    payload={'batch':'cpu_gdn_per_test','counts':{'scope':len(packets),'mapped':len(rows),'pending':len(pending)},'review_method':'Every assigned test body, referenced local helper and CPU adapter read; direct checks and input source retained without truncation. No runtime execution claimed.', 'file_reviews':[{'snapshot':'vllm','path':PATH,'source_sha256':hashlib.sha256((SOURCE/PATH).read_bytes()).hexdigest(),'review':'Full file read, including explicit gated-delta recurrence, FP32 convolution oracle, state-layout helpers and native/Torch execution wrappers.'}], 'product_sources':[{'path':p,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p in product_paths], 'entries':entries}
    (HERE/'evidence-batch6.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(payload['counts'])
if __name__=='__main__':main()
