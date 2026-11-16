"""Explicit per-entry CUTLASS helper review; skipped/opcheck-only tests remain pending."""
import ast,csv,hashlib,json
from extract_review import HERE,SOURCE
from review_cpu_gdn_batch6 import exact_entry
PATH='tests/kernels/quantization/test_cutlass_scaled_mm.py'
FP8={
 'test_cutlass_fp8_gemm': 'MNK_FACTORS, tensor/per-token activation and tensor/per-output-channel weight scales, optional bias',
 'test_cutlass_fp8_blockwise_scale_gemm': 'MNK_FACTORS + BLOCKWISE_PREFILL_FACTORS, (1,128)/(128,128) block scales, no bias; non-divisible shapes return before oracle',
 'test_cutlass_fp8_gemm_output_dtype': '512x512x512, BF16/FP16 output, scale granularities and bias',
 'test_cutlass_fp8_blockwise_scale_gemm_dtype': '512x512x512, block scales, BF16/FP16 output, no bias',
 'test_cutlass_fp8_gemm_devices': '512x512x512, CUDA_DEVICES, BF16 output, scale granularities and bias',
 'test_cutlass_fp8_gemm_m_sweep': 'N=K in [32,64,96], M=1..127, scale granularities and bias',
}
INT8={
 'test_cutlass_int8_gemm': 'MNK_FACTORS, tensor/per-token activation and tensor/per-output-channel weight scales, optional bias',
 'test_cutlass_int8_gemm_output_dtype': '512x512x512, BF16/FP16 output, scale granularities and bias',
 'test_cutlass_int8_gemm_devices': '512x512x512, CUDA_DEVICES, BF16 output, scale granularities and bias',
 'test_cutlass_int8_gemm_m_sweep': 'N=K in [32,64,96], M=1..127, scale granularities and bias',
}

def main():
 packets=[p for p in (json.loads(l) for l in (HERE/'source-packets.jsonl').open()) if p['key'][1]==PATH]
 assert {p['key'][2] for p in packets}==set(FP8)|set(INT8)|{'test_cutlass_int8_azp','test_cutlass_int8_azp_bias_fold','test_cutlass_support_opcheck'}
 rows=[];pending=[];entries=[]
 for p in packets:
  snapshot,path,name,line=p['key'];e=exact_entry(path,name);unit='compute.operator_reference_relation';helper=[];note='Custom-op opcheck is additional registration/schema checking, not a numerical reference oracle.'
  if name in FP8:
   helper=[exact_entry(PATH,'cutlass_fp8_gemm_helper')];cond=FP8[name];prop='FP8 scaled GEMM output matches FP32 dequantized group-broadcast matmul (cast before bias) at rtol=0.5, atol=0.15.'
  elif name in INT8:
   helper=[exact_entry(PATH,'cutlass_int8_gemm_helper')];cond=INT8[name];prop='INT8 scaled GEMM output matches FP32 dequantized group-broadcast matmul (cast before bias) at rtol=0.1, atol=1.0.'
  elif name=='test_cutlass_int8_azp':
   cond='M=[32,64,128], N=[16,32,64], K=[64,128,256], BF16/FP16, optional bias, per-token or tensor activation zero point'
   prop='Asymmetric INT8 GEMM matches both dequantized FP32 matmul and CPU INT32 matmul plus scales/bias, atol=1e-3 and rtol=1e-2 BF16 or 1e-3 FP16.'
  elif name=='test_cutlass_int8_azp_bias_fold':
   unit=None;cond='Unconditional @pytest.mark.skip';prop='unconditional_skip_prevents_numerical_oracle_execution';note='Body has numerical comparisons, but this collected definition is unconditionally skipped because bias folding loses precision.'
  else:
   unit=None;cond='capability_int from current platform';prop='custom_operator_registration_opcheck_requires_distinct_unit';note='Calls tests.kernels.utils.opcheck on a capability predicate, with no expected capability verdict or numerical output asserted.'
  checks=[{'path':PATH,**c} for c in e['checks']]+[{'path':PATH,**c} for h in helper for c in h['checks']]
  refs=';'.join(f"{c['path']}:{c['line']}" for c in checks)
  entries.append({'key':p['key'],'source_sha256':p['sha256'],'unit_id':unit,'property':prop,'input_state':cond,'scope_limit':note,**e,'shared_helper_sources':helper,'all_checks':checks})
  if unit:rows.append([snapshot,path,name,line,unit,prop,refs,cond+'; exact test/helper source: evidence-batch7.json','vllm_cutlass_per_entry_helper_review_20261116'])
  else:pending.append([snapshot,path,name,line,prop,refs or f'{path}:{line}'])
 for file,fields,data in [('crosswalk-batch7.tsv',['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],rows),('pending-batch7.tsv',['snapshot','path','test_name','line','reason','evidence_source'],pending)]:
  with (HERE/file).open('w',newline='') as f:w=csv.writer(f,delimiter='\t');w.writerow(fields);w.writerows(data)
 sourcepaths=[PATH,'tests/kernels/utils.py','vllm/_custom_ops.py']
 payload={'batch':'cutlass_shared_helper_numeric','counts':{'scope':len(packets),'mapped':len(rows),'pending':len(pending)},'review_method':'Full test bodies and both shared numeric helpers read; baseline_scaled_mm and group broadcasting inspected. No source-family automatic propagation or runtime pass claim.', 'file_reviews':[{'snapshot':'vllm','path':PATH,'source_sha256':hashlib.sha256((SOURCE/PATH).read_bytes()).hexdigest(),'review':'Every assigned entry reviewed, both GEMM helpers and shared baseline; unconditional skip and opcheck-only entry remain pending.'}], 'source_hashes':[{'path':p,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p in sourcepaths], 'reference_helpers':[exact_entry('tests/kernels/utils.py',n) for n in ['baseline_scaled_mm','to_fp8','to_int8','opcheck']],'entries':entries}
 (HERE/'evidence-batch7.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n');print(payload['counts'])
if __name__=='__main__':main()
