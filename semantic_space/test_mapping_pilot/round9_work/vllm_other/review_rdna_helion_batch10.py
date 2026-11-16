"""Every RDNA determinism entry + eight reviewed Helion picker checks."""
import ast,csv,hashlib,json
from extract_review import HERE,SOURCE
from review_cpu_gdn_batch6 import exact_entry
RD='tests/kernels/quantization/test_rdna3_w4a16_determinism.py'
REPEAT={
 'test_scalar_splitk_bit_repeatable':'20 identical calls, M1 K4096 N4096 seed1234',
 'test_scalar_small_m_bit_repeatable':'20 identical calls, M8 K4096 N4096 seed1235',
 'test_scalar_single_split_bit_repeatable':'20 identical calls, M1 K256 N4096 seed1238',
 'test_wmma_splitk_bit_repeatable':'20 identical calls, M12 K6656 N4096 seed1236',
 'test_wmma_large_m_bit_repeatable':'20 identical calls, M128 K6656 N4096 seed1237',
 'test_wmma_v7v8_no_split_bit_repeatable':'20 identical calls, M512 N25600 K6656 seed1245; local replica of dispatch arithmetic checks no-split geometry',
 'test_scalar_splitk_100x_bit_repeatable':'100 unsynchronized identical calls, M1 K4096 N512 seed1249',
 'test_scalar_two_way_split_bit_repeatable':'20 identical calls, M1 K512 N512 seed1253',
}
NUMERIC={
 'test_scalar_matches_fp32_reference':'M1 K4096 N512 seed1240, scalar tolerance',
 'test_scalar_multi_m_matches_fp32_reference':'M2/3/4/5/8 K4096 N512 seed1247, scalar tolerance; repeat helper receives only one output and is vacuous',
 'test_scalar_single_split_matches_fp32_reference':'M1 K256 N512 seed1241, scalar tolerance',
 'test_wmma_matches_fp32_reference':'M12 K4096 N512 seed1242, WMMA tolerance',
 'test_wmma_single_split_matches_fp32_reference':'M12 K256 N512 seed1243, WMMA tolerance',
 'test_wmma_large_m_matches_fp32_reference':'M128 K4096 N512 seed1244, WMMA tolerance',
 'test_wmma_v7v8_no_split_matches_fp32_reference':'M512 N25600 K6656 seed1246, WMMA tolerance; local replica checks no-split geometry',
 'test_scalar_multi_row_tile_bit_repeatable':'M100 K4096 N2056 seed1250, scalar tolerance, also20 outputs bitwise equal',
 'test_scalar_clipped_tail_splitk':'M1 K4224/group128 or K2112/group32 N512 seed1251, scalar tolerance; repeat helper receives one output and is vacuous',
 'test_scalar_small_group_matches_fp32_reference':'M1 K2048 N512 group32/64 seed1252, scalar tolerance; repeat helper receives one output and is vacuous',
}
SPECIAL={
 'test_dispatch_boundary':('compute.kernel_dispatch_binding','Profiler kernel names identify scalar at M11 and WMMA at M12, excluding the other name family; both outputs additionally satisfy separate FP32 numerical oracle.',['_launched_kernel_names','_assert_close_to_ref']),
 'test_scalar_multistream_concurrent':('compute.execution_relation','Final outputs of25 invocations on each of two streams with different inputs/weights equal their respective serial outputs bitwise; intermediate iterations are not asserted.',[]),
 'test_scalar_graph_capture_replay':('compute.execution_relation','Captured result equals eager bitwise after first replay and after two further replays with unchanged inputs.',[]),
}

def main():
 packets=[json.loads(l) for l in (HERE/'source-packets.jsonl').open()]
 helion={(r['path'],r['test_name']) for r in csv.DictReader((HERE/'pending-batch3.tsv').open(),delimiter='\t') if r['reason']=='needs_kernel_configuration_selection_unit'}
 selected=[p for p in packets if p['key'][1]==RD or (p['key'][1],p['key'][2]) in helion]
 assert len(selected)==len(REPEAT)+len(NUMERIC)+len(SPECIAL)+len(helion)
 rows=[];entries=[];review_files=set();product_files=set()
 for p in selected:
  snap,path,name,line=p['key'];e=exact_entry(path,name);review_files.add(path)
  note='Source inspection only, no runtime verdict.'
  if path==RD:
   hn=['_build_layer','_prepare','_reference','_run_op','_outputs_and_ref']
   if name in REPEAT:unit='compute.execution_relation';cond=REPEAT[name];prop='Every repeated W4A16 GEMM output is torch.equal to the first output; numerical reference is built but not asserted.';hn+=['_assert_repeatable']
   elif name in NUMERIC:unit='compute.operator_reference_relation';cond=NUMERIC[name];prop='Finite output and max absolute error bounded against FP32 x@((q-8)*rounded_group_scales): scalar BF16<=0.10/FP16<=0.025; WMMA BF16<=0.25/FP16<=0.05.';hn+=['_assert_close_to_ref','_path_for']
   else:unit,prop,extra=SPECIAL[name];cond='gfx1100-only, FP16/BF16, dist_init rank/world1; exact input construction retained';hn+=extra
   if name=='test_scalar_multi_row_tile_bit_repeatable':hn+=['_assert_repeatable']
   if 'v7v8' in name:hn+=['_assert_v7v8_no_split_routing','_compute_wmma_k_split_mn'];note+=' Geometry helper computes a local heuristic replica, not observed backend dispatch.'
   helpers=[{'path':path,**exact_entry(path,h)} for h in hn]
   helpers.append({'path':'tests/conftest.py',**exact_entry('tests/conftest.py','dist_init')})
   product_files.update(['vllm/model_executor/kernels/linear/mixed_precision/rdna3_w4a16.py','csrc/rocm/q_gemm_rdna3.cu','tests/conftest.py'])
  else:
   unit='compute.kernel_dispatch_binding'
   cond='FakeTensorMode BF16 inputs; setup_method clears _pick_cache; literal candidate config keys in exact source'
   prop='Config picker returns the exact asserted CaseKey for '+('nearest available dimensions and next sufficient token capacity.' if name.endswith('closest_match') else 'oversized dimensions/token count falling back to largest available config.')
   note='Config-key selection only; no actual kernel launch or numerical comparison.'
   helpers=[{'path':path,**exact_entry(path,'_generate_fake_input')},{'path':path,**exact_entry(path,name.split('::')[0]+'::setup_method')}]
   prod='vllm/kernels/helion/ops/'+path.split('/')[-1].removeprefix('test_')
   helpers.append({'path':prod,**exact_entry(prod,'pick_config')});product_files.add(prod)
  checks=[{'path':path,**c} for c in e['checks']]+[{'path':h['path'],**c} for h in helpers for c in h['checks'] if h['path']==path]
  refs=';'.join(f"{c['path']}:{c['line']}" for c in checks)
  assert refs,(path,name)
  rows.append([snap,path,name,line,unit,prop,refs,cond+'; exact source/helper evidence-batch10.json','vllm_rdna_helion_per_entry_review_20261116'])
  entries.append({'key':p['key'],'source_sha256':p['sha256'],'unit_id':unit,'property':prop,'input_state':cond,'scope_limit':note,**e,'shared_helper_sources':helpers,'all_checks':checks})
 with (HERE/'crosswalk-batch10.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']);w.writerows(rows)
 with (HERE/'pending-batch10.tsv').open('w',newline='') as f:csv.writer(f,delimiter='\t').writerow(['snapshot','path','test_name','line','reason','evidence_source'])
 ev={'batch':'rdna_helion','counts':{'scope':len(selected),'mapped':len(rows),'pending':0},'file_reviews':[{'snapshot':'vllm','path':p,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest(),'review':'RDNA full file and helper/fixture chain individually reviewed; Helion assigned closest/fallback test bodies, FakeTensor helper, cache-reset setup, and concrete picker implementation reviewed.'} for p in sorted(review_files)],'product_sources':[{'path':p,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p in sorted(product_files)],'entries':entries}
 (HERE/'evidence-batch10.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n');print(ev['counts'])
if __name__=='__main__':main()
