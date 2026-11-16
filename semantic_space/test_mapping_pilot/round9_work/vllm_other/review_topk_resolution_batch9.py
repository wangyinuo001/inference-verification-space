"""Revisit previously reviewed pending top-k checks after seed expansion."""
import csv,hashlib,json
from extract_review import HERE,SOURCE
from review_cpu_gdn_batch6 import exact_entry
PATH='tests/kernels/test_top_k_per_row.py'
DIS='compute.kernel_dispatch_binding';OUT='compute.output_constraints'
D={
 'test_aiter_c4a_topk_kernel_selection':(DIS,'Compressed-indexer AITER predicate accepts/rejects explicit prefill/decode compression/row/length cases and rejects when disabled.',['_enable_aiter_topk']),
 'test_aiter_dsa_topk_kernel_selection':(DIS,'Uncompressed DSA accepts topk2048 decode and >=32768 prefill, rejects other topk or short/unknown prefill length and disabled AITER.',['_enable_aiter_topk']),
 'test_dsv4_native_top_k_window':(DIS,'Native DSV4 preference holds only for tested topk512,row128,column524288; row385,column1048577,or topk1024 cases reject.',[]),
 'test_cooperative_topk_512_tie_workspace_is_per_row':(OUT,'For two rows with distinct tie scores, row0 selected IDs are all>=2048, excluding its lower-score prefix; no full top-k reference equality.',['_has_device_capability']),
 'test_cooperative_topk_overflow_nan':(OUT,'With2048 injected NaNs and topk512, all selected IDs are unique,in-range and reference NaN input positions.',['_run_topk_backend','_has_cooperative_topk']),
 'test_persistent_topk_masked_row_tail':(OUT,'For lengths k+1..k+4 and masked fill, selected IDs are unique and within valid length and include all declared finite columns; padded tail cannot be selected.',['_run_topk_backend']),
 'test_sparse_indexer_topk_backend_resolution':(DIS,'Auto follows cooperative→persistent→per_row constraints; explicit backends honor support or raise matching errors; BF16 compatibility respects16-byte row alignment.',['_has_cooperative_topk','_has_flashinfer_topk']),
 'test_sparse_indexer_topk_aiter_requires_rocm':(DIS,'Explicit AITER backend on CUDA raises RuntimeError containing ROCm.',['_make_topk']),
 'test_sparse_indexer_topk_backend_resolution_aiter':(DIS,'On ROCm, auto stays per_row, explicit eligible AITER resolves; unsupported k,disabled tuned kernels,or disabled AITER raise specific RuntimeErrors.',['_patch_aiter_topk','_enable_aiter_topk','_make_topk']),
 'test_sparse_indexer_topk_aiter_applies_no_shape_gate':(DIS,'Explicit supported AITER dispatch calls mocked AITER twice despite crossing call-site performance threshold and never uses native fallback.',['_patch_aiter_topk','_enable_aiter_topk','_make_topk']),
 'test_sparse_indexer_topk_auto_stays_on_per_row':(DIS,'Auto ROCm topk1024 invokes native per-row mock exactly once and never invokes AITER.',['_patch_aiter_topk','_enable_aiter_topk','_make_topk']),
 'test_sparse_indexer_topk_aiter_expands_mtp_row_ends':('compute.kernel_call_binding','Two-step decode seq_lens[100,200] is forwarded to AITER with next_n1 and expanded row ends[99,100,199,200].',['_patch_aiter_topk','_enable_aiter_topk','_make_topk']),
}

def main():
 packets=[p for p in (json.loads(l) for l in (HERE/'source-packets.jsonl').open()) if p['key'][1]==PATH and p['key'][2] in D]
 assert len(packets)==len(D)
 rows=[];entries=[]
 for p in packets:
  snap,path,name,line=p['key'];unit,prop,hn=D[name];e=exact_entry(path,name);helpers=[exact_entry(path,h) for h in hn]
  checks=[{'path':path,**c} for c in e['checks']]
  refs=';'.join(f"{path}:{c['line']}" for c in checks)
  cond='; '.join(d['source'].replace('\n',' ') for d in e['decorators'])+'; exact input and mocks in evidence-batch9.json entry '+name
  entries.append({'key':p['key'],'source_sha256':p['sha256'],'unit_id':unit,'property':prop,**e,'shared_helper_sources':helpers})
  rows.append([snap,path,name,line,unit,prop,refs,cond,'vllm_topk_pending_revisited_20261116'])
 with (HERE/'crosswalk-batch9.tsv').open('w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']);w.writerows(rows)
 with (HERE/'pending-batch9.tsv').open('w',newline='') as f:csv.writer(f,delimiter='\t').writerow(['snapshot','path','test_name','line','reason','evidence_source'])
 prod={'vllm/model_executor/layers/indexer_topk.py':[174,264,324,405],'vllm/_aiter_ops.py':[2962,3002]}
 ev={'batch':'previously_pending_topk','counts':{'scope':len(packets),'mapped':len(rows),'pending':0},'file_reviews':[{'snapshot':'vllm','path':PATH,'source_sha256':hashlib.sha256((SOURCE/PATH).read_bytes()).hexdigest(),'review':'Preserves batch1 numerical review and adds complete-body rereview of12 formerly pending checks using now-accepted dispatch/call/output units; chunk partition tests remain pending.'}], 'product_sources':[{'path':p,'lines':lines,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p,lines in prod.items()], 'entries':entries}
 (HERE/'evidence-batch9.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n');print(ev['counts'])
if __name__=='__main__':main()
