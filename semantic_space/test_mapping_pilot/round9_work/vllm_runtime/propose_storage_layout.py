"""Explicit vLLM-only storage layout proposal; no source execution."""
import ast,csv,hashlib,json
from pathlib import Path
import build_runtime_review as b
H=b.HERE;R=b.ROOT;U='cache.storage_layout_mapping'
P='vllm/distributed/kv_transfer/kv_connector/v1/offloading/canonical_mapping.py'
Q='vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/layout.py'

def ref(path,line,kind='source'):
 data=(R/path).read_bytes();src=data.decode().splitlines()[line-1].strip();assert src
 return dict(path=path,line=line,kind=kind,source=src,source_sha256=hashlib.sha256(data).hexdigest(),snapshot_commit='76a0ccc209343971c6f4b56f6380a76202428385')
unit={'id':U,'name':'缓存本地布局与存储页映射','kind':'obligation','framework_scope':['vllm'],'phase':'cache','state_scope':'shared','conditions':['cache_storage_transition','parallel_topology','numeric_format','execution_layout'],
'property':'缓存页序列化为所声明的存储布局或从中恢复时，映射描述符保持逻辑层、头、token 或状态分量对应的地址、偏移、长度与次序；分片和副本写者按约定覆盖存储页，不支持的几何布局明确拒绝或采用声明的不透明页回退。',
'observable':['local_offsets','storage_offsets','segment_lengths','logical_block_ids','shard_ids','writer_coverage','layout_rejection_or_fallback'],
'oracle':'按声明布局展开描述符，与独立的字节位置计算或逻辑切片比较；检查本地页和选定写者对存储页的精确覆盖，以及不同 TP/块大小表示的往返。描述符检查不替代真实存储完成后的内容比较。',
'evidence':{'vllm':[ref(P,x) for x in [115,139,144,292,295,296,324,327,328,339,379,389,454,458]]+[ref(Q,x) for x in [218,264,297,301,396,409,431,621,623]]},
'snapshot_evidence_note':'Only fixed vLLM commit is asserted. This storage-page mapping obligation is distinct from P/D request transfer placement and from actual completed tier IO. No SGLang TP-portable storage-layout equivalence is claimed for admitted 49e384ce9d304648e9959666ecb8ce8cd98d0deb; a separate body-level comparison is required.'}
# Validate oracle label anchors, replacing a multiline test anchor with an
# actual assert line selected from the already-reviewed corresponding body.
for path in ['tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py','tests/v1/kv_connector/unit/test_mooncake_store_layout.py']:
 if True:
  s=(R/path).read_text();t=ast.parse(s)
  name='test_split_nhd_placement_rank2_of_4' if 'canonical_mapping' in path else 'test_rank_local_descriptors_match_reference'
  fn=next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
  a=next(n for n in ast.walk(fn) if isinstance(n,ast.Assert));unit['evidence']['vllm'].append(ref(path,a.lineno,'existing_oracle'))
ready=json.loads((H/'units-ready.json').read_text());ready['units']=[u for u in ready['units'] if u['id']!=U]+[unit];(H/'units-ready.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
needs=json.loads((H/'needs-space.json').read_text());needs['proposals']=[u for u in needs['proposals'] if u.get('candidate_id')!=U]+[dict(candidate_id=U,unit_id=U,property=unit['property'],framework_scope=['vllm'],vllm_evidence=[f"{v['path']}:{v['line']}" for v in unit['evidence']['vllm']],sglang_evidence=[],scope_note=unit['snapshot_evidence_note'])];(H/'needs-space.json').write_text(json.dumps(needs,ensure_ascii=False,indent=2)+'\n')
allrows=list(csv.DictReader(b.ALL.open(),delimiter='\t'));by={(x['path'],x['test_name'].split('::')[-1]):x for x in allrows if x['snapshot']=='vllm'}
rows=list(csv.DictReader((H/'proposed-crosswalk.tsv').open(),delimiter='\t'));rows=[x for x in rows if x['unit_id']!=U];new=[];evidence=[]

def propose(path,name,prop,cond,verify_tiling=False):
 row=by[(path,name)];data=(R/path).read_bytes();src=data.decode();lines=src.splitlines();tree=ast.parse(src);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.lineno==int(row['definition_line']));checks=b.direct_checks(fn,lines);citations=[f"{path}:{c['line']}" for c in checks];helpers=[]
 if verify_tiling:
  hb=(R/P).read_bytes();hs=hb.decode();hf=next(n for n in ast.parse(hs).body if isinstance(n,ast.FunctionDef) and n.name=='_verify_tiling');hc=b.direct_checks(hf,hs.splitlines());calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='_verify_tiling'];assert calls
  citations.extend(f"{P}:{c['line']}" for c in hc);helpers=[dict(path=P,function='_verify_tiling',source_sha256=hashlib.sha256(hb).hexdigest(),line=hf.lineno,end_line=hf.end_lineno,source='\n'.join(hs.splitlines()[hf.lineno-1:hf.end_lineno]),checks=hc,calls=[dict(line=c.lineno,expression=ast.unparse(c)) for c in calls],binding='Direct imported assertion helper verifies exact local intervals and selected-writer canonical intervals; exceptions propagate to test')]
 assert citations,(path,name)
 out={k:row[k] for k in ['snapshot','path','test_name','line']};out.update(unit_id=U,property=prop,oracle_source=';'.join(citations),conditions=cond+'; pinned source static review; no storage/GPU/test execution',rule_id='runtime_storage_layout_source_review_20261116');new.append(out)
 evidence.append(dict(**out,source_sha256=hashlib.sha256(data).hexdigest(),definition_line=fn.lineno,end_line=fn.end_lineno,definition_source='\n'.join(lines[fn.lineno-1:fn.end_lineno]),full_decorator_sources=[dict(line=d.lineno,source=ast.get_source_segment(src,d)) for d in fn.decorator_list],direct_checks=checks,helper_oracles=helpers,input_and_call_source=[dict(line=n.lineno,source=lines[n.lineno-1].strip(),expression=ast.unparse(n)) for n in ast.walk(fn) if isinstance(n,ast.Call)]))
p='tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py'
for n,prop,cond,helper in [
('test_split_nhd_placement_rank2_of_4','Split NHD rank2 TP4 maps local K/V token fragments into exact canonical offsets','Canonical4096bytes;8x128bytefragments;numwriters1',False),
('test_packed_nhd_placement_rank2_of_4','Packed NHD rank2 TP4 maps each token row to exact canonical offset','Four256byte runs offset512,1536,2560,3584',False),
('test_packed_hnd_placement_rank1_of_4','Packed HND adjacent rank head region coalesces into one canonical run','Local0->canonical1024,length1024',False),
('test_single_rank_coalesces_to_one_run','Single-rank supported layouts coalesce entire local page into one canonical run','Four split/packed NHD/HND forms;1024bytes',False),
('test_gqa_replicated_heads_rotate_writer','Replicated GQA ranks map same head bytes but alternate elected block writer with exact page tiling','2headsTP4;rank2/3writerrotation;directoffsets and _verify_tiling',True),
('test_mla_replicas_rotate_writer','MLA replicas share one latent-page representation and alternate writer by block','TP2;256bytecanonicalpage;oppositewriterbooleans atblocks0/1',False),
('test_dcp_interleaves_tokens_within_replicas','DCP interleaves canonical token positions inside head replicas and tiles page exactly','TP4DCP2,2heads;2048bytepage;notparallelismagnostic;allranksnumwriters1',True),
('test_mla_dcp_shards_latent_tokens','MLA DCP ranks map alternating latent token rows with exact tiling','TP2DCP2;512bytepage;rank0offset2o/rank1offset2o+64',True),
('test_pcp_tokens_and_tp_heads_compose','TP head shards and PCP token shards compose into an exact canonical page partition','TP2PCP2,2heads;nonempty per-rankruns and helpertiling',True),
('test_interleave_chunks_stay_contiguous','DCP interleave2 maps two-token chunks as contiguous canonical fragments','MLA TP2DCP2;rank0two128bytefragments at0/256',True),
('test_all_ranks_tile_canonical_page','All TP rank mappings cover each local page and tile canonical page exactly once','Four split/packed NHD/HND forms,TP4; direct _verify_tiling helper is soleoracle',True),
('test_cross_tp_store_load','Canonical descriptor store/load algebra reconstructs same reference bytes at TP4,2,1','8globalheads,4096bytes,4physicalforms; manual _load_one/_store_all, no actualstorage',False),
('test_cp_round_trip','DCP canonical descriptor load/store reconstructs full reference bytes','TP4DCP2,2heads,2048bytes; manual copies',False),
('test_replica_rotation_round_trip','Each elected MLA replica reproduces same canonical page','TP2;256bytepattern;block0/1writerrotation',False),
('test_fail_closed_cases','Canonical certification declines incompatible topology/spec/layout combinations','Headmismatch,nondividingTP,DCPbeyondreplication,interleave3,quantperhead,compressedMLA,unknownstrides,n attention KVspec; Noneassertions',False),
('test_opaque_fallback_places_page_whole','Opaque fallback places each rank page at disjoint whole-page storage offset','1024bytes4ranks rank2->offset2048,canonical4096;tilingverified',True),
('test_derive_mixed_model_with_dcp','Mixed canonical derivation retains all layers and degrades uncertifiable quant layer to opaque representation','Attention/MLA/quant TP4DCP2;quantcanonical4*unpaddedpage,notportable',False),
('test_derive_refuses_foreign_worker_groups','Canonical derivation declines worker world containing foreign PP groups','TP2PP2 instead ofTPxPCP exactworld; empty mapping',False)]:propose(p,n,prop,cond,helper)
p='tests/v1/kv_connector/unit/test_mooncake_store_layout.py'
for n,prop,cond in [
('test_tp_shared_layout_loads_partial_prefix_from_physical_block','Partial-prefix storage descriptors retain requested physical block for each local shard','Headmajor localTP2storeTP4;8oftokens16; descriptorcounts andblockIDs only, no copied-byte equality'),
('test_gdn_store_shard_segments_preserve_projection_boundaries','GDN shard descriptors preserve conv subprojection and SSM boundaries in DS/SD layouts','Exactsegment sizes andbase-relative offsets forshards0/1'),
('test_rank_local_descriptors_match_reference','Rank-local store descriptors equal independent block-address and span calculation','96regions/96or2lengths and1/1;random300blockIDs,chunks1/2/4; Pythoninttypes'),
('test_rank_local_descriptors_handle_empty_and_invalid_chunks','Rank-local descriptor builder emits empty result for no chunks and rejects unaligned span','Emptychunks; span129 withhashblock128 raisesAssertionError'),
('test_rank_local_layout_skips_block_padding','Rank-local descriptor lengths omit trailing page padding but include interblock padding inside contiguous multiblock chunks','Two layers,3pages64bytes,state40bytes;single40/multi104 lengths'),
('test_rank_local_database_api_rejects_tp_shared_layout','Rank-local database mutation rejects TP-shared store layout','set_block_len onHeadMajorTP2storeTP4 raisesRuntimeError rank-local'),
('test_attention_store_chunk_is_independent_of_k3_local_pages','Storage chunk size and schema identity remain independent of local attention page size','MLA block1536/11776,requestedstorechunk512;samefingerprint'),
('test_chunk_descriptors_preserve_order_across_kernel_blocks','Chunk descriptors serialize exact requested logical head/token slice in caller order across kernel blocks','Head/tokenmajor,chunks3/4;32queries vsfirst4 parity; expected explicitlogicalslice bytes'),
('test_tp_shared_layout_handles_kernel_blocked_compressed_states','Compressed-state kernel-block descriptors preserve four state rows in byte order','tokens_per_state4,logicalblock16;four8bytesegments andbytes0..31')]:propose(p,n,prop,cond)
rows+=new
with (H/'proposed-crosswalk.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,b.FIELDS,delimiter='\t');w.writeheader();w.writerows(rows)
full_sources=[]
for path in [P,Q,'tests/v1/kv_connector/unit/offloading_connector/test_canonical_mapping.py','tests/v1/kv_connector/unit/test_mooncake_store_layout.py']:
 rb=(R/path).read_bytes();full_sources.append(dict(path=path,source_sha256=hashlib.sha256(rb).hexdigest(),source=rb.decode(),review_scope='Complete test/helper bodies reviewed; production citation bodies reviewed'))
output=dict(unit=unit,entries=evidence,source_context=full_sources,count=len(new),snapshot='vllm',snapshot_commit='76a0ccc209343971c6f4b56f6380a76202428385')
(H/'storage-layout-proposals.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
pe=json.loads((H/'proposed-evidence.json').read_text());pe['storage_layout_mapping']=dict(count=len(new),file='storage-layout-proposals.json',unit=unit,method='Complete source-body review, concrete inputs and direct/helper assertions; vLLM-only pending seed merge')
(H/'proposed-evidence.json').write_text(json.dumps(pe,ensure_ascii=False,indent=2)+'\n')
print('storage proposal entries',len(new),'cumulative proposed',len(rows))
