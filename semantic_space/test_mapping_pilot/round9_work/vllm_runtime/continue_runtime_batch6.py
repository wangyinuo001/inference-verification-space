"""Complete native offloading configuration source family."""
import json,hashlib
import continue_runtime_review as r
p='tests/v1/kv_connector/unit/offloading_connector/test_config.py';start=len(r.ledger['decisions'])
props=[
('test_worker_kv_bytes_preserves_tensor_layout','Packed or unpacked views of same allocation yield same16-byte worker block size, worldsize6 and2blocks/chunk','TP3PP2; extra blocksize32; packedtwo-layer versus separateoffsetdescriptors'),
('test_offloading_skips_scratch_group','Worker and scheduler offload groups omit scratch while preserving original group indices and byte accounting','Attention,scratch,Mamba groups; expectedIDs0,2; one physical pagebytes'),
('test_hisparse_offloads_only_indexer_group','HiSparse offload metadata includes indexer group and excludes host-resident source','Bothworker/scheduler expectedgroup1 only'),
('test_hisparse_partial_group_size_survives_scheduler_flattening','HiSparse worker and flattened scheduler representations derive identical aggregate block-byte count','Two-layer uniform indexer versus flattened firstspec; equalswrapped.page_size_bytes'),
('test_zero_blocks_skips_tensor_layout_validation','Zero-block offload config reports zero worker bytes per block','Unpackedtwo-runallocation with num_blocksreset0'),
('test_prefill_context_parallelism_does_not_scale_group_blocks','PCP leaves token-per-block and hash span unchanged while explicit offload blocksize determines chunk ratio','PCP2; groupspan16,hash16,chunk4 at requested64'),
('test_dcp_scales_attention_but_not_mamba_group_blocks','DCP scales attention block span to32 while Mamba remains16 with onechunk window','TP2DCP2; scheduler alignment set; blockmask may be None or exactlyindices1,3, so no stronger reachablemask claim'),
('test_dcp_scales_uniform_type_attention_group_blocks','Uniform attention wrapper inherits DCP scaling for group and hash token spans','DCP1/2 ->16/32'),
('test_dcp_scales_uniform_type_group_alongside_mamba','Uniform attention group scales under DCP while uniform Mamba and standalone Mamba do not','DCP2; attention32/Mamba16, secondgroup16'),
('test_group_carries_the_chunks_the_tier_keeps_per_request','Offload group window metadata follows underlying attention/state specification','Full/MLA/hidden/unboundedwrapperNone,Mamba1,SWAandSWAwrapper8'),
('test_model_config_carries_the_longest_request','Offload model metadata retains configured max_model_len','32768 copied into offloading_config.model'),
('test_preserves_data_parallel_config','Offload parallel metadata preserves DP index,size and localrank','Index2,size4,localrank1'),
('test_resolves_heterogeneous_hybrid_block_sizes','Offload group token spans retain heterogeneous12/16 blocks and global hash unit4','Full+MLAoverlay; chunkratio1'),
('test_replicated_layout_enabled_for_pure_mla_tp_mp_single_node','Pure MLA single-node mp TP enables replicated offload layout in both runner versions','TP2/4/8; helper verifies scheduler/worker replication and blockbyte agreement'),
('test_replicated_layout_enabled_for_swa_mla','SWA-MLA group enables replicated offload layout','TP2/4; helper scheduler/worker agreement'),
('test_replicated_layout_dsv4_flash','Five synthetic uniform MLA/SWA groups enable replicated offload layout','TP2/4/8;2layers pergroup; not fullmodelrun'),
('test_replicated_layout_dsv3_2','Single synthetic uniform MLA group enables replicated offload layout','TP2/4/8;4layers; not fullmodelrun'),
('test_replicated_layout_enabled_for_uniform_wrapper_mla','Uniform MLA wrapper with different layer page sizes retains replicated layout','Two layers headsize512/256; aggregatepage accounting'),
('test_replicated_layout_excludes_unproven_cache_shapes','Replicated layout remains disabled for hidden-state and mixed MLA/full or MLA/Mamba cache shapes','Three explicit cachetensor/spec fixtures'),
('test_replicated_layout_supported_cache_shapes','Replicated layout enabled for SWA-MLA, distinct-size uniform MLA and multiple MLA groups','Three explicit tensorlayouts with contiguous layer runs'),
('test_replicated_layout_for_dsa_style_groups','Uniform DSA-style MLA+indexer groups replicate only when every inner spec is replicated','61+61layers; MLA+uint8MLA true versus MLA+fullattention false'),
('test_replicated_layout_parallel_gate','Replicated layout declines unsupported parallel/executor combinations','Ten overrides:TP1,nonMLA,PP2,PCP2,DCP2,world!=TP,ray,uni,class,multinode'),
('test_parallelism_agnostic_for_single_full_attention_group','Direct offload layout reports parallelism-agnostic only for tested single full-attention V1 configuration','V2false,singlefullgroup'),
('test_parallelism_agnostic_excluded','Direct V1 layout excludes MLA,SWA and multiple attention groups from portability flag','Three groupfixtures; flagfalse'),
('test_canonical_layout_gate','Canonical portability flag follows per-group certifiability and direct layout remainsfalse','MLA,attentionhybrid,uniformMLAtrue;SWA-MLA,Mambahybrid,compressedinnerfalse'),
('test_uncertifiable_canonical_layout_resets_replication','Uncertifiable canonical request disables replication instead of retaining direct-layout replica optimization','SWA-MLA or compressedMLA; directreplicationtrue,canonicalfalse; bothnotparallelismagnostic'),
('test_canonical_layout_certifies_v2_model_runner','Canonical full-attention config may report portability under V2 runner','DefaultV2 true; live tensor registration certificate not exercised here'),
('test_canonical_mla_dsa_rows_are_tp_independent','Canonical MLA/indexer config derives fixed chunkbyte size and capacity independent of TP and scheduler flattening','TP1/2/4,schedulerfalse/true;7layers uint8;196608bytes/chunk,8chunks;FileMapper recordsTP1replicatedtrue'),
('test_parallelism_agnostic_disabled_on_v2_model_runner','Direct V2 full-attention config disables parallelism-agnostic flag','Singlefullattention,zeroallocatedblocks'),
('test_accepts_blocks_per_chunk_for_heterogeneous_groups','Explicit blocks_per_chunk applies while heterogeneous group spans remain12/16','chunk2 independent of span mismatch'),
('test_block_size_and_blocks_per_chunk_are_mutually_exclusive','Offload config rejects simultaneous block_size and blocks_per_chunk','64 and2; ValueError Specifyonlyone'),
('test_blocks_per_chunk_must_be_positive','Offload config rejects nonpositive blocks_per_chunk','Zero; ValueError greaterthan0'),
]
for n,prop,cond in props:
 unit=r.A if n in ['test_block_size_and_blocks_per_chunk_are_mutually_exclusive','test_blocks_per_chunk_must_be_positive'] else r.E
 if n=='test_canonical_mla_dsa_rows_are_tp_independent':unit='cache.capacity_plan'
 helpers=['_replicated_layout'] if n.startswith('test_replicated_layout_') else []
 r.add(p,n,unit,prop,cond+'; explicit synthetic source/config fixtures; no actual tier IO or model execution',helpers)
# Retain the reviewed external mock class used by scheduler conversion.
extra='tests/v1/kv_connector/unit/offloading_connector/utils.py';b=(r.ROOT/extra).read_bytes()
for d in r.ledger['decisions']:
 if d['path']==p and any(x in d['test_name'] for x in ['offloading_skips_scratch','hisparse_offloads_only','dcp_scales_attention_but_not']):
  d['reviewed_extra_source_files']=[dict(path=extra,source_sha256=hashlib.sha256(b).hexdigest(),purpose='MockOffloadingSpec at140-189 reviewed: wraps OffloadingConfig, mock manager and worker; no actual offload byte IO in these configuration tests')]
(r.HERE/'reviewed-decisions.json').write_text(json.dumps(r.ledger,ensure_ascii=False,indent=2)+'\n')
summary={'new_entries':len(r.ledger['decisions'])-start,'total_decisions':len(r.ledger['decisions']),'complete_file':p,'entries_reviewed':len(props),'execution':'Pinned-source only; source code not imported/executed'}
(r.HERE/'batch6-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
