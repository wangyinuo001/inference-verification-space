"""Source-reviewed KV connector mapping and WRITE lifecycle families."""
import ast,json,hashlib
import continue_runtime_review as r
H=r.HERE
K='disagg.kv_mapping';G='disagg.transfer_gate'
start=len(r.ledger['decisions'])
def add(p,n,u,prop,cond):r.add(p,n,u,prop,cond)
p='tests/v1/kv_connector/unit/test_tp_mapping.py'
for n,prop,cond in [
('test_source_ranks_homogeneous','Homogeneous TP2 consumer rank1 reads source rank1','_compute_mapping constructs TransferTopology directly'),
('test_source_ranks_d_gt_p','TP4 consumer rank2 reads source rank1 from TP2 producer','Synthetic topology'),
('test_source_ranks_p_gt_d','TP1 consumer reads both TP2 producer ranks','Synthetic topology'),
('test_head_sharded_draft_under_mla_target_gets_gqa_offset','GQA draft head override under MLA target restores expected rank head offset while keeping same source ranks','Three TP/head tuples: offsets3,1,2; base MLA offsetzero'),
('test_mla_dcp_source_ranks','MLA source ranks reflect local and remote DCP partitioning','Five explicit TP/DCP tuples and exact expected rank tuples'),
('test_dcp_consumer_count_matches_readers','Each source rank advertised consumer count equals number of computed reader mappings','All consumer ranks evaluated over five TP/DCP topologies; no transport or wait execution'),
('test_build_src_split_handles','TP fan-in split descriptors shrink head slice length while retaining physical block stride and count','TP1 consumer with TP2/4 producers; 8 dense1024-byte blocks'),
('test_fa_and_ssm_different_split_factors','FA and SSM regions derive different source-rank slice offsets according to their reader sets','Two FA200-byte runs and one SSM400-byte run; explicit descriptor arrays'),
('test_hetero_block_size_splits','Block-size-ratio FA sub-block runs pass through while SSM region splits per source rank','ratio2; FA runs count2 and SSM400-byte run; exact returned descriptor arrays'),
('test_hetero_block_size_head_sharded_asserts','Head-sharded FA reads reject simultaneous block-size mismatch','Two FA sources and ratio2; expected Head-sharded assertion'),
('test_csa_linear_tp_layout_boundary','CSA linear transfer accepts only local/remote TP configurations on same KV-head layout side','Eight total-head/localTP/remoteTP tuples; unsupported sharding boundary raises ValueError')]:add(p,n,K,prop,cond+'; CPU synthetic worker/config with no network data transfer')
p='tests/v1/kv_connector/unit/test_moriio_tp_mapping.py'
for n,prop,cond in [
('test_write_tp_mapping_targets','WRITE mapping chooses destination ranks, local/remote head slices and writer counts for homogeneous/fan-in/fan-out TP','Ten explicit MLA/GQA cases incl replicated producer no-write targets'),
('test_mla_fan_in_with_mamba_still_writes_state_from_every_producer','MLA non-latent-writer producer retains Mamba state slice contribution','Producer3/TP4 to TP2; zero heads, two writers, remote slice1/2'),
('test_write_tp_mapping_rejects_unmappable_topologies','WRITE topology planner rejects nondividing TP sizes and impossible head sharding','Four invalid TP/head cases; ValueError match'),
('test_every_consumer_head_and_state_slice_is_written_exactly_once','Across every producer the computed WRITE plan covers each consumer head and state slice once and advertises actual writer count','Cartesian valid TP1/2/4/8, heads1/2/4/8/16, MLA and Mamba booleans; asserts global head identity and exact sorted coverage'),
('test_head_slices_land_byte_exact','Computed source/destination head slice offsets reconstruct consumer head bytes and leave untouched block zero','CPU byte buffers; all9 pairs LBNHC/LBHNC/LHBNC layouts; TP1to2/2to4/2to1; blocks[0,2]to[2,1]; _apply simulates copies'),
('test_head_slices_of_separate_kv_planes','Head slice offsets move both separate K/V planes to requested destination block only','CPU uint8 shape[2,3,4,4,4] to twohead destination; sourceblock1heads2:4 -> destblock2'),
('test_token_major_head_slices_merge_to_one_run_per_token','Token-major adjacent head slices coalesce to four16-byte transfer runs','4 tokens;2 selected heads; descriptor size assertion only'),
('test_mamba_state_fans_out_and_gathers_back','Mamba conv projections and SSM state retain bytes across TP fan-out then fan-in','CPU uint8 padded slots; replicated B/C false/true; _move_state applies offsets; source slot1->two slot2 shards->slot0'),
('test_mamba_state_rejects_mismatched_shards','Mamba slice planner rejects incompatible projection shard sizes','Big(8,4,4) versus small(3,2,2); ValueError cannot map')]:add(p,n,K,prop,cond)
add(p,'test_consumer_completes_after_every_writer_reported',G,'Consumer exposes completed transfer only after advertised writer count and clears count; absent count defaults to one','Synthetic structured write_done twice with num_writers2 then legacy one-writer message; no actual network')
add(p,'test_consumer_write_failure_drops_partial_write_count',G,'Consumer failure clears partial writer counter and reports transfer failed','One of two expected writes recorded then write_failed; failed ID set exact')
p='tests/v1/kv_connector/unit/test_moriio_kv_layout.py'
for n,prop,cond in [
('test_supported_layouts_compute_expected_geometry_and_offsets','Per-layer KV geometry produces declared local/remote strides and exact byte transfer offsets','Nine shapes covering separated/interleaved/shuffled/kernel-block/MLA; localblocks[1,3],remoteblocks[4,5]'),
('test_kernel_block_layout_without_spec_dimensions_rejects_ambiguous_axes','Kernel-block geometry rejects ambiguous axes without full attention spec dimensions','Shape(2,32,8,2,3) with blocksize-only namespace'),
('test_mixed_layers_compute_distinct_offsets_per_layer','Separated/interleaved/MLA layer descriptors yield distinct offset triples for same block IDs','Three cache tensors/specs; only pairwise inequality asserted, not exact triples'),
('test_write_transfer_plan_caches_offsets_per_layer_geometry','WRITE plan records separately computed offsets for each layer including same-shaped dense layers','Fake offset helper returns incrementing markers; three calls and three cached entries; no equivalent-geometry dedup claim'),
('test_hybrid_write_attention_uses_its_own_group_not_the_first_group','Hybrid attention write selects its layer group block IDs and region session','draft group2 uses local[30,31],remote[60,61] and session5; mock argument assertion'),
('test_local_block_ids_longer_than_remote_raises','Transfer offset planner rejects source block list longer than destination list','Local four blocks versus remote two; ValueError'),
('test_shorter_local_block_ids_still_transfers_only_what_decode_allocated','Shorter source block list emits only one transfer run despite two remote allocations','One sourceblock; count assertion only'),
('test_block_ids_must_fit_registered_regions','Transfer offset planner rejects negative source or out-of-range destination block','Source-1 or remote8 with8 physicalblocks'),
('test_empty_local_block_ids_is_free_only_noop','Empty source block list produces empty source offsets, destination offsets and sizes','Empty local and nonempty remote[4,5]'),
('test_registration_regions_do_not_split_interleaved_or_mla_cache','Registration descriptors split separated K/V planes but preserve interleaved and MLA whole regions','Checks exact data pointers and byte lengths; no actual register_memory called'),
('test_registration_regions_use_layer_num_blocks','Registration descriptor extent uses actual layer tensor block count','Four physicalblocks versus worker.num_blocks8; expected4*2*48bytes'),
('test_unsupported_shape_raises_value_error','Unsupported cache shape is rejected against full attention spec','Shape(8,3,5,7), incompatible dimensions; ValueError'),
('test_mamba_transfer_geometry_is_slot_strided','Mamba transfer geometry spans full padded slot strides rather than compact element count','Conv5slots stride200 and SSM5slots stride64; expected region bytes from shape0*stride0*itemsize'),
('test_kda_conv_ssm_unpacks_padded_pages_as_aliasing_strided_views','Conv/SSM views preserve packed page offsets, strides and shared storage with correct component writes','4 uint8 pages padded128bytes; BF16 conv(3,4),SSM(2,2,4); writes observed in correct backing byte windows'),
('test_kda_conv_ssm_accepts_strided_blocks_without_copying','Conv/SSM views preserve padded interblock stride without copying backing storage','128-byte views over256-byte physical stride; BF16 view stride128elements and same data pointer'),
('test_kda_conv_ssm_rejects_noncontiguous_page_bytes','Conv/SSM unpacking rejects stride-two inner page bytes','uint8 pages sliced[...,::2]; ValueError contiguous byte dimension'),
('test_kda_conv_ssm_rejects_non_byte_packed_cache','Conv/SSM unpacking rejects non-byte cache dtype','BF16 packed-cache input; expected byte-sized dtype rejection'),
('test_mamba_transfer_geometry_requires_matching_slot_counts','Mamba geometry rejects different conv/SSM slot counts','Conv5 versus SSM4; ValueError'),
('test_mamba_transfer_geometry_rejects_empty_caches','Mamba geometry rejects zero-slot cache pair','Both leading dimensions zero; ValueError'),
('test_mamba_conv_ssm_offsets_homogeneous_tp','Homogeneous Mamba offset template places conv subprojections and whole SSM slots at explicit padded-stride byte locations','Two slots[1,2],convstride200,SSMstride64; exact offsets/sizes derived from reviewed GDN split fixture'),
('test_mamba_conv_ssm_offsets_heterogeneous_tp_is_gated','Legacy Mamba offset template explicitly rejects unsupported heterogeneous TP ratio','tp_ratio2 raises NotImplementedError; distinct from newer compute_mamba_slice_offsets supporting hetero TP'),
('test_local_mamba_slots_must_fit_registered_regions','Mamba template rejects local slot outside registered range','Local6 with6slots; expected ValueError'),
('test_mamba_slot_counts_must_match','Mamba template requires equal local/remote slot list lengths','Two versus one and one versus two'),
('test_remote_mamba_slots_must_be_non_negative','Mamba template rejects negative remote slot','Local1,remote-1; expected ValueError')]:add(p,n,K,prop,cond+'; ROCm+mori module guard; fixture constructs CPU tensors/mock worker; source review only')
for n,prop,cond in [
('test_write_scheduler_deduplicates_layers_and_seals_expected_count','WRITE scheduling deduplicates repeated layer and seals expected count to queued unique layers','dense0 repeated and indexer; queue2 and writes_expected2'),
('test_hybrid_write_posts_both_regions_before_layer_completion','Hybrid layer write retains both conv/SSM completion statuses and increments layer count without premature completion notification','Mock event synchronize; exact two write calls partition3conv+1SSMrun; writes_done1,completion_notified false'),
('test_write_completion_notifies_once_after_all_sealed_writes_finish','WRITE completion waits scoped statuses and sends one notification only after both sealed layers finish, then cleans transfer state','Fake waiting recorder; first mark no notification, second+repeat finalize one notification at port7002, one ACK, statusclear and terminal marker'),
('test_moriio_wrapper_waits_scoped_statuses_without_global_drain','Waiting one transfer checks only its supplied status and preserves unrelated global pending status','Fake Succeeded counters; scoped1/global0; no network'),
('test_write_done_cleans_scheduler_state','Completed WRITE transfer cleans all scheduler bookkeeping and marks terminal','Inspect direct body before assignment; synthetic writer and wrapper'),
('test_schedule_write_rejects_terminal_transfer_without_recreating_state','Terminal transfer rejects late layer schedule without queueing or recreating bookkeeping','Mark request done then schedule indexer; empty queue and no counters/sets'),
('test_late_remote_blocks_message_is_ignored_after_transfer_done','Terminal transfer ignores delayed remote_blocks metadata without recreating allocation state','Msgpack payload transfer xfer after terminal mark')]:
 # Name of cleanup test is resolved from the body reviewed at lines580-603.
 if n=='test_write_done_cleans_scheduler_state':
  src=(r.ROOT/p).read_text();tree=ast.parse(src)
  n=next(x.name for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.lineno<=590<=x.end_lineno)
 add(p,n,G,prop,cond)
# Valid/invalid message routing mixes payload interpretation and completion
# semantics. Preserve as unresolved instead of treating the parameterized
# entry as a single KV placement/transfer-gate assertion.
# Precise process failure transport for second batch, avoiding generic wrappers.
for d in r.ledger['decisions']:
 if not d.get('helper_oracles'):continue
 refs=[]
 for h in d['helper_oracles']:
  expr=h['call_expression']
  if expr.startswith('_run_two_gpu_quick_allreduce_test('):refs.append((d['path'],151))
  elif expr.startswith('_run_cudagraph_replay_test('):refs.append((d['path'],264))
  elif expr.startswith('run_quick_reduce_llm_e2e('):refs.append((d['path'],619))
  elif expr.startswith('_distributed_run('):refs.append((d['path'],101 if d['path'].endswith('test_dcp_a2a.py') else 126))
 for path,line in set(refs):
  b=(r.ROOT/path).read_bytes();source=b.decode().splitlines()[line-1].strip();assert source
  ref=dict(path=path,line=line,source=source,source_sha256=hashlib.sha256(b).hexdigest())
  if ref not in d.setdefault('helper_transport_evidence',[]):d['helper_transport_evidence'].append(ref)
(H/'reviewed-decisions.json').write_text(json.dumps(r.ledger,ensure_ascii=False,indent=2)+'\n')
summary={'new_entries':len(r.ledger['decisions'])-start,'total_decisions':len(r.ledger['decisions']),'files':[f'tests/v1/kv_connector/unit/{s}' for s in ['test_tp_mapping.py','test_moriio_tp_mapping.py','test_moriio_kv_layout.py']],'pending':['test_moriio_wrapper_routes_valid_messages: mixed metadata routing/completion parameter cases','test_moriio_wrapper_rejects_invalid_messages: protocol admissibility needs specific unit'],'execution':'Pinned-source static audit only'}
(H/'batch3-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
