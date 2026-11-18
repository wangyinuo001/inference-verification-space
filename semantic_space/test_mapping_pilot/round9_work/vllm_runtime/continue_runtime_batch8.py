"""Storage allocator metadata and CPU payload-copy assertions."""
import ast,json,hashlib
import continue_runtime_review as r
start=len(r.ledger['decisions']);I='cache.state_invariants';K='disagg.kv_mapping';B='disagg.request_binding';T='cache.tier_transfer_content'
p='tests/v1/kv_connector/unit/test_hf3fs_metadata_server.py'
props=[
('test_allocate_pages','Rank page allocator returns requested count when capacity suffices and empty list otherwise','3freepages;allocate3or5; successfulrelease restores3'),
('test_release_pages_restores_count','Releasing allocated rank pages restores free-page count','4pages;allocate2 thenrelease ->2then4free'),
('test_release_pages_no_duplicates','Repeated release of same page does not increase free pool beyond capacity','3pages;allocate1 thenreleasepage0twice ->3free'),
('test_is_complete_false_until_all_ranks','Cache-key completion metadata remains false until each expected TP rank has a page','TP2 rank0page5 thenrank1page10'),
('test_get_rank_page_returns_none_for_missing_rank','Cache-key page lookup returns recorded page for present rank and None for absent rank','rank0page3,rank1absent'),
('test_get_all_pages','Cache-key metadata reports exact recorded rank/page pairs','Mapping{0:1,1:2}; set equality'),
('test_uninitialized_rank_raises_on_allocate','Page allocation for unknown rank is rejected','rank99; pytest.raises((ValueError,Exception)) accepts any Exception, not precise ValueError proof'),
('test_uninitialized_rank_raises_on_get_locations','Page location lookup for unknown rank is rejected','rank99; any Exception accepted by test'),
('test_basic_allocation_and_confirm','Confirmed allocation exposes a completed key at the exact allocated rank page','Singleinitializedrank4pages; allocateK>=0,confirm,existsTrue,locationexact'),
('test_allocate_pages_cache_hit_does_not_leak_pages','Mixed cached/new allocation reuses recorded page and consumes only one new free slot','Artificialcachedpage2 remains listed in freepoolfixture; only asserted countdelta1 and returnedcachedpage2, not global occupancy consistency'),
('test_allocate_pages_all_cache_hits_frees_all_slots','All-cache-hit allocation leaves free-page count unchanged and reuses recorded pages','ArtificialK1page0/K2page1 metadata,5freepages; returned0/1 and countunchanged'),
('test_allocate_returns_minus_one_when_pool_exhausted','Insufficient page pool rejects entire new-key allocation batch with sentinel-1','Onefreepage for2keys; all-or-nothing'),
('test_confirm_write_releases_pages','Explicit pages_to_release during confirmation restores free-page count','AllocateK thenconfirmwithsamepage inrelease list; count+1 only, not assertion that metadata removed')]
for n,prop,cond in props:r.add(p,n,I,prop,cond+'; metadata data structures only, no HF3FS server or IO')
p='tests/v1/kv_connector/unit/test_transfer_topology_sharded.py'
for n,prop,cond in [
('test_legacy_register_remote_engine_uses_pp_rank_zero','Legacy remote-engine metadata is indexed at PP0 and resolves expected TPsource','Localrank1TP4,remoteTP2; target[0]'),
('test_register_remote_engine_stores_pp_ranks_separately','Remote engine stores separate PP-stage topology/layer metadata without overwriting defaultPP0','SameenginePP0TP2/layers0:16 andPP1TP1/layers16:32'),
('test_helpers_use_requested_pp_rank','Transfer topology helpers use requested PP stage to select replication and source ranks','LocalTP2rank1; remotePP0TP1->[0],PP1TP4->[2,3] andreplicatedtrue'),
('test_engine_info_fields_have_backward_compatible_defaults','Remote topology metadata defaults missing PP rank and layer bounds to zero','Registeredlegacyengine; exactdefaultfields')]:r.add(p,n,K,prop,cond+'; pureCPU topology metadata, no transport')
p='tests/v1/kv_connector/unit/test_moriio_unmap.py'
for n,prop,cond in [
('test_exact_match_unmap_clears_both_tables','Unmapping exact request clears both request/transfer association tables','rid1<->tid1 thenrequestunmap'),
('test_unmap_via_transfer_id_after_request_id_mutation','Stable transfer ID unmaps original association after external request-ID mutation','cmpl-abc-0 to appendedsuffix; tid42 fallback'),
('test_unmap_miss_is_noop_and_does_not_raise','Unknown request/transfer unmap preserves unrelated live mapping','Unknownrid withNoneandunknowntid; existingrid1/tid1 retained'),
('test_exact_match_preferred_over_transfer_id_fallback','Exact request unmap with same supplied transfer ID clears both associations','BothIDsagree; despite testname does not exercise conflicting fallbackID precedence')]:r.add(p,n,B,prop,cond+'; lightweightstand-in binds realmap/unmapmethods')
p='tests/v1/kv_connector/unit/test_mooncake_store_layout.py'
for n,helper,prop,cond in [
('test_gdn_store_shards_round_trip_in_both_tp_directions','_assert_state_store_round_trip','GDN stored shard payload bytes read back into new consumer layout across both TP directions','TP4to2/2to4; DS/SDconv;4storeshards; explicitbytepatternspersegment/shard; keyandschema equalityadditional'),
('test_mamba2_replicated_groups_round_trip_in_both_tp_directions','_assert_state_store_round_trip','Mamba2 stored shard payload bytes read back into new consumer layout with replicated projection groups','TP4to2/2to4;DS/SDconv;sourcecache andnewdestcache differ; helpercompares descriptor-read bytes, not full logical unshardedstate')]:r.add(p,n,T,prop,cond+'; ctypes CPU simulated store/read; no Mooncake service',[helper])
r.add(p,'test_tp_shared_layout_round_trip_across_tp_sizes',T,'Stored TP4 attention shard bytes reconstruct TP2 consumer heads in expected layer order','Sixsame-layout plus3compatiblecross-layout pairs;2layers;realCPUtensors andctypescopy; expectedtorch.cat of paired producerheads; no realstorage')
r.add(p,'test_attention_store_layout_round_trip_across_local_block_sizes',T,'Stored attention chunks preserve values through small-to-large and large-to-small local-page roundtrip','Localblock400/800,storechunk400,head/tokenmajor;newzeroedconsumerandrestorebuffers; assert_close to originaltensors')
(r.HERE/'reviewed-decisions.json').write_text(json.dumps(r.ledger,ensure_ascii=False,indent=2)+'\n')
summary={'new_entries':len(r.ledger['decisions'])-start,'total_decisions':len(r.ledger['decisions']),'execution':'Source static only; no ctypes calls, storage, GPU or server execution'}
(r.HERE/'batch8-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
