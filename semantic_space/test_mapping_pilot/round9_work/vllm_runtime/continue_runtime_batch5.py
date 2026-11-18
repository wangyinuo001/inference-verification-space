"""MoRIIO byte-copy simulation and routing plus connector configuration."""
import ast,json,hashlib
import continue_runtime_review as r
K='disagg.kv_mapping';G='disagg.transfer_gate';W='schedule.work_distribution'
start=len(r.ledger['decisions'])
def add(p,n,u,prop,cond,helpers=()):r.add(p,n,u,prop,cond,helpers)
p='tests/v1/kv_connector/unit/test_moriio_read_tp.py'
for n,prop,cond in [
('test_equal_tp_read_lands_every_byte','Equal-TP READ places every attention/state shard byte at expected destination and leaves other bytes zero','All six model fixtures with supported HMA true/false; TP1/2; token/head/mixed layouts'),
('test_read_without_advertised_prefill_tp_is_homogeneous','Absent advertised producer TP defaults to homogeneous READ with expected per-rank cache bytes','advertised_tp0; all model fixtures,HMAtrue,TP1/2,three layoutpairs'),
('test_replicated_read_releases_every_holder','Replicated MLA/GQA READ at heterogeneous TP preserves expected destination shard bytes','MLA six TPpairs; GQA replicatedTP2to4/4to2; three layoutpairs; release-count assertions are additional, mapping cites only byte-equality helper')]:
 add(p,n,K,prop,cond+'; fake transport applies planned byte slices on CPU; real worker offset/source resolver, no RDMA/GPU',['_read_all'])
 key=next(k for k in r.bykey if k[0]==p and k[1].split('::')[-1]==n)
 d=next(d for d in r.ledger['decisions'] if (d['path'],d['test_name'])==key)
 h=d['helper_oracles'][0];src=(r.ROOT/p).read_text();fn=next(x for x in ast.walk(ast.parse(src)) if isinstance(x,ast.FunctionDef) and x.name=='_read_all')
 h['oracle_lines']=[c['line'] for c in r.review.direct_checks(fn,src.splitlines()) if 'torch.equal(received, expected)' in c['expression']]
 h['binding']='Direct nested _read_all call returns into _assert_released; byte oracle is in _read_all for every consumer/layer. Fake transport copies actual planned slices; _expected independently fills destination blocks from unsharded state. Release assertions not used as byte oracle.'
for n,u,prop,cond in [
('test_flexible_mirror_spreads_reads_and_releases_every_holder',W,'RDMA mirror READ rotates selected producer over all four prewarmed TP sessions and releases all holders','Four repeated requests; selectedsets[{0},{1},{2},{3}]; exactsessionprewarm and holderrelease sets; no bytecomparison here'),
('test_xgmi_mirror_reads_the_fixed_source',W,'xGMI mirror READ consistently selects sole prewarmed producer session and releases all holders','Three requests; one prewarmed session; selection matches its rank, four holderreleases'),
('test_aborted_read_releases_every_holder',G,'Aborted READ sends releases to all computed holders without submitting payload reads','Emptylocal; TP4to2/2to4/2to2 with advertisedTP0 fallback; consumer.sent empty'),
('test_failed_read_releases_every_holder',G,'Failed READ releases every producer holder, marks destination blocks invalid and does not report successful completion','MLA TP2to1; fake transport copies but returns failure; no claim failed bytes usable'),
('test_head_read_rejects_peer_with_other_head_count',K,'Head-slice READ rejects peer metadata with mismatching per-rank head count','MixedlayoutGQA; peer numheads halved; MoRIIOError'),
('test_head_read_rejects_block_ids_outside_the_cache',K,'Head-slice READ rejects local or remote block IDs outside cache region','LocalNUM_BLOCKS,remoteNUM_BLOCKS and remote-1; ValueError'),
('test_head_read_rejects_more_local_than_remote_blocks',K,'Head-slice READ rejects longer destination local-block list than remote source list','MixedlayoutGQA; one extra localblock; ValueError'),
('test_read_holder_tp_ranks',K,'READ holder set matches homogeneous and heterogeneous TP replication mapping','Four explicit rank/topology tuples yield(1),(1),(2,3),(0..7)')]:add(p,n,u,prop,cond+'; CPU fake ranks/cache transport; source audit only')
ext='tests/v1/kv_connector/unit/test_moriio_hetero_tp_write.py';eb=(r.ROOT/ext).read_bytes()
for d in r.ledger['decisions']:
 if d['path']==p:d['reviewed_extra_source_files']=[dict(path=ext,source_sha256=hashlib.sha256(eb).hexdigest(),purpose='Reviewed imported _build_rank/_expected/_global_state/_groups plus MODELS and block constants; CPU real-byte reference fixture')]
p=ext
add(p,'test_write_lands_every_shard_byte_exact',K,'WRITE simulation places every consumer attention/recurrent-state shard byte exactly and leaves unwritten blocks zero','Six model fixtures inclsliding/GQA/MLA/GDNhybrid; six TPpairs,HMA supportedmodes,three layoutpairs; fake transport applies planned CPU bytecopies; direct finaltorch.equal oracle; completion counters are additional assertions')
add(p,'test_write_refuses_peer_with_different_block_size',K,'WRITE rejects incompatible producer/consumer attention block sizes before moving requested layer','MLA producerTP2consumerTP1; peerblocksize doubled; real writer with fake sessions; MoRIIOError same --block-size')
p='tests/v1/kv_connector/unit/test_moriio_routing_fairness.py'
for n,prop,cond in [
('test_kv_read_load_is_hardware_fair','Connector READ selection visits every producer GPU equally under a fair owner-DP input stream','Five TP8/DP8EP configurations;16rounds; Counter exactcoverage and equalcounts; fairness depends on supplied fair input stream'),
('test_flexible_gate_fires_only_for_tp_prefill_dp_decode','Flexible mirror routing is enabled only for singleton-DP replicated TP producer and TP1 decode','Five topology fixtures; exact flagset compared to dimensions'),
('test_symmetric_tp_is_a_bijection','Symmetric TP8 READ routing covers every producer rank once with flexible mode disabled','All8consumer ranks; sorted targetset0..7'),
('test_owner_dp_read_is_faithful_and_covers_every_rank','Owner-DP READ keeps TP0 and routes across explicitly supplied owning DP ranks','DP8producer,TP8consumer rank3; all ownerDP0..7; noflexible'),
('test_flexible_round_robin_is_deterministic_uniform_and_staggered','Flexible READ round-robin is uniform across cycles and staggered by consumer DP rank','64choices across8TP,firstcycle0..7; firstpicks across8DP coverall'),
('test_read_blocks_for_req_threads_chosen_tp','Resolved flexible TP choice is forwarded consistently into READ submission','ConsumerDP3 choosesTP3, capturedkwargs flexible true; notifyoffset3')]:add(p,n,W,prop,cond+'; real selector on __new__ mock workers; no measured hardware throughput')
p='tests/v1/kv_connector/unit/test_moriio_proxy_routing.py'
for n,prop,cond in [
('test_full_slot_space_is_covered_uniformly','Proxy route covers all instance/DP slots equally over full cycles','Six role shapes;3cycles; expectedNoneDP forDP1'),
('test_instance_and_dp_rank_marginals_are_balanced','Proxy route balances instance and DP-rank marginal counts','Six role shapes;5cycles; DP1 usesNone'),
('test_tp_instance_forwards_no_dp_rank','Proxy TP-only instance route emits no DP rank','TwoTPinstancesDP1;8requests; allDPNone'),
('test_two_instance_dp8_gives_every_node_all_ranks','Two-instance DP8 proxy route visits all eight DP ranks on each instance','16requestfullcycle; both ranksets0..7'),
('test_consecutive_requests_alternate_instances','Proxy route interleaves the first three requests across three instances','3instancesDP8; exactinstanceorder0,1,2')]:add(p,n,W,prop,cond+'; real example router loaded via _load_proxy_module with heavy thirdparty imports stubbed')
add(p,'test_discovery_heartbeat_progresses_while_worker_holds_gil',r.M,'Heartbeat child keeps delivering unchanged discovery payload while owner holds GIL','Linux PyDLLusleep300ms; at least3received messages; after idempotent shutdown no furtherheartbeat; source only no sockets/process launched')
add(p,'test_discovery_heartbeat_exits_when_worker_dies','shared_resource.lifetime','Heartbeat subprocess exits after owner death within deadline','Linux procfs; killowner, childabsent or zombie within5seconds; PID-specific observation; sourceonly')
add(p,'test_discovery_heartbeat_recovers_after_router_outage',r.M,'Heartbeat resumes delivering unchanged discovery payload after router outage','Actual testwould bind ROUTER, observe send-timeout then rebind; payload equality and childstillalive; sourceonly no network/processrun')
p='tests/v1/kv_connector/unit/test_config.py'
for n,u,prop,cond in [
('test_kv_connector',r.E,'Offload backend and size normalize to expected connector/role/extra-config while preserving existing fields','Native/lmcache/disabledfivecases; nativeglobalbytes4/8GiB,LMCacheMP omits serverownedcapacity; stubLMCacheclass'),
('test_kv_connector_rejects_expandable_segments',r.A,'Configured KV connectors reject expandable CUDA segments without protected allocator','Nixl/MooncakeV1/OOTconnector; envTrue; ValueError'),
('test_sleep_mode_refuses_unsupported_kv_connector',r.A,'Sleep mode rejects connector backend/protocol modes lacking support','NixlLIBFABRIC,NixlPush,MooncakeTCP/TENT explicitcases'),
('test_mooncake_store_supports_sleep_with_explicit_rdma',r.E,'Explicit RDMA with auto-discovery disabled preserves MooncakeStore sleep capability despite NVLink override envs','TempJSON protocolrdma/devicemlx5_0; MC_MS_AUTO_DISC0,MNNVL/NVLINK1; no realMooncakepackage'),
('test_kv_offloading_size_only_uses_native_default',r.E,'Offload size alone enables native OffloadingConnector with kv_both role and total byte budget','4GiB; expectedcpu_bytes4*2^30'),
('test_nixl_connector_transfer_mode',r.E,'Nixl factory resolves default/push modes and rejects differently cased unsupported mode','None->NixlPull,push->NixlPush,Push->ValueError; configuration binding only')]:add(p,n,u,prop,cond)
p='tests/v1/kv_connector/unit/test_hma_auto_config.py'
add(p,'test_hma_auto_config',r.E,'HMA auto-disable follows connector support including all/mixed MultiConnector children','Mock platform supportsHMA; SimpleCPUOffload false,Example true,allHMAchildren false,mixed true for disableflag')
add(p,'test_explicit_hma_with_non_hma_connector_errors_at_factory',r.A,'Explicit HMA enabled with unsupported connector is rejected by factory','CPUconfig,ExampleConnector,emptyKVCacheConfig; expectedValueError')
(r.HERE/'reviewed-decisions.json').write_text(json.dumps(r.ledger,ensure_ascii=False,indent=2)+'\n')
summary={'new_entries':len(r.ledger['decisions'])-start,'total_decisions':len(r.ledger['decisions']),'pending':['test_kv_connector_allows_expandable_segments_with_sleep_mode','test_kv_connector_allows_expandable_segments_with_cumem_allocator','test_kv_connector_allows_other_alloc_conf','test_no_kv_connector_ignores_expandable_segments'],'pending_reason':'No direct assertions; successful-return admissibility only, kept separate from effective-state guarantee.','execution':'Pinned-source review; no source package imports, GPU, server, process or socket execution'}
(r.HERE/'batch5-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
