"""Completion aggregation, async lookup and remote-decode shared helper cohort."""
import ast,json,hashlib,csv
import continue_runtime_review as r
G='disagg.transfer_gate';C='cache.lifetime';K='disagg.kv_mapping'
start=len(r.ledger['decisions'])
def add(p,n,u,prop,cond):r.add(p,n,u,prop,cond)
p='tests/v1/kv_connector/unit/test_output_aggregator.py'
for n,u,prop,cond in [
('test_aggregate_workers_output',G,'Aggregator exposes send/receive completion only after request vote threshold, unions current invalid blocks and retains failure until receive completion','Expected2workers; severalsteps with disjointcompletionvotes and req3failure before2recv votes'),
('test_aggregate_workers_output_with_expected_finished_count',G,'Updated expected worker count applies to new transfer counters while preserving already-started counter requirements','Initial4; req1remaining3 then2afterlaterack; newreq2 completeswith2; no retroactive req1 threshold reset'),
('test_replay_preserves_get_and_clear_fields',G,'Replayed salvaged completion/failure/invalid-block state combines with recovered step and reaches completion threshold','Initial4 changedto2; sendreq1,recvreq2/3,failedreq3,badblock7; metadataunion asserted additionally; cacheevents exclude salvagedoldvote'),
('test_replay_completes_finished_recving_votes',G,'Fully salvaged receive completion votes replay once even when recovered workers no longer mention request','Two salvaged req1recv votes plusfailure; two empty recovered outputs; complete+failedreq1'),
('test_salvaged_kv_cache_events_do_not_raise_quorum','cache.event_payload_binding','Salvaged cache-event votes do not increase live worker quorum or publish stale unmatched event','Expected2liveworkers; E0oneoldvote,E1twofreshvotes; emittedE1only andnumworkers2'),
('test_salvaged_votes_dropped_live_votes_unaffected_lmcache_exact_quorum','cache.event_payload_binding','LMCache event aggregation preserves a matching live event when handling salvaged votes','One salvagedE plusone liveE withotherworkerquiet; emittedtokenpayload[1,2,3,4]'),
('test_salvaged_votes_dropped_live_votes_unaffected_mooncake_quorum','cache.event_payload_binding','MooncakeStore event aggregation preserves matching live event with salvaged same-event vote','One salvagedE plusone liveE; pop_common_events emitsEexactpayload'),
('test_salvaged_event_votes_are_one_shot','cache.event_metadata_lifetime','Salvaged event votes are discarded after a quiet recovered step and cannot affect later event quorum','Recovery emitsNone; later2freshE1votes emitE1only')]:add(p,n,u,prop,cond+'; CPU synthetic ModelRunnerOutput and event objects, no actual KV transfer')
p='tests/v1/kv_connector/unit/test_umbp_async_lookup.py'
for n,u,prop,cond in [
('test_scheduler_async_lookup_defers_then_returns_hit','cache.hit_accounting','Async lookup stays unresolved until result and reports only one reusable block leaving final prompt token to compute','32tokens,16blocksize; repeatedpollNone then16; one key/context construction; gatedfakehandle'),
('test_async_lookup_discards_results_for_changed_request','cache.safe_reuse','Completed lookup from old request prefix context is discarded after local-progress,length,hash or request-object change','Fourchangevariants; secondfuture resubmitted thenallmiss yields0; same textualrequestID objectreplacement stillinvalidates'),
('test_scheduler_reset_clears_pending_lookup_state','cache.safe_reuse','Store reset cancels pending lookup future and discards cached load specification','Injected future notstarted; resettrue,futurecancelled,bothmaps empty'),
('test_async_lookup_failure_recomputes_and_allows_retry','cache.hit_accounting','Async lookup error or malformed result reports zero reuse and later valid retry reports one block','TimeoutError or wronglength[True,False] then[True]; expected0then16 on32tokenprompt')]:add(p,n,u,prop,cond+'; ROCm+mori guard,synthetic scheduler fixture; source only no thread run')
p='tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py'
for n,u,prop,cond in [
('test_basic_lifecycle',C,'Remote-decode prefill completion retains request KV block references until sending completion then releases scheduler/cache state','2.5blocks,max1newtoken; refs1afterfinish; scheduledfinish notification alone doesnotrelease; finished_sending triggers assert_scheduler_empty'),
('test_short_prompt_lifecycle',C,'Partial-block remote decode retains transfer metadata and cleans request/cache after finished_sending','Halfblockprompt; one remote_block_ids group asserted, not exactgroupblockcount; cleanuphelper checksallfree'),
('test_prefix_cache_lifecycle',K,'Remote-decode transfer metadata includes full and partial requested blocks despite prefix-cache hit','Prime3.5blocks thenremote2.5blocks; sum groupblockcounts==3; finished_sending cleanup alsochecked'),
('test_abort_during_kv_transfer',C,'Abort does not release pending remote-decode transfer state until sending completion','2.5blocks; scheduler.requests retainedbefore/afterabort; finished_sending thenempty/free/refzero')]:
 add(p,n,u,prop,cond+'; realCPU scheduler with synthetic model-runner outputs; no model decoding')
 d=next(d for d in r.ledger['decisions'] if d['path']==p and d['test_name']==n)
 hp='tests/v1/kv_connector/unit/utils.py';hb=(r.ROOT/hp).read_bytes();hs=hb.decode();hf=next(f for f in ast.parse(hs).body if isinstance(f,ast.FunctionDef) and f.name=='assert_scheduler_empty');checks=r.review.direct_checks(hf,hs.splitlines())
 ts=(r.ROOT/p).read_text();tf=next(f for f in ast.parse(ts).body if isinstance(f,ast.FunctionDef) and f.name==n);call=next(c for c in ast.walk(tf) if isinstance(c,ast.Call) and ast.unparse(c.func)=='assert_scheduler_empty')
 d['helper_oracles']=[dict(path=hp,function=hf.name,oracle_lines=[c['line'] for c in checks],call_line=call.lineno,call_expression=ast.unparse(call),binding='Direct final cleanup helper; checks scheduler queues/request maps empty, request-to-block maps empty, free count total minus null block, and every block reference count zero. Does not require hash cache eviction.')]
p='tests/v1/kv_connector/unit/test_multi_connector.py'
add(p,'test_loaded_groups_follow_the_connector_serving_the_request',K,'MultiConnector exposes loaded group IDs only from connector chosen to serve request','Two mockedconnectors; chosen0/1 reports16matchedtokens,other0; get_loaded_group notcalledonother')
for n,prop,cond in [
('test_multi_connector_children_use_configured_lookahead_hashes','Resolved lookahead hash count propagates to all connector children and nested leaves','12support/prefixenabled/disabled/role/nestedcases; count0/1/2; explicitMTPsetuphelper'),
('test_multi_connector_hma_support_detection','MultiConnector HMA capability is true only if all constructed children support HMA','Classdeclaresprotocol; none/all/mixedinbothorders inspected'),
('test_divergent_local_hybrid_hit_capability_is_conservative','Divergent local-hybrid-hit capability requires every child support it','TwoDivergenttrue versusmixedfalse'),
('test_multi_connector_role_follows_children','Effective consumer/producer roles derive from child roles independent of top-level role','Fourrolecombinations incltopproducer withbothchild andtopboth withproduceronly'),
('test_multi_connector_mixed_hma_disables_hybrid_kv_cache','Constructing LLM with mixed HMA children disables HMA and reports aggregate unsupported capability','Qwen3-0.6B,GPUguard,eager,V1multiprocessing0; FakeNixlWrapper plusMockConnector; configuration assertionsonly,no generation')]:add(p,n,r.E,prop,cond)
# Explicit shared input/reference source contexts, kept separate from oracles.
for d in r.ledger['decisions']:
 if d['path'] in ['tests/v1/kv_connector/unit/test_remote_decode_lifecycle.py','tests/v1/kv_connector/unit/test_multi_connector.py']:
  ep='tests/v1/kv_connector/unit/utils.py';eb=(r.ROOT/ep).read_bytes();d['reviewed_extra_source_files']=[dict(path=ep,source_sha256=hashlib.sha256(eb).hexdigest(),purpose='Reviewed config/request/scheduler/model-output constructor helpers and final assert_scheduler_empty at69-357; no source execution')]
 if d['path']=='tests/v1/kv_connector/unit/test_umbp_async_lookup.py':
  ep='tests/v1/kv_connector/umbp_test_utils.py';eb=(r.ROOT/ep).read_bytes();d['reviewed_extra_source_files']=[dict(path=ep,source_sha256=hashlib.sha256(eb).hexdigest(),purpose='Reviewed _kv_cache_config/_vllm_config/_SchedulerHandle at260-410; blocksize16,8slots,lookupreturns suppliedhits')]
(r.HERE/'reviewed-decisions.json').write_text(json.dumps(r.ledger,ensure_ascii=False,indent=2)+'\n')
# Every remaining MultiConnector entry was read, but cannot be assigned a
# stronger existing cache/transport property merely because it asserts.
notes=json.loads((r.HERE/'pending-boundaries.json').read_text());known={(x['path'],x['test_name']) for x in notes['entries']};mapped={(x['path'],x['test_name']) for x in r.ledger['decisions']}
for row in r.allrows:
 if row['snapshot']!='vllm' or row['path']!='tests/v1/kv_connector/unit/test_multi_connector.py':continue
 key=(row['path'],row['test_name'])
 if key in mapped or key in known:continue
 n=row['test_name']
 if n.startswith('TestMultiConnectorStats::'):reason='Reviewed complete stats body: typed deserialization/object preservation/aggregation/reset/exporter/logging contract. No existing seed unit for connector metric-payload algebra; not startup config, KV content or runtime message delivery.'
 elif 'mem_pool_context' in n:reason='Provider selection/no-provider/multiple-provider conflict only; not shared-region attachment or resource lifetime.'
 elif 'consistency' in n:reason='Compares two persisted copies and callback event ordering, but neither original KV reference nor model output equality. Preserve distinct cross-store consistency/property gap.'
 elif 'worker_metadata' in n:reason='Connector-indexed metadata aggregate/demultiplex with nested assertion callbacks; no actual transport. Needs exact metadata composition unit.'
 elif 'engine_id' in n:reason='Two fresh configs get distinct engine IDs; no request-to-transfer binding or socket endpoint proof.'
 elif 'cache_hit_sources' in n:reason='Delegates source-count dictionary object to selected connector; no recomputed token count/upper-bound check.'
 else:reason='Callback delegation or API override completeness only; no full transfer readiness/lifetime consequence asserted.'
 notes['entries'].append(dict(path=row['path'],test_name=n,reason=reason))
(r.HERE/'pending-boundaries.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')
summary={'new_entries':len(r.ledger['decisions'])-start,'total_decisions':len(r.ledger['decisions']),'complete_source_review_files':['test_multi_connector.py','test_output_aggregator.py','test_umbp_async_lookup.py','test_remote_decode_lifecycle.py'],'multi_connector_unmapped':sum(x['path'].endswith('/test_multi_connector.py') for x in notes['entries']),'execution':'Pinned-source only; no tests/model/threads/processes executed'}
(r.HERE/'batch7-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
