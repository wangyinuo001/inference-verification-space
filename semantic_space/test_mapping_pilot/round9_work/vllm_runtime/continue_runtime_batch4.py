"""MoRIIO READ/WRITE completion and topology notification source cohort."""
import json
import continue_runtime_review as r
K='disagg.kv_mapping';G='disagg.transfer_gate';B='disagg.request_binding'
start=len(r.ledger['decisions'])
def add(p,n,u,prop,cond):r.add(p,n,u,prop,cond)
p='tests/v1/kv_connector/unit/test_moriio_tp_ack.py'
for n,u,prop,cond in [
('test_remote_tp_rank_same_tp_maps_to_self',K,'Same-TP remote rank mapping is identity','4 local and remote ranks'),
('test_remote_tp_rank_p4_d8_floor_maps_decode_to_prefill',K,'TP8 decode ranks map to TP4 prefill in repeated pairs','Exact [0,0,1,1,2,2,3,3]'),
('test_remote_tp_rank_p8_d4_maps_to_first_prefill_rank_per_pair',K,'TP4 decode ranks map to first prefill rank in each TP8 pair','Exact[0,2,4,6]'),
('test_remote_tp_rank_invalid_non_multiple_tp_raises',K,'Remote-rank mapping rejects nondividing TP sizes','TP6to4 and4to6 raise ValueError'),
('test_heterogeneous_tp_head_guard_rejects_split_kv_heads',K,'Legacy heterogeneous-TP head guard rejects unsupported split-head transfers','GQA4heads TP4to2/2to4; NotImplementedError replicated KV heads'),
('test_expected_ack_count_for_homogeneous_or_smaller_consumer_tp_is_one',G,'Producer release threshold is one for same/smaller consumer TP','Producer4consumer4 and producer8consumer4'),
('test_expected_ack_count_for_decode_fan_in',G,'Release threshold equals consumer-to-producer TP ratio','Producer4consumer8 ->2'),
('test_expected_ack_count_rejects_non_multiple_fan_in',G,'Release-count derivation rejects unsupported nondividing fan-in','Producer4consumer6 ValueError'),
('test_plain_string_ack_is_backward_compatible_single_ack',G,'Legacy string ACK immediately completes one live transfer and leaves no partial count','Live tx-plain; completedset exact'),
('test_structured_release_ack_waits_for_all_expected_acks',G,'Structured release waits two fan-in ACKs before completing and clearing partial count','ProducerTP4consumerTP8; firstNone/count1 then ID/countclear'),
('test_duplicate_ack_after_completion_does_not_resolve_twice',G,'Repeated post-completion release ACK cannot complete same transfer twice','Same structuredACK three times: None,ID,None; completedset retained'),
('test_ack_for_non_live_transfer_is_ignored',B,'ACK for absent transfer does not change live completion or count state','tx-stale versus live tx-live'),
('test_worker_get_finished_counts_structured_release_fan_in',G,'Producer get_finished exposes mapped request only after second fan-in release','READ workerTP4; fake wrapper successive ACK batches from consumerTP8'),
('test_worker_reports_unexpected_heartbeat_exit',G,'Worker completion polling raises when heartbeat subprocess exits unexpectedly','killed or stdinEOF; baseline empty completion then RuntimeError includes actual exit code; source only no subprocess run here'),
('test_read_completion_sends_structured_release_with_consumer_tp_size',G,'Completed READ statuses release producer blocks with consumer TP count and clear receive bookkeeping','Two DoneStatus reads under one layer; release metadata consumerTP8'),
('test_aborted_read_without_local_blocks_only_releases_remote_blocks',G,'Aborted READ with no local blocks sends remote release to selected peer with consumer TP count','Emptylocal, remoteattention+state blocks; chosenTP2 means notify7002; only send-notification assertion'),
('test_read_completion_waits_for_every_posted_read',G,'READ completion retains request and sends no release while any layer component remains pending','KDA success+pending status pair; infinite start time excludes timeout'),
('test_failed_read_reports_blocks_only_without_hma',G,'Failed READ reports invalid attention block IDs only when hybrid Mamba recovery is not enabled','has_mamba false returns{1,2},true empty; direct record_failed_recv'),
('test_hybrid_step_barrier_fails_closed',G,'Hybrid READ step barrier propagates transfer failure','has_mamba true; wrapper raises TransferError failed'),
('test_step_barrier_awaits_every_read_before_forward',G,'Step barrier submits all issued attention/indexer/draft reads once then clears issued list','Recording wrapper; repeated barrier invokes wait once with all3objects'),
('test_wait_for_layer_load_is_noop',G,'MoRIIO per-layer wait callback returns without touching worker','Worker __getattr__ raises on access; only per-layer noop established, step barrier separate'),
('test_step_barrier_without_mamba_does_not_raise_on_failure',G,'Nonhybrid step barrier absorbs terminal READ error and clears issued reads','Wrapper raises TransferError; this test does not itself assert invalid-block reporting'),
('test_step_barrier_failure_still_awaits_other_requests',G,'Nonhybrid barrier drains another pending request after one failed then reports failed blocks separately','Fake clock; late status succeeds on5thpoll; returned tx-b and invalid{1,2}'),
('test_hybrid_step_barrier_failure_does_not_wait',G,'Hybrid barrier fails immediately without polling unrelated pending request to completion','Fake clock no sleeps; pending remainsin_progress; SQ full error'),
('test_step_barrier_timeout_fails_closed_while_reads_in_flight',G,'Step barrier raises on timeout with reads in flight regardless of Mamba mode','Fake clock; transfer_timeout0.05 and two never-completing statuses'),
('test_step_barrier_native_timeout_fails_closed_while_reads_in_flight',G,'Native batched wait uses bounded polling deadlines and raises on overall timeout','Mock wait_all returns IN_PROGRESS; every timeout_ms<=10; fake clock>=0.05; optional native-enum guard'),
('test_step_barrier_noop_when_no_reads_posted',G,'Empty issued-read list skips completion wait','Recording wrapper called false'),
('test_write_mode_read_blocks_posts_no_read',G,'WRITE-mode read callback leaves read list and notification list empty','Nonempty source/destination metadata; direct callback'),
('test_producer_start_load_kv_posts_no_read',G,'Producer load callback requests asynchronous completion polling without a READ barrier','Empty transfer map; async_waited true, waited false, issuedreads empty'),
('test_requested_cudagraph_mode_is_never_overridden',r.E,'Connector reports no forced piecewise cudagraph override in READ or WRITE mode','Direct requires_piecewise_for_cudagraph false for read_mode true/false; no runtime graph/barrier guarantee')]:add(p,n,u,prop,cond)
p='tests/v1/kv_connector/unit/test_moriio_transfer_completion.py'
for n,u,prop,cond in [
('test_timeout_must_be_positive_and_finite',r.A,'Transfer timeout parser rejects zero, negative, infinity and NaN','Expected ValueError finite and greater than zero'),
('test_positive_finite_timeout_is_preserved',r.E,'Transfer timeout parser converts finite positive string to float','Input1.5 ->float1.5'),
('test_poll_transfer_batch_considers_every_status',G,'Batch completion verdict considers all statuses with failure precedence and empty batch DONE','Five lists including failed/pending before final success'),
('test_wait_raises_with_detail',G,'Transfer wait reports terminal failure detail or pending timeout','Fake failed SQfull or neverpending; timeout0.05'),
('test_wait_blocks_until_terminal',G,'Transfer barrier remains blocked until delayed success','Timer marksstatus at0.2seconds; elapsed>=0.15; source only no timer executed'),
('test_native_wait_receives_all_statuses_and_a_positive_deadline',G,'Native wait receives full status list and positive deadline bounded by overall timeout','Timeout0.05/0.004/0.0001 ->10/4/1ms; mock returnsSUCCESS'),
('test_native_wait_reports_failure_or_timeout',G,'Native wait results propagate status-specific failure, timeout or native error code','ERR_RDMA_OP/IN_PROGRESS/ERR_INVALID_ARGS mock result matrix'),
('test_wait_falls_back_without_native_status_enum',G,'Absent native status enum falls back to status polling and preserves failure detail','StatusCode None; failed+success; nativewait notcalled'),
('test_nonblocking_poll_does_not_call_native_wait',G,'Nonblocking pending-state poll and empty wait avoid native blocking API','Fake wait_all callcountzero'),
('test_native_wait_drains_only_the_unscoped_status_snapshot',G,'Unscoped completion wait passes pending snapshot once and clears global retained list','MockSUCCESS,exactstatuses and10ms deadline'),
('test_native_wait_returns_once_the_statuses_succeed',G,'Native IN_PROGRESS result does not prevent return once all individual statuses succeeded','Mock mutates late status on call1 or3; exactcallcount and10ms deadlines'),
('test_native_wait_after_a_failure_waits_for_in_flight_statuses_unless_fail_fast',G,'Failure waiting drains remaining in-flight statuses unless fail_fast is requested','ERR_RDMA_OP with failed+late; failfast1call and pending; otherwise3calls and late success; both raise SQfull')]:add(p,n,u,prop,cond+'; ROCm+mori module guard, mocked status/native surface')
p='tests/v1/kv_connector/unit/test_moriio_write_done_routing.py'
for n,u,prop,cond in [
('test_execute_write_task_ip_is_per_task_under_overwrite',B,'WRITE task completion address retains request-local host despite overwritten worker routing state','taskA host.1 versus workerhost.2'),
('test_deferred_task_ip_is_per_task_under_overwrite',B,'Deferred WRITE resolves task-local completion host despite worker routing overwrite','One deferredtask processed; completionhost.1 and queueclear'),
('test_two_requests_resolve_distinct_hosts',B,'Two WRITE requests retain their independent remote hosts','taskA.1,taskB.2 and workerhost.2; distinct completion addresses'),
('test_finalize_notify_port_uses_pod_local_rank',B,'WRITE completion resolves owning pod host and pod-local DP notify port','DP9,localDP8,twohosts ->secondhost and DP1port'),
('test_finalize_port_ignores_stale_worker_dp_local',B,'WRITE completion port uses task DP-local metadata rather than stale worker field','Tasklocal8 versus workerlocal1; DP9 portmatches9%8'),
('test_finalize_single_pod_port_uses_global_rank',B,'Single-pod WRITE completion uses original host and global DP/TP notification offset','DP3,TP2,emptyhostlist; expected portbase+offset(3,2)'),
('test_execute_write_task_writes_every_fanout_target',K,'WRITE task posts through each mapped consumer TP session and retains corresponding statuses','MLA producer1/TP2 ->consumerTP4 ranks2,3; fake plan/write helpers; no data equality'),
('test_fanout_write_done_notifies_every_target_after_completion',G,'Fan-out WRITE waits all target statuses after sealed layer completion before notifying every target','Before writes_done no waits/notifications; after1 exacttwo-statuswait and2notifyports'),
('test_write_blocks_for_req_forwards_routing_to_schedule_write',B,'Request WRITE scheduling receives request-local host list, DP size and source/destination block IDs','Recorded schedulekwargs; doesnotinstall mutable worker.multi_pod_hosts'),
('test_write_blocks_for_req_falls_back_when_multi_pod_hosts_empty',B,'Empty host list scheduling falls back to request remote_host and remote_dp_size','Captured host.9 and DP4; source/destination blocks preserved'),
('test_write_blocks_for_req_schedules_mla_write_fanout',K,'Request scheduling forwards MLA fan-out TP mapping and consumer TP size','Producer1/TP2 consumerTP4 ->targetranks2,3'),
('test_fanout_write_retries_handshake_after_one_target_failed',G,'Missing one fan-out session triggers handshake retry and suppresses write','Ready flagtrue but only TP0 remoteagent; TP1missing'),
('test_fanout_write_proceeds_once_every_target_session_exists',G,'Fan-out WRITE proceeds only once every target session exists','BothTP0/1 remoteagents; nohandshake and one recordedwrite'),
('test_rank_with_nothing_to_write_does_not_wait_for_a_handshake',G,'Replicated MLA rank with no target skips handshake wait while invoking normal write scheduling path','Producer1/TP2 consumerTP1; noagents; elapsed<5 seconds and recorded write req; no RDMA claim'),
('test_release_reaches_every_prefill_rank_and_no_other',G,'WRITE release notification targets exactly declared producer TP ranks with request transfer ID','remote_tp_size2 overrides,legacytp_size8,defaultlocal4; exact port list'),
('test_deferred_write_timeout_fails_transfer_on_every_decode_rank',G,'Expired deferred WRITE notifies possible decode owners, completes producer and rejects late allocation resurrection','Two pods with2DP each; stale enqueue; terminal transfer allocation remainsempty'),
('test_deferred_write_timeout_notifies_every_fanout_target',G,'Expired fan-out WRITE broadcasts failure to all candidate DP and mapped TP targets','DP0/1 xTP2/3; exact four notifications'),
('test_write_scheduler_waits_for_worker_ack_past_defer_deadline',G,'Scheduler retains expired WRITE deadline until actual worker completion ACK arrives','First empty output suppressed toNone; second finishedreq clearsdeadline'),
('test_write_failed_is_reported_as_failed_recving_on_owning_rank',G,'WRITE failure resolves only locally owned transfer and retains unmatched other-rank failure','tA->req1 versus tOtherRank; finished+failed req1, no successful writecompletion'),
('test_write_rejects_unmappable_tp_before_scheduling_any_layer',K,'Unmappable WRITE TP topology rejects before scheduling any attention layer','LocalTP2 remoteTP3 hybridworker; ValueError and schedule_notcalled'),
('test_remote_blocks_message_normalizes_to_grouped_block_ids',K,'Remote allocation message preserves grouped block IDs including empty attention group and wraps legacy flat list','Three explicitflat/hybridpayloads; decodeDP2 preserved'),
('test_empty_layer_plan_posts_no_rdma_write',K,'Empty layer transfer plan returns no statuses and posts no vector or single RDMA write','All offset/size lists empty; both mock methods notcalled'),
('test_hybrid_write_requires_exactly_one_running_state_slot',K,'Hybrid WRITE plan rejects missing or multiple local running-state slots','Localstate[] or[1,2]; remoteoneslot9; MoRIIOError')]:add(p,n,u,prop,cond+'; CPU mocked worker/wrapper; no network execution')
(r.HERE/'reviewed-decisions.json').write_text(json.dumps(r.ledger,ensure_ascii=False,indent=2)+'\n')
summary={'new_entries':len(r.ledger['decisions'])-start,'total_decisions':len(r.ledger['decisions']),'pending':['test_heterogeneous_tp_head_guard_allows_supported_layouts','test_heterogeneous_tp_head_guard_allows_mla_layouts','test_wait_returns_immediately_when_nothing_pending'],'pending_reason':'These three are successful return/no-exception smoke checks without direct assertions; not promoted to the broader placement/completion property.','execution':'Source review only; no worker, subprocess, GPU, network or timer execution'}
(r.HERE/'batch4-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(summary)
