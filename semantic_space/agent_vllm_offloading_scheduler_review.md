# vLLM OffloadingConnector 调度测试的源码核查

固定 vLLM 快照 `tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py` 实际包含 162 个源码级测试定义，全部已列入逐项台账。每条记录保存测试条件、产品或辅助函数调用、直接断言表达式与行号、产品源码位置及候选验证单元。`agent_vllm_offloading_scheduler_review.json` 保存完整证据，`agent_vllm_offloading_scheduler_detail.tsv` 便于逐项筛查；`agent_vllm_offloading_scheduler_review.tsv` 已由 `audit_source_review_batch.py` 校验。源码中共有 493 处直接断言；另外 5 项没有测试体内的直接断言，但调用 `RequestRunner.run`，其 `expected_loaded`、`expected_stored`、`expected_flushed` 在 `tests/v1/kv_connector/unit/offloading_connector/utils.py` 第 703、714、717 行与实际块集合比较。

这些测试不能只按“缓存正确”归为一个单元。调度决定包括命中前缀与滑窗边界、缺块时的延迟、每步加载预算、哪些块可存及 EAGLE lookahead 边界；状态生命周期包括在途 store/load、worker 确认、失败、取消、请求结束和 reset fence；跨层数据或事件包括 offload key 与组号、GPU 块源目标、命中来源、connector metadata、事件及指标。三类会在同一测试中并存，例如一次 store/load 回合既检查可加载块，又涉及请求结束后 store job 的完成顺序。台账的多标签计数为调度决定 149、状态生命周期 86、跨层数据或事件 79；它们可重叠，不是互斥分区。

逐项记录中有 91 项可在测试体中定位到 `OffloadingConnectorScheduler` 或 `RequestOffloadState` 的直接方法调用；其余 71 项主要经 `request_runner`、局部辅助函数或更高层的 scheduler 流程进入产品代码。对这 71 项只给出相关产品方法的**待核查入口**，明确标为 `topic_candidate_not_observed_call`，未把方法名相似当作已证实的调用路径。五项辅助断言也另标 `delegated_helper_oracle`，避免被误算成“无 oracle”。现有空间中的 `cache.safe_reuse`、`cache.lifetime` 等单元描述了较宽的目标，但未表达本文件断言的具体 offloading 决策、任务确认与块映射，因此 162 项均保留 `needs_refinement`，不把源码测试名直接等同于已完成的跨框架验证性质。

从源码证据可进一步区分至少八类候选性质：命中和加载边界、可存块选择、传输键与块映射、任务生命周期、事件与指标、命中来源、资源预算，以及完整 store/load 回合。台账对应建议名称分别为 `cache.offload_lookup_decision`（54 项）、`cache.offload_store_decision`（15 项）、`cache.offload_transfer_mapping`（16 项）、`cache.offload_job_lifecycle`（48 项）、`cache.offload_event_accounting`（9 项）、`cache.offload_hit_source_mapping`（5 项）、`schedule.offload_resource_budget`（9 项）和 `cache.offload_roundtrip_relation`（6 项）。这些是本批源码审阅的组织建议，尚未写入共享空间模型，也不意味着测试之间可替代。

几个例子说明边界。`test_sync_load_budget_is_shared_per_step` 通过同一步两次请求检查同步加载是否共用预算；`test_complete_store_waits_for_all_worker_acks` 检查一个 store job 在所有 worker 回执到达前后的状态；`test_external_cache_hit_sources_preserve_partial_tail_origin` 检查分组与 token 区间对应的命中来源；`TestMambaHybridOffloadServing::test_shared_group_mtp_serves_like_non_speculative` 比较 store/finish/lookup 后可服务的 token 数，而非模型生成文本。这些断言分别要求预算、任务确认、来源区间与回合边界的条件，不能互相替代为单一的缓存命中检查。

复核命令：

```sh
experiments/.venv/bin/python experiments/semantic_space/agent_vllm_offloading_scheduler_review.py
experiments/.venv/bin/python experiments/semantic_space/audit_source_review_batch.py --review experiments/semantic_space/agent_vllm_offloading_scheduler_review.json --snapshot vllm --path tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py --suite '*' --output experiments/semantic_space/agent_vllm_offloading_scheduler_review.tsv
experiments/.venv/bin/python -m unittest discover -s experiments/semantic_space/tests -p test_agent_vllm_offloading_scheduler_review.py -v
```

校验结果为 162/162 条已记录，4 项独立工具测试通过。当前核查依据为固定测试和产品源码，没有运行 GPU，也没有把静态路径候选或测试断言升级为运行结论。
