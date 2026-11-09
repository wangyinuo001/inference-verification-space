# vLLM offloading 事件测试的源码审查

本批完整回读固定快照的 `tests/v1/kv_connector/unit/offloading_connector/test_events.py`，覆盖其中 33 个测试入口，并核对文件内的 `_tracker`、`_group_config`、`_record_chunks` 等辅助函数与产品实现 `vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py`。逐项审查记录在 `agent_vllm_offloading_events_review.json`，经 `audit_source_review_batch.py` 校验后生成 `agent_vllm_offloading_events_audit.tsv`。33 项均有可定位的源码断言，其中 2 项的局部性质可以归入现有 `cache.lifetime` 或 `cache.state_invariants` 单元，31 项标为 `needs_refinement`。状态描述的是源码中的检查及其适用条件，未推断这些测试已在特定运行配置执行。

这份测试检查的核心不是跨层 KV **数据内容**是否相等，而是缓存状态变化能否向外提供正确的**事件记录**。`OffloadingEventsTracker` 将原始 store/remove 通知与请求快照、KV 组规格和驻留状态结合，输出 `BlockStored`、`BlockRemoved`。测试检查哈希及前驱关系、token 与 lookahead 边界、block 大小、组号、存储介质、locality 和 ownership；同时检查关闭自描述模式、滑动窗口组、Mamba 状态 checkpoint 时的占位或缩减表达。另一些测试在主层移除、次层仍驻留、pending 写入失败、同键重试和 reset 后核对元数据何时保留与清除。最后三项测试检查分层缓存 backpressure 配置的字段继承及覆盖关系。对这些性质，现有 `cache.tier_transfer_content` 聚焦的是迁移前后真实 KV 内容，不能代替事件记录的核对。

建议在现有缓存层次下细化四类验证义务：一是**事件载荷对应关系**，将 store/remove 事件与源请求、块哈希链、token 区间和存储层关联；二是**事件元数据生命周期**，检查多存储层驻留、失败重试和跨批次先移除后存储时，元数据不被过早清除，也不在最后移除后残留；三是**组语义与事件配置**，按 full attention、Mamba、滑动窗口、部分尾块和自描述开关定义事件内容；四是**存储层策略配置**，检查 backpressure 的顶层默认值和 tier 覆盖规则。这些是从本文件断言提出的待细化单元，不等同于已证明的框架通用分类。

SGLang 固定快照也生成 `BlockStored`、`BlockRemoved` 和 `AllBlocksCleared`：`python/sglang/srt/mem_cache/events.py:34–122` 从 radix 节点按 page 生成事件，`test/registered/unit/mem_cache/test_radix_cache_unit.py:490–516` 检查插入和淘汰后的事件类型及移除哈希。两者可共同抽象为“缓存状态转换对应可解释的存储、移除或清空事件”，并用哈希链、token 范围、存储介质及事件顺序约束消费者所见状态。vLLM 的原始 offloading 事件转换、Mamba checkpoint、跨层 ownership 和 pending 元数据，以及 SGLang 的 page/radix 节点与 `AllBlocksCleared`，应作为各自的条件或扩展；源码相似不能直接证明跨框架载荷完全等价。
