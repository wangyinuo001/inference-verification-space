# SGLang 运行入口源码审查

审查 218 条运行观察到的候选源码入口；其中 100 条有源码断言支持的局部性质映射，118 条不强行映射。运行记录只用于选取审查对象，不作为性质证据。

## 按单元统计

- `needs_refinement_or_unconfirmed`：118
- `cache.lifetime`：21
- `cache.state_invariants`：15
- `cache.eviction_outcome`：15
- `sampling.per_request_modifier_binding`：12
- `compute.operator_reference_relation`：10
- `cache.tier_transfer_content`：6
- `cache.hit_accounting`：6
- `cache.eviction_policy_configuration`：4
- `abort.cleanup`：3
- `cache.eviction_candidate_order`：2
- `cache.event_payload_binding`：2
- `spec.draft_candidate_binding`：2
- `cache.hash_bigram_equivalence`：1
- `cache.namespace`：1

## 空间缺口和候选误归类

- `test_penaltylib.py` 的累积频率、presence 和最小新 token 惩罚检查是 logits 变换，现有 `sampling.policy` 描述最终采样 token；应有独立的采样前 logits/penalty 变换单元。
- `test_hicache_staged_write_back_dispatch.py` 的部分测试只断言 staged/fallback 分派及 host index 所在设备，没有比较跨层缓存内容；需要独立的迁移路径选择单元。
- `test_ngram_corpus.py` 的多语料增删、重复 ID、错误加载保留已有语料，是草稿语料生命周期/配置管理，不是草稿 token 与历史绑定。
- `test_reasoning_parser.py` 的构造器默认值不证明输出通道解析；`test_mem_cache_utils.py` 的哈希格式和拆页也不证明请求间隔离。
- `test/registered/unit/constrained/test_grammar_manager.py` 中语法编译对象的缓存，不是 KV 缓存；候选映射到 `cache.state_invariants` 是领域错位。

## 证据与限制

每条映射保留固定快照文件哈希、测试函数体哈希、断言行和断言文本。TSV 的 `property` 限定为测试实际检查的局部关系。模拟存储和 mocked CUDA transfer 的结论仅适用于模拟路径；本轮未据此推断真机传输或端到端结果。
