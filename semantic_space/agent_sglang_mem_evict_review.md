# SGLang 缓存驱逐策略测试源码审查

固定快照 `sglang-v0.5.14-complete` 中，`test/registered/unit/mem_cache/test_evict_policy.py` 有 23 个测试入口。本批逐项核对 `_make_node`、各策略的 `get_priority` 实现及断言。审查 JSON 和 TSV 均覆盖全部 23 项；`audit_source_review_batch.py` 校验通过，结果为 21 项 `mapped`、2 项 `needs_refinement`。这里的结果是源码方法级审查，不代表重新运行测试。

测试使用 `_make_node` 创建带访问时间、命中次数、创建时间或显式优先级的 `MagicMock` 节点，然后检查比较键或由比较键形成的顺序。LRU 与 FIFO 分别按访问时间和创建时间升序；MRU 与 FILO 使用对应时间的负值；LFU 先按命中次数，再按访问时间；显式优先级策略先按 `priority`，再按访问时间。SLRU 把命中次数与阈值比较，产生试用段或保护段的标记，同段再按访问时间排序。两项排序测试把多个节点交给 Python `sorted(key=strategy.get_priority)`，检查完整顺序。这些断言可落入现有 `cache.eviction_priority` 单元，其具体输入和断言位置逐项记录在 TSV。

`test_custom_threshold` 同时检查阈值 5 两侧的段标记；现有单元的 `conditions=[]` 尚未表达 `protected_threshold` 及 `hit_count` 相对于阈值的边界关系，因此标为 `needs_refinement`。`test_default_threshold_is_2` 直接检查无参构造后的默认配置值，不是对节点计算比较键；建议另设 `cache.eviction_policy_configuration` 单元，记录策略名、默认值与显式配置。其余 SLRU 测试在 `applicability_note` 保留了阈值 2 和节点状态，即使映射到已有单元，后续按空间条件剪枝时仍需显式加入这些条件。

这些单元测试没有调用 `RadixCache.evict`。实际驱逐在 `python/sglang/srt/mem_cache/radix_cache.py:558-585` 先取可驱逐叶子、用策略比较键建堆，再弹出节点并释放 KV 资源。因而局部排序键正确与实际释放顺序、锁定节点保护及释放数量是不同的检查。跨框架可抽象的性质是“对当前可驱逐缓存项，按照声明的策略选择驱逐顺序”。SGLang 在 `evict_policy.py:16-65` 提供多个可配置排序键；vLLM 的 `vllm/v1/core/kv_cache_utils.py:262-278` 通过空闲块队列维护 LRU 次序，并在同一访问时间下优先考虑链尾块。两者的具体策略和状态单位不同，应作为空间条件保留，不能把 SGLang 的七种策略测试直接当作 vLLM 队列语义的验证。
