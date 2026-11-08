# vLLM 请求结束后的 KV 资源释放审查

本批完整审查固定 vLLM 快照中的 [`tests/v1/core/test_deferred_block_free.py`](../third_party/reference_repos/vllm-20261105/tests/v1/core/test_deferred_block_free.py) 的 17 个测试入口；在现有主索引中，这 17 项审查前均为 `not_reviewed`。逐项结果见 [`agent_vllm_review_5.tsv`](agent_vllm_review_5.tsv)，每行分别记录目标调用、断言源码行、输入与状态条件以及局部验证性质。`merge_agent_reviews.py` 已验证全部 17 项的入口键和固定快照行号。

这些测试围绕一个具体性质展开：请求结束或抢占后，KV block 应在**最后一个可能读写它的在途步骤完成前不可重用**，之后能够回到空闲池。vLLM 的调度器在重叠批次与消费端 KV connector 同时存在时启用延迟释放；请求的 `last_sched_seq`、已处理的 `processed_step_seq` 和 `deferred_frees` 控制释放时间。对应实现位于 [`scheduler.py:169`](../third_party/reference_repos/vllm-20261105/vllm/v1/core/sched/scheduler.py#L169)、[`scheduler.py:3241`](../third_party/reference_repos/vllm-20261105/vllm/v1/core/sched/scheduler.py#L3241) 和 [`scheduler.py:3269`](../third_party/reference_repos/vllm-20261105/vllm/v1/core/sched/scheduler.py#L3269)。测试分别改变终止原因（停止 token、中止、抢占、达到 token 上限）、在途步骤数、分段 Prefill、分配失败、CoW 拷贝及异步 KV load；断言检查请求状态、待释放队列、block 引用计数、空闲池数量、后续请求调度和 connector mock 的调用时序。

两个边界尤其影响空间建模。第一，同一时刻系统还有在途步骤，不等于**这个请求的** block 仍可能被写入：短请求达到 `max_tokens` 后，后续步骤只调度另一请求，测试检查短请求立即释放（测试 `:378–412`）。第二，分段 Prefill 和非异步全量 Prefill 的在途标记不同：前者检查 `is_prefill_chunk`，后者检查 `last_sched_seq`，两者都在中止后保留 block 至对应步骤完成（测试 `:415–486`）。因此，验证条件应记录请求终止原因、该请求是否被后续步骤调度、最后调度与完成的步骤序号、KV 所有权和在途复制/加载，而不只记录“是否调用 abort”。

与 SGLang 的可共享抽象是**请求终止时的资源生命周期**：运行中或分段 Prefill 中止时，应先使请求停止继续执行，并在已有工作不再引用其状态后完成 KV 释放。SGLang 固定快照在 [`scheduler.py:3774`](../third_party/reference_repos/sglang-v0.5.14-complete/python/sglang/srt/managers/scheduler.py#L3774) 区分等待队列、分离式传输队列及运行中请求的中止处理；[`scheduler.py:2466`](../third_party/reference_repos/sglang-v0.5.14-complete/python/sglang/srt/managers/scheduler.py#L2466) 对分段 Prefill 的中止先保留待处理标记，待安全时清理并释放 KV。这里的共同义务是“停止后无过早复用、最终可释放”；vLLM 使用步骤 fence 和 `deferred_frees`，SGLang 使用队列/批次状态与延后处理，二者的内部状态变量不能直接合并成同一个检查项。若要在独立验证仓库实现这一义务，需要按各框架适配终止事件、在途工作边界和资源可重用观察点。

本批证据是 CPU 测试的源码级检查。部分测试强制打开延迟释放开关或模拟深度 3 的在途步骤；异步 load 测试使用 connector mock。它们明确验证调度器状态和调用顺序，尚不直接验证真实 GPU 写入、NIC/RDMA 并发或跨框架运行等价。独立测试设计可以沿用上述抽象性质，但应将真实设备并发列为另外的执行条件。
