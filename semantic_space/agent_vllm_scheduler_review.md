# vLLM Scheduler 源码审查

固定源码为 vLLM `76a0ccc209343971c6f4b56f6380a76202428385`，范围是 [`tests/v1/core/test_scheduler.py`](../third_party/reference_repos/vllm-20261105/tests/v1/core/test_scheduler.py)。逐项核对该文件的 182 个顶层 `test_` 入口，记录输入和运行状态、实际产品调用、断言行与检查性质；对应实现主要位于 [`vllm/v1/core/sched/scheduler.py`](../third_party/reference_repos/vllm-20261105/vllm/v1/core/sched/scheduler.py)，另涉及 Diffusion 调度器与 EngineCore。明细见 [`agent_vllm_scheduler_review.tsv`](agent_vllm_scheduler_review.tsv)，可复核的结构化材料见 [`agent_vllm_scheduler_review.json`](agent_vllm_scheduler_review.json)。这些是源码级结论，参数化实例是否被具体 CI 收集、是否实际运行，需另用运行证据确定。

审查结果为 39 项 `mapped`、140 项 `needs_refinement`、3 项 `outside_request_verification`，没有遗漏入口。`mapped` 表示现有验证单元能够表达断言的主要性质；`needs_refinement` 表示测试的条件或判定比现有单元更具体，不能只按宽泛的“调度进度”收纳；3 项范围外入口分别检查配置字段结构、默认构造是否存在 EC connector，以及测试文件自己的 `_skip_drafts_per_step` 辅助函数。

最密集的新增空间要求集中在五组。第一，**请求准入策略**要区分优先级及到达时间、缓存感知排序的候选窗口和占用阈值、Prefill 节流间隔，以及阻塞请求的重排；例如 `test_priority_scheduling_arrival_time_tiebreaker` 检查的是同优先级时的到达顺序，`test_cache_aware_*` 检查的是暖前缀排序条件，不能合成一个“请求能继续执行”的性质。第二，**编码器输入与缓存**要表达媒体项边界、重复项计算一次、标识符对应 embedding 数、容量不足、抢占后保留或重算，以及 EC connector 的异步可用状态；`test_encoder_cache_rejects_mismatched_embed_count` 还同时检查拒绝、错误状态和异常信息。第三，**分层 KV 与远端转移**需要记录各 KV group 的共同可用前缀、外部加载状态及资源释放顺序；混合 FA/Mamba 前缀命中测试明确比较各组命中长度及最终报告的已计算 token。第四，**推测解码和 Diffusion**需要分开表示 draft 预算、Prefill 阶段禁用 draft、异步占位、每请求 canvas 宽度和 read 的延迟调度，后者由 [`diffusion_scheduler.py`](../third_party/reference_repos/vllm-20261105/vllm/v1/core/sched/diffusion_scheduler.py) 实现。第五，**诊断统计**本身构成检查对象，包含命中数量及来源、排队和跳过数量、连接器两侧统计的合并、逐步草稿接受情况；这些断言验证观测值，不直接证明模型输出正确。

审查也揭示了条件与性质应分别记录。`test_schedule_order` 在同一输入池下切换 chunked prefill，断言本轮准入长请求的数量；`test_kv_connector_honors_skip_reading_prefix_cache` 把 prompt-logprob scoring 与普通请求放在一起，要求前者跳过外部 KV 读取但仍允许本地前缀复用；`test_hybrid_fa_deeper_hit_respects_connector_lookup_policy` 随 connector 是否支持组间不一致命中改变已计算 token 的可接受边界。这些测试都可能调用 `Scheduler.schedule()`，但输入状态和预期性质不同，不能凭调用同一函数就视作同一验证单元。

校验命令为：

```bash
python3 experiments/semantic_space/agent_vllm_scheduler_review_build.py
python3 experiments/semantic_space/audit_source_review_batch.py \
  --review experiments/semantic_space/agent_vllm_scheduler_review.json \
  --snapshot vllm \
  --path tests/v1/core/test_scheduler.py \
  --output experiments/semantic_space/agent_vllm_scheduler_review.tsv
```

校验确认 182 项均有记录，所填断言行来自固定测试源码，现有单元 ID 与固定空间定义一致，引用的固定源码路径和行号存在。下一步合并空间时，应先抽象上述可跨框架表达的义务，再将 vLLM 的具体队列、连接器和模型执行模式作为适用条件或实现差异保存；对 SGLang 须以其自身源码与断言独立核对对应关系。
