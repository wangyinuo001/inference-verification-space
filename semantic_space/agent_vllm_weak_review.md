# vLLM 弱证据归类留出审查

从 [`test-entry-classification.tsv`](test-entry-classification.tsv) 中仅选 vLLM 的 Python 测试入口，按 `test_name`、`file_path`、`component_test_directory` 三种弱依据分层，固定种子 `20261109`，分别抽取 24、16、40 条。各层按原 `primary_area` 轮转并优先选不同源码文件，共得到 80 条、79 个文件。逐条读取固定 vLLM 源码 `76a0ccc209343971c6f4b56f6380a76202428385` 的测试体、断言和必要的 helper，记录在 [`agent_vllm_weak_review.tsv`](agent_vllm_weak_review.tsv)。可用 [`agent_vllm_weak_review_build.py`](agent_vllm_weak_review_build.py) 复现抽样与核查表。

按源码中的**主要检查对象**判断，44 条原一级领域基本吻合，11 条只描述执行条件或部分性质，25 条会把主要检查放到另一领域。三种依据分别为：测试名 9/6/9，文件路径 9/1/6，组件目录 26/4/10（顺序为吻合/部分/错位）。这是有意覆盖不同领域的诊断抽样，不能将 25/80 作为全仓库误归类率；吻合也只表示一级领域合理，不等于具体验证性质已经确认。

误差有清楚的来源。`test_scheduler_config_defaults_are_none` 位于调度测试并含 scheduler 名称，但只断言队列配置默认值（`tests/v1/engine/test_admission_control.py:543`）；`test_draft_lookahead_raises_on_interior_miss` 名称中的 draft 触发 sampling 候选，实际在第 224 行检查 encoder cache miss；`test_fp32_head_runs_projection_in_fp32` 位于 sample 目录，却在 `tests/v1/sample/test_head_dtype.py:64` 对照 FP32 logits 数值。`tests/entrypoints/rl/` 下的 `test_clear_cache_preserves_output_and_controls_prefix_cache`（第 173 行）检查缓存清理与输出保持，`test_rejected_request_is_not_recorded`（第 154 行）检查拒绝操作不记时长；把路径统一归为训练性质会丢掉两者差异。`test_distributed` 使用 TP=2 运行 GGUF，但它调用的 `check_model_outputs` 在 `tests/plugins_tests/gguf/test_gguf_plugin_generate.py:87` 对照 GGUF 与原模型输出和 logprob，分布式是运行条件，主要判定是模型结果关系。

可在批量扫描中加入以下保守修正规则：

1. **先读断言的数据来源，再使用名称和目录。** 当断言可追到具体产品调用时，用被检查的字段、异常或结果关系生成候选领域；仅凭 `draft`、`scheduler`、`model` 等词不确定主要性质。例如第 224 行的 encoder cache 异常、第 543 行的 config 默认值、第 64 行的 logits 参考比较。
2. **区分运行条件与检查对象。** TP、GPU、Diffusion、RL 或 speculative 配置可作为条件字段；如果断言检查的是模型输出、缓存状态、配置接受性或指标记录，应优先保留该对象。GGUF 的 TP=2、Diffusion read 的 eligible step，以及 RL 目录中的缓存与指标案例属于此类。
3. **对无直接断言的入口追一层本地 helper。** `test_distributed` 在第 130 行调用 `check_model_outputs`，后者完成输出比较；`test_no_runtime_jit` 在第 140 行调用 `can_run_without_jit`，由 helper 启动 shape battery 并在运行期 JIT 时失败。只看测试入口名称或直接 `assert` 数量会失去判定依据。
4. **识别测试工具与实际请求路径。** `tests/benchmarks/test_random_multimodal_dataset_video.py:50` 检查的是 benchmark 数据生成器对 image/video 的分类，不是推理服务输入绑定；`tests/benchmarks/test_mm_processor_cli.py:73` 检查 benchmark 预处理统计，不是服务吞吐。此类目录应保留“测试工具/基准工具”标记，不能直接当作产品行为覆盖。
5. **保留多领域或未决状态。** 远端 KV 命中按块对齐同时涉及 transfer 与 cache；Quick AllReduce 环境变量拒绝检查的是分布式组件的配置条件。源证据只支持局部属性时输出多个候选及其依据，暂不强迫唯一领域或宣称性质已确认。

这批审查只改进全量快速扫描的候选优先级，不修改共享索引和 Verification Space。下一步若批量应用，应先对保留下来的弱依据候选标注 `candidate`，让断言和调用证据决定能否提升为更具体的验证义务。
