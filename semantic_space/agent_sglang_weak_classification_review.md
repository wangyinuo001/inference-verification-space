# SGLang 弱证据归类抽样核查

从固定快照的 `test-entry-classification.tsv` 中，仅取 `basis=test_name`、`file_path`、`component_test_directory` 的 SGLang 源码入口，排除只有文件名而无测试定义的记录。使用固定随机种子 20261109，按三种依据分别抽取 30、20、30 条，并在每组内轮流覆盖原有 `primary_area`。逐条回读测试体及必要的 helper，保留原标签，记录候选修正领域、断言或调用源码行和修正规则。完整 80 条见 [核查表](agent_sglang_weak_classification_review.tsv)。这里的候选领域是分诊结果，具体验证性质仍需调用、输入与断言证据确认。

| 原归类依据 | 抽样数 | 原领域与源码主对象大体一致 | 原领域偏离主对象 | 涉及多个领域 |
| --- | ---: | ---: | ---: | ---: |
| 测试名 | 30 | 20 | 8 | 2 |
| 文件路径 | 20 | 11 | 8 | 1 |
| 组件测试目录 | 30 | 15 | 7 | 8 |
| 合计 | 80 | 46 | 23 | 11 |

较明确的误归类来自名称或目录的宽泛含义。`test/registered/cpu/test_topk.py:212–237` 比较 Llama4 MoE 路由函数与 fused topk 算子输出，原标签 `sampling` 会误指向生成采样；`test/registered/unit/server_args/test_server_args.py:308–329` 检查弃用的 Context Parallel 参数如何映射到配置字段，原标签 `distributed` 会误指向分布式执行结果；`test/registered/rl/test_fp32_lm_head.py:52–97` 检查 LM head 运算前的 dtype 转换，原标签 `training` 不能由 RL 目录证明。类似地，`test/registered/unit/function_call/test_function_call_parser.py:2093–2127` 检查的是工具调用 grammar/structural tag 的构造，而不是模型输出解析。

另一些入口确实跨领域。`test/manual/models/test_qwen3_asr.py:558–651` 同时检查 WebSocket 会话前后 item ID 关系和第二段转写 WER；`test/manual/chunked_prefill/test_scripted_kv_pressure.py:433–471` 同时检查 KV 压力下的请求调度和资源回收。将它们压成单一 `model` 或 `scheduling` 标签会丢掉另一项检查。对于 `test/registered/prefill_only/test_multi_item_scoring.py:401–404` 这类仅委托 helper 的入口，还需展开 `_compare_scores`（350–377 行）才能看到批量与逐项评分的近似对照。`test/manual/ep/test_deepep_intranode.py:23` 是带外部参数的手动分布式测试函数，源码入口不应直接等同于可独立收集的 pytest 用例。

全量快速扫描可先改三处规则：第一，测试名和目录只用于候选召回；主领域由被测调用及断言目标决定，例如区分 MoE `topk` 与解码采样。第二，遇到辅助函数、fixture 或跨对象状态时展开一层调用并保留未解析状态，避免把“测试体没有断言”当成“没有检查”。第三，为一个测试保留组件、请求阶段和被检查性质等多个标签，使跨领域检查可检索。以上规则针对本次样本发现的错误类型；样本按领域轮流抽取，表中的 46/80 不应外推成整个 SGLang 测试库的准确率。
