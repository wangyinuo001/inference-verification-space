## 启动前配置：vLLM 与 SGLang 源码对照

| # | 共同的验证需求 | vLLM（固定快照） | SGLang 0.5.14（固定快照） | 需要保留的差异 |
|---|---|---|---|---|
| 1 | 显式配置和默认值合成生效配置，显式值优先 | `vllm/config/vllm.py:1120-1147` 的 `_apply_optimization_level_defaults` 递归填入未指定的默认值；`1997-2000` 根据优化级别选择默认表 | `python/sglang/srt/server_args.py:2456-2483` 的 `__post_init__` 依序处理默认、校验和旧参数别名 | 默认策略及字段不同，比较的是各自声明规则下的生效配置，不是两个框架字段逐项相等 |
| 2 | 某些默认值由运行模式决定 | `vllm/config/vllm.py:1997-2000` 根据优化级别设置编译和内核选项 | `python/sglang/srt/server_args.py:2630-2647` 将 `load_balance_method=auto` 按 PD 模式定为 `follow_bootstrap_room` 或 `round_robin` | 模式条件应作为验证输入；不能只记录命令行原始文本 |
| 3 | 模型与硬件能力不满足时在启动前拒绝 | `vllm/config/vllm.py:2027-2045` 检查 HiSparse 的平台和 PP/CP 组合 | `python/sglang/srt/server_args.py:6630-6679` 检查 TP/PP/节点数、图填充与编译、流水与推测解码等组合 | 支持矩阵依框架版本和平台变化，不应抽象为同一组固定合法配置 |
| 4 | 并行配置与模型形状相容 | `vllm/config/model.py:1559-1578` 检查注意力头数可被 TP 大小整除 | `python/sglang/srt/server_args.py:6630-6642` 检查 TP×PP 与节点数的关系 | 两者都检查并行合法性，但所引用的模型参数和约束不同 |

原有测试也覆盖这两种性质：vLLM `tests/test_config.py:3685-3718` 比较默认配置、`815-824` 检查不支持的平台；SGLang `test/registered/unit/server_args/test_server_args.py:34-82` 检查配置合并及 PD 模式默认值，`85-132` 检查不兼容后端的拒绝。这些测试是空间构建的证据，尚未直接映射为新模型中的已核验测试入口。
