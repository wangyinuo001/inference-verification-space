# 继承公共方法的第二轮扩展

固定源码仍为 vLLM `76a0ccc` 与 SGLang `49e384c`。本轮回读 SGLang 公共测试方法及被调用的断言辅助函数，为 214 个继承入口增加带源码位置、具体判定和适用条件的交叉映射。逐项内容在 [inherited-property-crosswalk.tsv](inherited-property-crosswalk.tsv)；两页均已使用该表更新。绿色只表示源码中的局部检查得到确认，不表示实际运行或完整覆盖对应空间单元。

| 方法组 | 新增入口 | 已核对的局部检查 |
|---|---:|---|
| [Unified Radix](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kits/unified_radix_cache_kit.py#L18) | 36 | GSM8K/MMLU 门槛，三轮 logprob 重算关系，分支前缀或 Decode 命中数 |
| [Reasoning 用量](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kits/reasoning_kit.py#L10) | 20 | 思考开关下的 reasoning_tokens 范围、流式 usage 存在、原生接口报告数与思考结束 ID 位置相等 |
| [确定性辅助测试](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/test_deterministic_utils.py#L51) | 16 | 单提示词或四种前缀长度在多种批大小下的输出文本唯一数为一 |
| [流式会话](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kits/streaming_session_kit.py#L24) | 40 | 会话继承的 cached_tokens、首轮或后续轮取消后的状态与 prompt_tokens 恢复 |
| [推测解码公共方法](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kits/spec_server_kits.py#L32)及[单请求速度](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kits/spec_decoding_kit.py#L12) | 102 | JSON 可解析性、logprob 数量或重算差值、短请求响应字段、零温度非推测文本对照、吞吐和接受长度门槛 |

合计 214 条均为此前未标绿的入口，进度页由 1,679 增至 1,893。离单轮净增约一千条的目标仍差 786 条；不能把尚未核对的静态调用—断言关系直接改成绿色。

本轮特别保留以下边界：

- `test_multiturn_logprobs_match` 只在继承类提供 `decode_cache_assert` 时检查实际缓存命中；否则它是生成与重算的 logprob 关系，不能称为缓存命中结果等价。
- `test_single` 在调用 `test_deterministic` **之后**才设 `temperature=0.5`；该赋值不影响本次检查。`test_prefix_with_logprobs` 的辅助函数打印 logprob 比较结果，外层只断言输出文本唯一数。
- 推测解码的 `test_first_token_finish`、`test_batch_generation` 和 ragged logprob 测试主要检查实际返回项含 `text`；部分方法未断言返回项数量，空列表可绕过逐项检查。JSON 方法不检查完整 Schema；logprob 重算采用服务自身 prefill，而非独立数值真值。
- 流式会话的并发压力辅助函数主要检查 HTTP 状态和后续健康状态，没有直接证明无 KV 泄漏或请求间隔离。本轮未把这两组方法映射成所有权或隔离性质。

这些源码审查结果仍对照早期候选空间。76 单元的源码派生模型需要按其输入、状态、判定边界和组合缺口重新校准，不能自动继承此处的绿色数量。
