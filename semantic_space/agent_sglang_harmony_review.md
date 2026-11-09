# SGLang Harmony parser 源码逐项核查

固定源码为 SGLang 0.5.14 的 `test/registered/unit/parser/test_harmony_parser.py` 与 `python/sglang/srt/parser/harmony_parser.py`。审阅了文件内全部 73 个测试方法，记录每项测试的输入线索、直接产品调用、240 处显式断言、实际检查内容及对应产品实现位置。逐项证据在 `agent_sglang_harmony_review.json` 和 `agent_sglang_harmony_detail.tsv`；`agent_sglang_harmony_review.tsv` 已通过 `audit_source_review_batch.py` 对方法名、断言行、单元 ID 与源码位置的校验。这是源码核查，不推断测试是否在某次 CI 中收集或执行。

这批测试覆盖的是内部 Harmony 文本到事件的解析关系，不能直接归入现有的 HTTP 响应交付、工具调用实参解析或生成输出单元。67 项可用下列框架无关的性质组织：词法标记与字符边界 15 项，解析后的事件类型、内容及顺序 33 项，跨块输入的暂存与内容关系 14 项，canonical/text 格式的识别及回退 5 项。对应建议单元分别为 `output.harmony_lexical_boundary`、`output.harmony_event_sequence`、`output.harmony_incremental_relation` 和 `output.harmony_format_selection`。这些名称是本批审查建议，尚未写入公共空间模型；现有对象中的类名与具体内部字段不作为单元定义。

另有 5 项仅检查 `Event`、`Token` 或解析策略对象的初始化字段，记录为实现层证据，不单独扩展成跨框架的请求验证性质。1 项 `test_canonical_standalone_structural_token_filtered` 仅断言 `len(events) >= 0`，对列表恒成立，无法支撑其注释所称“独立结构标记被过滤”，标为 `oracle_unresolved`。若要检查这一性质，需要断言实际事件内容或剩余缓冲区。同组其他几个边界测试也只检查最终答案子串出现，不能由此推断噪声标记已经被过滤；逐项记录已按断言而非测试名称措辞。

源码证据说明了四种性质的边界。`prefix_hold` 在产品源码第 24 行起计算需要保留的最长标记前缀；`iter_tokens` 第 46 行起产生标记类型及原文字符区间。`CanonicalStrategy.parse` 第 137 行起将通道块、工具调用和普通文本变为事件，`TextStrategy.parse` 第 441 行起处理没有结构标记的回退格式。`HarmonyParser.parse` 第 514 行起保持缓冲区并根据输入选择策略。两项“分块等价”测试分别比较 reasoning 与 normal **拼接内容**，其中 text 格式先去除首尾空白；它们没有要求每块产生相同数量、边界或顺序的事件。这一粒度差异需要在未来的独立验证仓库中明确。

复核命令：

```sh
experiments/.venv/bin/python experiments/semantic_space/agent_sglang_harmony_review.py
experiments/.venv/bin/python experiments/semantic_space/audit_source_review_batch.py --review experiments/semantic_space/agent_sglang_harmony_review.json --snapshot sglang --path test/registered/unit/parser/test_harmony_parser.py --suite '*' --output experiments/semantic_space/agent_sglang_harmony_review.tsv
experiments/.venv/bin/python -m unittest discover -s experiments/semantic_space/tests -p test_agent_sglang_harmony_review.py -v
```

本批校验结果为 73/73 方法均有记录，67 项 `needs_refinement`、5 项 `outside_request_verification`、1 项 `oracle_unresolved`；3 项独立工具测试通过。
