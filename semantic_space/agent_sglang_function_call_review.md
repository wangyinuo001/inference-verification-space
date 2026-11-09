# SGLang 0.5.14 工具调用解析测试源码审查

审查对象是固定快照 `test/registered/unit/function_call/test_function_call_parser.py` 的全部 210 个源码级 `test_` 方法。逐项记录测试体中的固定输入与配置、实际调用、断言表达式及其行号，并将解析器实现的定义位置一并保存。逐项记录见 [JSON](agent_sglang_function_call_review.json) 和 [TSV](agent_sglang_function_call_review.tsv)；[生成脚本](agent_sglang_function_call_review.py) 从固定源码与断言台账生成记录。这里审查的是源码中能直接支持的检查性质，没有把测试名或运行结果当作证据。

## 归类结果

| 结果 | 入口数 | 依据 |
| --- | ---: | --- |
| 映射至已有验证单元 | 199 | 其中 170 项检查模型输出的工具调用解析关系，20 项检查工具约束结构，9 项检查实参的数值或结构解析。 |
| 需要细化验证单元 | 10 | 9 项仅检查流式解析返回对象的类型，未断言解析内容；1 项检查 `get_structure_constraint` 的 `thinking_mode=False` 签名默认值。 |
| 不属于请求行为验证 | 1 | `test_json_detector_has_no_ebnf` 仅检查解析器对象没有 `build_ebnf` 属性。 |

解析关系的检查包括完整输出与流式分块两种入口。已有断言分别核对普通文本是否保留、工具名与调用顺序、参数值和类型、未完成调用是否暂缓输出、非法格式的回退，以及流式状态和工具编号。具体成立范围随输入文本格式、分块位置、提供的工具列表和解析器实现而变化；例如 Qwen3 Coder 的不完整调用测试只检查返回 `StreamingParseResult` 类型，不能归为“不完整内容被正确缓冲”的证据。GLM4/GLM4.7 的转义 JSON 数组案例通过测试内 `check_params` 和 `check_single_todos` helper 断言数组长度、元素字段与值，审查记录包含 helper 内的实际断言行。

结构约束测试单独检查 `tool_choice`、`strict` 与解析器能力的组合。Kimi、DeepSeek、Qwen 和 GLM 等实现使用不同的工具标记及结构标签；这些格式差异是验证条件，能够抽象出的共同性质是“给定工具定义和选择策略，生成相应的约束表示”。解析器的工具标记检测、工具输出解析、实参类型转换、结构约束生成是不同检查，不因涉及同一工具调用而互相替代。

本批提示两个更窄的空间单元：`流式解析接口返回类型` 和 `工具约束接口默认参数`。它们可保留源码中已有的弱断言，同时避免把返回类型或签名检查误记为输出内容正确。批次校验命令：

```bash
python3 experiments/semantic_space/audit_source_review_batch.py \
  --review experiments/semantic_space/agent_sglang_function_call_review.json \
  --snapshot sglang \
  --path test/registered/unit/function_call/test_function_call_parser.py \
  --suite '*' \
  --output experiments/semantic_space/agent_sglang_function_call_review.tsv
```

校验通过：210 项方法键均对应固定源码；断言行、产品源码行和空间单元引用均通过批次审计。`candidate_config_count=0` 表示本批不展开运行时参数化实例，也不把源码可见断言扩展成生产服务行为结论。
