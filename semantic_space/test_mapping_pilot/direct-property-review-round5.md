# 解析、请求转换与采样性质的源码审查（第五轮）

固定源码快照：vLLM `76a0ccc209343971c6f4b56f6380a76202428385`，SGLang `49e384ce9d304648e9959666ecb8ce8cd98d0deb`。本轮检查测试输入、执行条件与源码断言，并为每条局部性质保存测试定义和断言行。没有运行模型、服务、GPU 或保留评估集。

在[第四轮](direct-property-review-round4.md)已净增 560 条的基础上，本轮又写入 **770 条**直接审查映射。其中 SGLang 三份文件的 **316 条**测试在旧的逐项审查台账中已经是绿色，本轮只是补充具体断言和源码锚点，不能重复计为新增；vLLM 五份文件净增 **454 条**。因此这次连续工作由 **2,144** 增至 **3,158 / 31,968**，**净增 1,014 条**。直接审查映射总数为 1,541，另有 244 条原确认映射和 891 条继承方法映射，三者键互不重叠；其余绿色来自此前的逐项源码审查台账。

| 固定源码文件 | 本轮新增映射 | 净增绿色 | 审查内容与边界 |
| --- | ---: | ---: | --- |
| SGLang `test_function_call_parser.py` | 193 | 0 | Pythonic、Mistral、Llama、Kimi、DeepSeek、Qwen、GLM 等工具解析的文本、名称、实参、顺序和分块；排除只检查内部计数、类型或注册形状的测试 |
| SGLang `test_harmony_parser.py` | 66 | 0 | 标记前缀、通道事件、格式选择与分块内容；排除构造器字段及无实质约束的断言 |
| SGLang `test_sampling_params.py` | 57 | 0 | 温度、top-p、数值边界、停止条件与正则缓冲上界；11 条仅以未抛异常为接受条件而无显式断言的测试未加入 |
| vLLM `test_gemma4_streaming_reasoning.py` | 63 | 63 | 推理/正文和工具调用分离、流式与完整输入关系、schema 类型转换；排除内部转换次数及纯状态恢复检查 |
| vLLM `test_inkling.py` | 61 | 61 | 参数转换、标记边界、工具/推理通道及分块输出；排除只检查解析器类或内部初始状态的测试 |
| vLLM `test_serving_conversion.py`（Cohere） | 104 | 104 | 请求消息、工具、模板参数、引用索引、完成响应、用量及错误分类；排除两个只检查常量集合的测试 |
| vLLM `test_anthropic_messages_conversion.py` | 87 | 87 | 图像、工具结果、推理块、系统消息、用量、停止原因、参数和错误响应 |
| vLLM `test_structural_tag_registry.py` | 73 | 73 | 工具约束语法接受/拒绝、选择模式和 schema 传递；排除五个只检查注册表或类字段的测试 |
| vLLM `test_chat_parsing.py` | 66 | 66 | 响应模板、流式事件、前缀截断和工具参数类型；另九条已在原确认映射中，未重复加入 |

映射保存在 [`direct-property-crosswalk.tsv`](direct-property-crosswalk.tsv)，由 [`build_direct_property_crosswalk.py`](build_direct_property_crosswalk.py) 在固定文件 SHA-256、测试定义行、单元 ID 和实际断言位置均匹配时重建。条目中的性质描述列出具体断言表达式，适用条件指向测试构造的输入和配置。绿色表示**局部源码性质已确认并放入候选空间**；不表示测试已运行、完整单元已覆盖或端到端请求通过。
