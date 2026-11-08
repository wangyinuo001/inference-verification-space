# 流式响应的验证单元与独立测试规格

流式接口将一次请求的结果拆成多个事件。验证要同时检查事件如何开闭，以及这些事件承载的内容能否组成最终结果。vLLM 与 SGLang 都实现了 OpenAI Responses 风格的流式路径，内部状态机、解析器和引擎分块方式不同；共同的检查对象是请求、内容项和事件之间的关系。空间因此将 `response.stream_event_lifecycle` 与 `response.stream_payload_consistency` 分开，二者均依赖同一次请求的标识、事件序列和最终响应。

`response.stream_event_lifecycle` 检查创建、运行、内容项开启、增量、内容项结束与请求终止的合法顺序，同一内容项的 ID 和索引，以及接口提供序号时的连续性。`response.stream_payload_consistency` 按内容项分别拼接文本、推理内容或工具实参的增量，与 done 事件和最终响应核对；不同字段不能串入目标内容。适用条件至少记录接口种类、输出模式、内容种类及顺序、引擎分块边界、工具调用数量和结束原因。事件顺序正确与内容一致是两项不同的判定，某项测试只检查其中一项时分别记录。

独立仓库可用同一组脚本化引擎结果分别驱动两个框架的适配入口，不复制其原测试。第一组输入产生三段累积文本，检查事件开闭、连续序号、拼接后的文本与最终响应。第二组将工具名称与实参分在不同块，并在一次请求中安排“正文—工具—正文”，检查内容项顺序、ID 及最终输出。第三组改变 JSON 字符串的切分位置，在工具实参中加入非目标字段，检查输出只包含目标字段；完成事件中的内容还要与最终响应一致。适配器需报告无法观察到的字段；接口不提供序号时该项检查不适用。脚本化输入只验证流式转换，模型实际输出到转换器的交接仍需另行记录。

源码对照已逐项登记 [vLLM 的 14 项事件处理测试](../semantic_space/vllm-response-stream-source-review.tsv)与 [SGLang 的 3 项非 Harmony 流式测试](../semantic_space/sglang-response-stream-source-review.tsv)。vLLM 的 `split_delta` 和 `SimpleStreamingEventProcessor` 分离复合增量并控制内容项的开闭；SGLang 的 `responses_stream_generator_non_harmony` 为 SSE 事件赋序号，按引擎输出和解析结果生成内容项。vLLM 另有一项用解析器调用次数检查长工具输入的处理成本，已标为待细化性能要求；它不替代内容正确性的断言。两套审阅记录均为固定源码和断言核对，尚未作为真实请求的运行追踪。
