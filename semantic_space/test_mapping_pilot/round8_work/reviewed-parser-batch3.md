# vLLM 解析与渲染候选源码核查（本地批次 3）

本批从 `candidates.tsv` 固定抽取 10 个 vLLM 源文件、225 个尚未进入既有 crosswalk 或前两批 review 的测试入口。脚本逐项核对固定源码 SHA256、测试定义行、断言所在 AST 节点，并保存测试体中的调用线索。156 条有明确断言的入口写入 `reviewed-parser-batch3.tsv`；69 条留在 `reviewed-parser-batch3-evidence.json` 的 `needs_refinement` 队列。此处是局部源码审查，未增加公共绿色映射，也未运行测试。JSON 的 `file_reviews` 逐文件记录快照、路径、哈希、候选数与核查说明。

| 文件 | 候选 | 本批映射 | 主要核查对象 |
| --- | ---: | ---: | --- |
| `tests/parser/test_harmony.py` | 39 | 35 | 完整与分块 Harmony 消息的推理、正文、工具事件；4 项单独计数检查待细化 |
| `tests/parser/test_streaming.py` | 21 | 21 | 分块输出、推理与工具解析组合、工具选择的输出行为 |
| `tests/parser/test_parse.py` | 19 | 18 | 完整解析与提示词边界；1 项历史工具调用计数待细化 |
| `tests/reasoning/test_minimax_m3_reasoning_parser.py` | 15 | 15 | 推理标记与正文分离、跨块结束状态 |
| `tests/parser/engine/test_llama_json.py` | 23 | 22 | JSON 工具名、参数、回退和分块关系；1 项注册别名待细化 |
| `tests/parser/engine/test_plamo3.py` | 22 | 22 | 完整与流式解析、截断标记及多工具调用 |
| `tests/parser/engine/test_minicpmv_reasoning.py` | 22 | 20 | 推理与工具输出；注册检查及独立换行规范化待细化 |
| `tests/renderers/test_inkling.py` | 21 | 1 | 仅 1 项逐项比较生成的 token ID；其余主要比较渲染文本 |
| `tests/tokenizers_/test_deepseek_v4.py` | 21 | 0 | 提示词文本、配置和异常判定，未核对参考 token ID |
| `tests/tokenizers_/test_deepseek_v41.py` | 22 | 2 | 图像 UUID、实际图像数据次序及被丢弃图像的缺席 |

源码核查纠正了几类候选路由。`test_streaming.py` 和 `test_parse.py` 的若干 `tool_choice` 测试检查的是已解析的调用或回退文本，没有检查生成的约束 schema，因此映射到工具调用解析关系。`test_harmony.py` 中完整消息解析与分块解析分开映射；单独的 `count_reasoning_tokens` 检查不等同于响应总用量。`test_llama_json.py` 的注册别名测试只检查类名。对只断言工具名称、数量或字段存在的测试，TSV 的 `property` 保留实际断言表达式，不推断未比较的参数值正确。

现有空间还需要一个独立于 token ID 对照的“聊天提示词渲染”性质：在给定消息角色、内容、工具与模板参数时，渲染后的文本、特殊标记、顺序及错误处理符合模板规则。vLLM `vllm/renderers/inkling_encoding.py:93` 与 `tests/renderers/test_inkling.py:196` 等检查渲染文本；SGLang `python/sglang/srt/parser/conversation.py:107` 的 `Conversation.get_prompt` 与 `test/registered/unit/parser/test_conversation.py:39-53` 检查角色、分隔符和提示词文本。这两侧的证据支持将渲染文本与后续 token ID 对照拆成不同验证要求。vLLM 的推理 token 计数、注册别名、历史调用计数和换行规范化目前仅记录为待细化性质，尚未据此增加跨框架单元。
