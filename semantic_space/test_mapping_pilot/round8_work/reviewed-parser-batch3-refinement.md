# 渲染与分词候选的验证性质补充

对前一批暂留的 61 个 vLLM renderer/tokenizer 入口，逐项核对固定源码中的断言后，形成 8 个候选验证单元。逐项映射与原始断言见 `reviewed-parser-batch3-refinement.tsv`；`reviewed-parser-batch3-refinement.json` 还记录每个候选单元的性质、条件、可观察量，以及固定 SGLang 源码或测试的行号与 SHA256。这些是空间定义建议，尚未写入 `space_seed.json` 或公共测试映射。

| 建议单元 | 条目数 | 具体检查 | SGLang 对照 |
| --- | ---: | --- | --- |
| 聊天提示词文本构造 | 24 | 消息角色、正文、工具和最终续写标记在渲染文本中的内容与次序 | `parser/conversation.py:107` 的 `get_prompt`；`test_conversation.py:39` 起直接检查文本 |
| 推理控制与提示词前缀 | 18 | 推理开关、预算及默认值如何进入渲染模式和提示词前缀 | `serving_chat.py:493,739`；`test_serving_chat.py:1729` 检查推理开关判定 |
| 渲染输入有效性 | 10 | 对不合法角色、媒体、工具参数、推理值或缺失特殊 token 的具体拒绝结果 | `serving_chat.py:498` 的参数拒绝；`test_serving_embedding.py:331` 的模板错误分类；条件按错误类型分开 |
| 媒体占位标记渲染 | 4 | 图像/音频标记与周围文本的顺序 | `parser/conversation.py:574`；`test_conversation.py:520,554` 检查媒体标记文本 |
| 渲染输入稳定性 | 2 | 消息未被改写、可重复使用内容再渲染的一致性 | `serving_chat.py:691,699` 使用深拷贝；尚需同性质的 SGLang 断言 |
| 等义入口表示的渲染关系 | 1 | Chat 与 Responses 形式的同义文本部分是否得到同一提示词 | `test_serving_responses.py:80` 已检查输入规范化；最终渲染关系待补断言 |
| 分词器调用约定 | 1 | BOS 和截断参数如何传给底层 encode | `serving_chat.py:805,818` 的特殊 token 策略与 encode 调用；SGLang 直接断言待补 |
| 格式标识与实现绑定 | 1 | 注册表为指定格式选中哪一种渲染器或模板 | `parser/conversation.py:495`、`test_conversation.py:729` 是相近的模板注册语义，具体接口不同 |

最明显的空间缺口是“渲染后的提示词文本”与“经分词器得到的 token ID”目前共用同一候选标签。前者可以在不调用分词器的情况下检查角色、工具、分隔符与特殊标记；后者还需给定分词器并逐项比较 token ID。两者应分开作为验证要求，媒体占位文本也应与媒体数据、位置和张量的实际绑定分开。建议单元的跨框架证据强度已在 JSON 中逐一标明：对仅有对应实现、尚无同性质断言的条目，仍保留待验证状态。
