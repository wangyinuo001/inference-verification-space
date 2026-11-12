# 提示词文本与媒体标记的源码核查

固定快照中的 28 个入口分别检查提示词文本构造（24 条）和媒体标记在文本中的位置（4 条）。每条记录的真实断言、源码行及 SHA256 见 `reviewed-rendering-batch1.tsv` 和 `reviewed-rendering-batch1-evidence.json`。vLLM 的 DeepSeek-V4 渲染函数返回提示词字符串，SGLang 的 `Conversation.get_prompt` 也返回字符串；两侧测试分别检查角色、正文、分隔符和续写前缀。媒体测试检查图像或音频标记相对于正文的位置。两项单元已在 `space_seed.json` 中给出条件、oracle、执行边界和双框架证据，未修改公共测试映射。

文本渲染发生在分词器编码之前。提示词字符串符合模板，并不能直接证明编码后的 token ID 与指定分词器一致；后者还受分词器版本、特殊 token 策略影响。媒体标记也只是文本中的占位位置，不能说明图像或音频载荷已经被读取、展开并绑定到正确的张量位置。因此这两项与现有 `input.chat_template_tokenization`、`input.multimodal_binding` 分别定义。单条测试若仅断言前缀、子串或标记缺席，映射记录只主张该局部性质。
