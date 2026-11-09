# SGLang 跨领域关联核查

核查 `second-pass-cross-scope.tsv` 中全部 19 条 SGLang `linked_call_crosses_area_candidate` 记录。逐项回读固定 0.5.14 测试源码、实际断言及被调产品函数，结果见 [逐项表](agent_sglang_cross_scope_review.tsv)。这 19 条集中在两个产品函数族，不能作为整个测试库的随机样本。

其中 18 条原候选领域为 `output`，产品模块提示为 `ingress`。测试直接调用 `detect_jinja_template_content_format` 或 `process_content_for_template_format`，检查的是模型执行前的 Jinja 模板内容格式选择、消息文本整理与图像／音频／视频 URL 提取。前一函数在 `python/sglang/srt/parser/jinja_template_utils.py:81` 根据模板内容循环等条件返回 `openai` 或 `string`；后一函数在同文件 123 行起将请求消息整理为模板可消费的输入，并可更新多模态数据列表。例如测试 `test_process_content_openai_format` 在 164–179 行检查 `image_data` 和整理后的消息，`test_process_content_string_format` 在 200–209 行检查压成文本后不提取图像。这些断言指向请求输入处理，原 `output` 候选领域应改为 `ingress`。本批未发现这 18 条同时验证最终模型输出的证据。

余下一条是 `TestPortArgsIPv6::test_init_new_with_ipv6_address`。`PortArgs.init_new` 定义在 `server_args.py:7269`，但测试在 `test_network_address.py:243–260` 给定 IPv6 初始化地址后，检查 tokenizer、scheduler、detokenizer 的 TCP 地址与 NCCL 端口。原 `runtime_support` 候选领域与检查目标相符，产品模块的 `configuration` 提示只是代码位置；这里无需改变测试领域。该方法依赖 `ServerArgs` 配置输入，但断言对象是派生出的运行时通信地址。

因此，这批冲突揭示两种不同处理：测试名或候选标签把“模板的输出格式”误读为“模型输出”时，以调用和断言修正领域；产品函数位于配置模块、实际却检查运行时地址时，保留测试领域并将模块位置标为实现细节。交叉领域提示适合作为复核队列，不能直接合并两个领域，也不能单凭模块路径覆盖断言所指向的对象。
