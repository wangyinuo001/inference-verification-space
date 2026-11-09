# 验证空间测试入口第五轮全量扫描

本轮处理“预期异常”检查。此前的返回值比较规则无法识别 `pytest.raises` 与 `self.assertRaises` 包围的被测调用，因此脚本重新扫描固定版本 vLLM 和 SGLang 的 31,968 条入口，记录异常类型、被测调用及异常上下文的源码行。运行 `experiments/.venv/bin/python experiments/semantic_space/fifth_pass_exception_contracts.py` 约需 5 秒，输出[逐项表](fifth-pass-all-tests.tsv)、[同目标审查分组](fifth-pass-review-groups.tsv)及[汇总](fifth-pass-all-tests.json)。

扫描得到 1,004 条直接的“产品调用位于预期异常上下文”静态关联，其中 vLLM 744 条、SGLang 260 条；与前三轮去重后，新增加 750 条入口的局部调用—判定关系。另有 257 条异常上下文包含多个候选调用、额外语句或嵌套操作，留待逐项核对。当前直接调用比较、辅助函数返回值比较及预期异常三类静态关联的并集为 4,736 条。这个数字表示源码关系，已确认映射到具体验证单元的 838 条仍按原证据等级单独统计。

例如，vLLM 的 `test_non_numeric_logprobs_rejected` 将 `CompletionRequest` 构造置于 `pytest.raises(VLLMValidationError)` 中；SGLang 的 `test_more_tokens_than_data_raises` 将 `_get_full_multimodal_text_prompt` 调用置于 `self.assertRaises(ValueError)` 中。两项均检查特定输入下的拒绝行为。目标调用和异常类型相同的 1,004 条入口分成 620 组，其中 542 条属于多成员组；分组用于安排源码审查，同组测试的非法输入和错误条件仍需分别核对。

抽样检查时发现，若一个产品对象的构造只是另一操作的参数，预期异常可能来自外层操作。脚本现仅接受异常上下文中单条语句直接调用产品函数的情况；多条语句、嵌套调用及多个产品调用均保留为待核查。全部 1,004 条关联的源码锚点存在，固定随机抽出的 20 条局部关联已回读；语义空间工具测试 201 项通过。此结果尚不说明特定参数实例实际执行，也不把“抛出某种异常”直接等同于某个框架无关性质。下一步需结合非法输入条件、错误消息和产品代码的拒绝分支，判断哪些关系对应同一验证单元，哪些需要补充空间定义。
