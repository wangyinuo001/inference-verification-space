# 局部契约抽取的留出样本核查

## 核查对象与方法

`agent_contract_extract.tsv` 对 9,427 条已有静态调用—断言链接抽取局部契约候选，其中 1,421 条标为 `mechanical_candidate`。本次核查只评价这一标签所对应的**产品调用与断言的源码级关系**，不评价测试是否实际运行、不评价验证空间单元归类，也不据此断言某项系统性质已经得到端到端验证。

先用 `(snapshot, path, test_name)` 排除此前四批人工审阅的入口，候选池剩余 1,372 条（SGLang 520、vLLM 852）。使用固定随机种子 `20261108`，按仓库和 `object_candidates` 分层抽取 80 条：每个仓库 10 个对象类别，每类 4 条。逐条回读固定源码快照，核对调用位置、断言或预期异常的作用域、被检查的返回值或状态、调用是否仅用于构造输入或期望值。每条结果及两端源码行写入 [`agent_contract_holdout.tsv`](agent_contract_holdout.tsv)。

判定规则：`true` 表示源码明确显示候选调用的返回值、构造时行为或其直接产出的状态接受所列检查；`false` 表示链接的断言实际检查另一操作，而候选调用只提供测试材料或期望值；`unclear` 表示仅靠源码无法确定关系。本次没有 `unclear`，但 `true` 仅表示**局部静态链接成立**，不表示运行覆盖或检查充分。

## 结果

| 固定源码快照 | 样本数 | true | false |
| --- | ---: | ---: | ---: |
| SGLang 0.5.14 | 40 | 27 | 13 |
| vLLM 2026-11-05 | 40 | 24 | 16 |
| 合计 | 80 | 51 | 29 |

29 条误连中，23 条把输入或状态的构造调用连到了**后续另一个方法**的断言，6 条把断言期望值里的构造或转换调用误当成被测调用。具体例子如下：

- SGLang `test_scheduler_flush_cache.py:55,61,65` 中，`FlushCacheReqInput` 构造请求；测试实际执行 `scheduler.flush_wrapper.handle(req)`，随后检查待处理请求身份和截止时间。将构造器当作 `handle` 的验证性质会丢失操作边界。
- SGLang `test_template_manager.py:44–48` 中，`ReasoningToggleConfig(...)` 出现在 `assertEqual` 的**期望值**一侧；测试核查的是模板检测输出。相同形式见 vLLM `test_manager.py:2869` 中的 `JobResult(...)`。
- vLLM `test_store.py:1453–1458` 中，`BlockObject(...)` 只是传给 `store.put` 的输入；`pytest.raises(RuntimeError)` 检查的是关闭后的存取拒绝。
- vLLM `test_output_processor.py:1760–1777` 中，`OutputProcessor(...)` 只是建立处理器，断言检查 `process_outputs` 后的请求存活和队列输出。
- 反过来，SGLang `test_conv_layer.py:168–169` 的 `Conv2dLayer(...)` 本身位于 `assertRaises` 作用域内，vLLM `test_multimodal_config.py:93–94` 的 `MultiModalConfig(...)` 本身位于 `pytest.raises` 作用域内；这两种构造调用确实是被检查的操作。因此不能简单剔除所有构造器。

## 对批量抽取的含义

`mechanical_candidate` 足以把大规模源码入口压缩成可审阅的调用、输入和断言材料，却不能直接作为“该测试验证此产品调用的某项性质”的标签。后续自动抽取应先辨认断言的实际值与期望值，追踪构造对象到被调用方法的关系，并识别异常作用域中的具体抛错操作。遇到多次状态变化时，应把构造、执行和断言作为一个候选调用链呈现，保留人工核查入口。此次样本按仓库和类别**等量抽样**，并非按 1,372 条候选的真实分布抽样；51/80 与 29/80 只描述这 80 条，不能外推成全量准确率。完整的测试性质判断还需结合输入配置、运行状态和实际执行证据。

## 独立 v2 保守过滤复核

基于上述反例，新建 [`agent_contract_extract_v2.py`](agent_contract_extract_v2.py) 作为独立过滤步骤，不修改第一版材料。它用 AST 识别候选调用是否位于比较的期望值侧、是否作为另一操作的输入、构造对象是否在断言前又被方法调用。此类候选标为 `abstain`，原有 `needs_review` 保持不变；过滤步骤不自动确认任何验证性质。全量结果保存在 [`agent_contract_extract_v2.tsv`](agent_contract_extract_v2.tsv)。

| 来源 | 原 mechanical_candidate | v2 保留 | v2 abstain |
| --- | ---: | ---: | ---: |
| SGLang | 520 | 377 | 143 |
| vLLM | 901 | 614 | 287 |
| 合计 | 1,421 | 991 | 430 |

同一 80 条留出样本中，原先 29 条误连全部转为 `abstain`；51 条源码确证对应中，49 条仍为 `mechanical_candidate`，2 条被保守降级。两条降级分别是 SGLang `test_runtime_context.py:96–99` 中借助 `getattr(get_parallel(), attr)` 检查 getter 委托，以及 vLLM `test_gc_utils.py:93–98` 中对 `untracked_list()` 结果经过列表操作后检查 GC 跟踪状态。v2 保留的 49 条在这 80 条里没有误连，但该数字不代表其余 9427 条或 991 条的准确率。五项针对期望值、输入构造、跨行断言及直接检查的单元测试已通过：`python3 -m unittest experiments/semantic_space/tests/test_agent_contract_extract_v2.py`。
