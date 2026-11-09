# 推理系统 Verification Space

这个研究预览把推理请求中的验证需求组织为性质、适用条件和跨组件关系。分析对象是固定版本的 vLLM 与 SGLang 0.5.14；交互页面按请求路径展示验证单元，左侧 Explorer 保留两套仓库的测试目录结构。

## 查看

[在线打开交互页面](https://wangyinuo001.github.io/inference-verification-space/semantic_space/space_view.html)，也可以下载本目录后在浏览器中打开 [离线页面](semantic_space/space_view.html)。页面包含所需数据，无需安装两个推理框架。文件较大，首次打开需要等待加载。

- 顶部选择文本生成、流式响应、前缀复用、运行中取消、分布式生成等十六类代表性路径，查看相关验证性质及其依赖。
- 左侧展开或搜索原仓库测试文件，点击测试入口，查看源码归类、已定位的调用和断言线索，以及对应的空间单元。图上标亮的具体证据等级以详情为准；静态归属不能代替运行和断言核查。
- 点击单元可查看要检查的性质、输入与状态条件，以及固定版本源码位置。紫色和绿色区分验证对象涉及的状态范围，不表示测试等价。

两套源码快照固定在 [vLLM `76a0ccc`](https://github.com/vllm-project/vllm/tree/76a0ccc209343971c6f4b56f6380a76202428385) 和 [SGLang `49e384c`](https://github.com/sgl-project/sglang/tree/49e384ce9d304648e9959666ecb8ce8cd98d0deb)。本预览保存源码索引及指向这两个提交的链接，不重复发布上游完整仓库。

## 当前材料

候选空间包含 **113 个验证单元、77 个条件维度、16 类代表性路径**。Explorer 索引覆盖 4,673 个测试相关源码文件及约三万个源码测试入口。[空间定义](semantic_space/space_seed.json)和[测试目录索引](semantic_space/test_catalog.json)可以分别检查验证需求及原仓库入口。

[全量审查台账](semantic_space/master-review-index.tsv)将 31,968 条测试定义、继承候选、Rust 测试及待解析文件记录分流，并保存每条记录的证据等级与下一步核查项。其中 9,427 条存在静态调用与断言关联候选，22,541 条按缺失关联的原因继续分诊。源码候选只用于安排阅读，不意味着已确认它检查了对应性质。对自动关联的[留出样本核查](semantic_space/agent_contract_holdout.md)发现输入构造与被测操作容易混淆；修订规则会把这类关联降为待审查。

台账将分批源码核查与静态候选分别记录。目前 1,293 条入口已有登记的分批审查结果，其中 838 条在方法级映射到验证单元；另有独立源码审查记录和运行追踪。每条的检查性质、断言行、适用条件与证据等级均可在台账中查看。

新增加的[批量调用—断言关联](semantic_space/bulk_call_oracle_links.py)在约 17 秒内扫描了 9,427 条已有产品调用候选，得到 3,540 条直接源码关联，其中 2,549 条超出此前的机械候选。它们在台账中单列为 `source_call_oracle_link_candidate`，仍需确认输入、状态及具体检查性质。对新增关联抽取的 80 条留出样本逐项回看，发现 1 条参考值误连；规则修订后，保留的 79 条局部关联均成立。[SGLang 核查](semantic_space/agent_bulk_sglang_holdout.md)与 [vLLM 核查](semantic_space/agent_vllm_bulk_holdout_review.md)保存逐项依据。

[全量快速初筛](semantic_space/fast-pass-report-20261109.md)已将 31,968 条索引记录逐项分流，保存候选领域、证据等级、风险与下一步核查动作；[逐项结果](semantic_space/fast-pass-all-tests.tsv)和[分组队列](semantic_space/fast-pass-review-families.tsv)可用于继续审查。两套框架各 80 条弱证据领域归类的源码抽查发现，名称和目录会将部分测试归错对象或漏掉跨领域性质，因此页面中的候选归类不能直接当成已确认的验证覆盖。当前源码级性质核查记录为 1,191 条，其中 838 条已映射空间单元。

对没有直接产品调用链接的 HTTP／SDK 测试，[进一步分诊](semantic_space/agent_http_boundary.md)已定位 914 个入口的请求调用和断言线索，其中 777 个有源码级候选关联。这些记录已进入全量台账，仍需逐项确认请求条件、实际执行和检查性质。

当前正在逐项回读测试、调用路径与断言，并用确认的共性和实现差异修订空间单元。目标是据此为独立验证仓库设计测试，而不是将 vLLM、SGLang 的现有测试直接拼接为新测试集。[源码对照摘要](docs/source-comparison.md)说明两个框架如何映射到同一验证需求。

[流式响应测试规格](docs/stream-response.md)区分事件开闭顺序与分段内容对最终结果的一致性，逐项源码记录见 [vLLM](semantic_space/vllm-response-stream-source-review.tsv) 和 [SGLang](semantic_space/sglang-response-stream-source-review.tsv)。

[工具调用解析测试规格](docs/tool-call-parse.md)记录生成文本、工具名称和实参之间的共同验证目标及框架适用条件，逐项源码记录见 [vLLM](semantic_space/vllm-pythonic-tool-source-review.tsv) 和 [SGLang](semantic_space/sglang-pythonic-tool-source-review.tsv)。

[适配器驻留测试规格](docs/adapter-residency.md)区分容量与淘汰策略、槽位身份及请求所用权重，逐项源码记录见 [vLLM](semantic_space/vllm-lora-residency-source-review.tsv) 和 [SGLang](semantic_space/sglang-lora-eviction-source-review.tsv)。

[缓存事件与输出解析测试规格](docs/cache-event-output-parser.md)新增缓存策略参数、事件载荷与元数据生命周期，以及结构化输出解析的词法、语义与增量关系。逐项审查见 [SGLang 缓存策略](semantic_space/sglang-eviction-policy-reconciled-source-review.tsv)、[vLLM 跨层事件](semantic_space/vllm-offloading-events-reconciled-source-review.tsv)和 [SGLang Harmony 解析](semantic_space/sglang-harmony-reconciled-source-review.tsv)。

## 文件

- `semantic_space/space_view.html`：可离线打开的当前交互页面；`space_view.png` 为静态预览。
- `semantic_space/space_seed.json`：框架无关的候选单元、条件、关系和路径，以及固定源码证据。
- `semantic_space/test_catalog.json`：Explorer 的测试目录与源码候选归类。
- `semantic_space/build_review_families.py`：批量抽取测试结构和审查分组；需在完整研究工作目录及两个固定源码快照下运行。
- `semantic_space/agent_contract_extract_v2.py`：保守抽取调用与断言关系的候选；输出仍需源码核查。
- `semantic_space/master-review-index.tsv`：逐入口审查队列，而非已确认验证覆盖清单。

此目录是供阅读和审查的预览。页面可以单独使用；重新生成全部索引和页面需要研究工作目录中的其余分析脚本、证据文件与固定源码快照。
