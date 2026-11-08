# 推理系统 Verification Space

这个研究预览把推理请求中的验证需求组织为性质、适用条件和跨组件关系。分析对象是固定版本的 vLLM 与 SGLang 0.5.14；交互页面按请求路径展示验证单元，左侧 Explorer 保留两套仓库的测试目录结构。

## 查看

[在线打开交互页面](https://wangyinuo001.github.io/inference-verification-space/semantic_space/space_view.html)，也可以下载本目录后在浏览器中打开 [离线页面](semantic_space/space_view.html)。页面包含所需数据，无需安装两个推理框架。文件较大，首次打开需要等待加载。

- 顶部选择文本生成、前缀复用、运行中取消、分布式生成等十五类代表性路径，查看相关验证性质及其依赖。
- 左侧展开或搜索原仓库测试文件，点击测试入口，查看源码归类、已定位的调用和断言线索，以及对应的空间单元。图上标亮的具体证据等级以详情为准；静态归属不能代替运行和断言核查。
- 点击单元可查看要检查的性质、输入与状态条件，以及固定版本源码位置。紫色和绿色区分验证对象涉及的状态范围，不表示测试等价。

两套源码快照固定在 [vLLM `76a0ccc`](https://github.com/vllm-project/vllm/tree/76a0ccc209343971c6f4b56f6380a76202428385) 和 [SGLang `49e384c`](https://github.com/sgl-project/sglang/tree/49e384ce9d304648e9959666ecb8ce8cd98d0deb)。本预览保存源码索引及指向这两个提交的链接，不重复发布上游完整仓库。

## 当前材料

候选空间包含 **100 个验证单元、71 个条件维度、15 类代表性路径**。Explorer 索引覆盖 4,673 个测试相关源码文件及约三万个源码测试入口。[空间定义](semantic_space/space_seed.json)和[测试目录索引](semantic_space/test_catalog.json)可以分别检查验证需求及原仓库入口。

[全量审查台账](semantic_space/master-review-index.tsv)将 31,968 条测试定义、继承候选、Rust 测试及待解析文件记录分流，并保存每条记录的证据等级与下一步核查项。其中 9,427 条存在静态调用与断言关联候选，22,541 条按缺失关联的原因继续分诊。源码候选只用于安排阅读，不意味着已确认它检查了对应性质。对自动关联的[留出样本核查](semantic_space/agent_contract_holdout.md)发现输入构造与被测操作容易混淆；修订规则会把这类关联降为待审查。

对没有直接产品调用链接的 HTTP／SDK 测试，[进一步分诊](semantic_space/agent_http_boundary.md)已定位 914 个入口的请求调用和断言线索，其中 777 个有源码级候选关联。这些记录已进入全量台账，仍需逐项确认请求条件、实际执行和检查性质。

当前正在逐项回读测试、调用路径与断言，并用确认的共性和实现差异修订空间单元。目标是据此为独立验证仓库设计测试，而不是将 vLLM、SGLang 的现有测试直接拼接为新测试集。[源码对照摘要](docs/source-comparison.md)说明两个框架如何映射到同一验证需求。

## 文件

- `semantic_space/space_view.html`：可离线打开的当前交互页面；`space_view.png` 为静态预览。
- `semantic_space/space_seed.json`：框架无关的候选单元、条件、关系和路径，以及固定源码证据。
- `semantic_space/test_catalog.json`：Explorer 的测试目录与源码候选归类。
- `semantic_space/build_review_families.py`：批量抽取测试结构和审查分组；需在完整研究工作目录及两个固定源码快照下运行。
- `semantic_space/agent_contract_extract_v2.py`：保守抽取调用与断言关系的候选；输出仍需源码核查。
- `semantic_space/master-review-index.tsv`：逐入口审查队列，而非已确认验证覆盖清单。

此目录是供阅读和审查的预览。页面可以单独使用；重新生成全部索引和页面需要研究工作目录中的其余分析脚本、证据文件与固定源码快照。
