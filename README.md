# 推理系统 Verification Space

研究推理系统需要验证哪些功能、行为与性质，以及如何把这些验证需求组织为一个分层、框架无关、可支持后续剪枝的空间。分析对象是两个固定版本：

| 框架 | commit |
|---|---|
| vLLM | [`76a0ccc`](https://github.com/vllm-project/vllm/tree/76a0ccc209343971c6f4b56f6380a76202428385) |
| SGLang 0.5.14 | [`49e384c`](https://github.com/sgl-project/sglang/tree/49e384ce9d304648e9959666ecb8ce8cd98d0deb) |

空间中的对象是验证需求，不是现有测试。后续独立验证仓库中的测试将按这些需求重新设计，而不是复用或拼接两个框架已有的测试。

## 空间结构

从两个框架的源码出发，沿服务启动和一次请求的执行路径（配置 → 入口 → 渲染分词 → 调度 → 跨步状态 → 步绑定 → 数值计算 → 采样 → 输出组装，以及并行拓扑）整理出 76 个验证单元。每个单元记录：性质、判定方式、适用条件、依赖的下层单元、组合缺口，以及两个框架中的源码位置和语义差异。

**分层依据是判定一条性质所需的最小真实组件集合**，而不是代码所在目录：

| 层 | 需要真实运行的部分 | 单元数 | 例子 |
|---|---|---:|---|
| B0 纯函数 | 单个函数或算子 | 14 | 注意力后端对 FP32 参考；进程组划分 |
| B1 单组件状态机 | 一个带状态的组件，模型可替换为伪实现 | 29 | KV 分配守恒；前缀键隔离；增量解码；取消清理 |
| B2 单步引擎 | 调度结果、状态与一次真实前向 | 11 | 写入槽位归属；CUDA 图填充行不写 KV |
| B3 单请求多步 | 一个请求的完整生成 | 14 | 缓存命中与冷启动等价；流式与非流式一致 |
| B4 多请求并发 | 多个请求同批执行 | 2 | 批不变性；混合 LoRA 隔离 |
| B5 多进程拓扑 | TP/PP/DP/EP/PD | 6 | 并行配置与单卡等价；PD 传输 KV 一致 |

同一组件的不同性质可能落在不同层。例如前缀缓存的“键能区分 LoRA 与多模态输入”在 B1 即可判定，“命中后输出与冷启动相同”则必须在 B3 观察；进程组划分虽属分布式模块，却是 B0 纯函数。

其余维度：

- **性质类型**：数值保真、确定性契约、资源守恒与所有权、协议与生命周期、执行等价、分布。
- **执行扰动**：前缀缓存、分块 prefill、抢占、重叠调度、CUDA 图、推测解码、并行、PD 分离等 15 种优化都在不改变请求语义的前提下改变执行方式。每个执行等价单元对应一种扰动。
- **条件**：35 个维度（页大小、前向模式、后端、dtype、采样方式、停止配置等），以及源码中已有的组合约束。
- **组合缺口**：16 个上层单元写明了下层性质都成立时仍不能推出的部分，后续剪枝不能简单省略这些单元。

完整论证与框架差异见 [docs/source-model.md](docs/source-model.md)，逐单元列表见 [source_model/units.md](source_model/units.md)。

## 查看

| 入口 | 内容 |
|---|---|
| [源码调研与分层模型](docs/source-model.md) | 执行路径对照、单元抽象、分层依据、跨层依赖、框架差异 |
| [验证单元表](source_model/units.md) | 76 个单元按层次排列，源码位置链接到固定 commit |
| [交互页面](https://wangyinuo001.github.io/inference-verification-space/semantic_space/space_view.html) | 按 16 类请求路径浏览早期候选空间，左侧可反查两套仓库的测试；文件约 49 MB，首次加载较慢 |
| [测试归类进度](https://wangyinuo001.github.io/inference-verification-space/review_progress/) | 独立页面；全部 31,968 个源码级入口按证据状态显示，可按仓库与状态筛选。绿色表示性质已核对，青色表示调用结果与断言有直接源码关联 |
| [本周示意图](docs/figures/verification-space-weekly-20261109.png) | 以前缀缓存为例：两个框架的实现、共同验证需求与测试映射（[SVG](docs/figures/verification-space-weekly-20261109.svg)） |
| [四条路径校准](docs/four-path-calibration.md) | 文本生成、前缀复用、运行中取消与张量并行的共同性质、已有断言及仍缺少的路径条件 |
| [文本生成与取消的跨环节校准](docs/generation-abort-handoff.md) | 请求身份、批次行、token 位置、取消事件和资源释放的观察条件；对应的[结构化规格](semantic_space/handoff_contracts.json) |
| [张量并行分片测试规格](docs/tensor-parallel-binding-spec.md) | 逐 rank 的输入、权重与局部输出绑定，以及归约后结果的判定 |

交互页面基于早期的候选空间（115 个候选单元，[space_seed.json](semantic_space/space_seed.json)），尚未改为由源码模型生成。

## 证据来源

**源码调研**（[source_model/traces/](source_model/traces/)）。原有六条调研线共 99 条记录，另补充启动配置与传输状态两条记录线；覆盖请求输入输出、调度与生命周期、KV 缓存、模型执行与采样、分布式与权重加载，以及推测解码、LoRA、多模态与输出解析。模型中的 239 处源码位置由 `check_model.py` 逐一核对存在。

**测试台账**（[semantic_space/](semantic_space/)）。两套仓库 4,673 个测试相关文件、31,968 个测试入口的源码索引与多轮静态扫描。其中 1,293 个入口已人工核查，838 个映射到候选单元。这部分只用于检查空间缺口，不决定空间分类；静态关联也不等于已确认的验证覆盖。

本轮又对全部入口做了粗粒度路由：26,044 个入口有单一验证环节线索；其中 1,371 个还建立了被测调用结果与断言的直接源码关联。另有 191 个入口的目录领域与源码目标指向不同环节，保留为冲突待核。这些数字与 838 个已核对具体性质的入口表示不同证据层级，详见[归类进度页](review_progress/)及[批量路由说明](review_progress/README.md)。

| 材料 | 内容 |
|---|---|
| [master-review-index.tsv](semantic_space/master-review-index.tsv) | 全部入口的证据等级与下一步核查项 |
| [fast-pass](semantic_space/fast-pass-report-20261109.md) · [second-pass](semantic_space/second-pass-report-20261109.md) · [third-pass](semantic_space/third-pass-report-20261109.md) · [fourth-pass](semantic_space/fourth-pass-report-20261109.md) · [fifth-pass](semantic_space/fifth-pass-report-20261109.md) | 逐轮扫描：初步分流、产品调用到断言的关联、辅助函数、辅助返回值数据流、预期异常 |
| [bulk_call_oracle_links.py](semantic_space/bulk_call_oracle_links.py) | 调用结果与断言的批量源码关联，留出样本核查见 [SGLang](semantic_space/agent_bulk_sglang_holdout.md)、[vLLM](semantic_space/agent_vllm_bulk_holdout_review.md) |
| [agent_http_boundary.md](semantic_space/agent_http_boundary.md) | HTTP/SDK 测试的请求调用与断言线索 |

**分专题测试规格**（[docs/](docs/)）：[源码对照摘要](docs/source-comparison.md)、[流式响应](docs/stream-response.md)、[工具调用解析](docs/tool-call-parse.md)、[适配器驻留](docs/adapter-residency.md)、[缓存事件与输出解析](docs/cache-event-output-parser.md)。

## 目录

```
docs/              研究说明、分专题测试规格、示意图
source_model/      源码派生的空间模型、校验与范围查询脚本、调研记录
semantic_space/    早期候选空间、交互页面、测试索引与逐轮扫描结果
review_progress/   独立的测试归类进度页面
third_party/       源码快照说明（完整快照不随仓库发布）
```

## 复现

```bash
# 将两个固定 commit 的完整仓库放在 third_party/reference_repos/ 下
# （vllm-20261105/ 与 sglang-v0.5.14-complete/），或设置 VERIFICATION_REFERENCE_REPOS
cd source_model
python build_model.py && python check_model.py && python render_units.py
python check_model.py --changed s:mem_cache/radix_cache.py   # 改动文件影响的单元范围
```

`--changed` 先找出源码位置落在改动文件中的单元，再沿依赖找出受影响的上层单元。没有单元引用的文件会报告“范围未知”，而不是“无影响”。这是空间上的范围查询，不是测试选择方法。

## 局限

- 76 个单元覆盖代表性路径；嵌入/池化请求、各注意力后端的独立实现、量化格式细节、编译缓存、观测指标尚未展开，许多后端文件还没有单元引用。
- 错误形式依据源码推断，尚未通过故障注入确认；判定方式只给出类型，具体容差需在测试实现时按 dtype 与后端确定。
- 文档中列出的两处 SGLang 前缀键编码问题（`extra_key` 与 `lora_id` 拼接、多模态身份截断为 30 位）只来自源码阅读，尚未运行验证。
