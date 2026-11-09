# 未关联入口的保守分流审查

本审查读取固定台账 `agent_unlinked_triage.tsv` 的 22,541 条入口，并与 `master-review-index.tsv` 按快照、路径、测试名和行号连接。源文件取自已固定的 vLLM 与 SGLang 快照。`agent_unlinked_routing_audit.py` 用稳定哈希在主要分组中抽出 50 条；逐条审查记录及源码位置保存在 `agent_unlinked_routing_sample.tsv`。下列数量表示**索引目前为何无法建立直接调用—判定关联**，不表示已经确定了测试的验证性质。

| 现有分组 | 入口数 | 下一步批处理 |
| --- | ---: | --- |
| 有直接判定，未识别直接产品调用 | 10,827 | 将调用目标与 fixture、输入辅助函数分开记录；分析 HTTP/SDK 请求、对象方法和常量声明 |
| 有产品调用，缺直接调用—判定边 | 6,333 | 建立同一测试中的赋值、返回值、异常上下文和断言辅助函数的链 |
| 判定在 helper 或 callback | 1,902 | 解析测试到 helper 的参数绑定和 callback 执行点 |
| 继承来的测试候选 | 1,341 | 解析 MRO，组合基类测试体与子类配置 |
| 产品调用未被链接索引解析 | 971 | 处理函数内导入、对象构造、属性方法及跨语句传递 |
| Rust 测试源码 | 630 | 使用 Rust 语法定位宏、调用和断言 |
| helper 解析缺口 | 254 | 检查装饰器、context manager、继承方法和运行时回调 |
| helper 未找到判定 | 141 | 检查 helper 实现和隐式不抛异常判定 |
| 文件级或未解析记录 | 93 | 将 fixture 数据、辅助模块、Rust 文件和真实测试定义分离 |
| 定义定位歧义 | 25 | 对照源码行，优先检查是否为继承类声明 |
| 非测试或收集辅助函数 | 24 | 核查 pytest 收集与调用者；不能凭 `test_` 前缀判为测试 |

索引中 21,990 条入口已有某种可见判定线索，551 条没有；这里的“可见”包括继承方法、Rust 宏及 helper 候选，并不等于目标调用已关联。前两组共 17,160 条，是最适合先做静态关联的部分。第一组内现有单标签路由分别为测试辅助函数 5,653、其他 2,179、fixture 注入 1,218、HTTP/SDK 客户端调用 916、本地被测对象方法 774、框架启动或外部服务 87。`expected_exception` 出现在 1,387 条入口的判定线索中，因而异常上下文应作为独立的判定链处理，不能归成“没有断言”。文件级记录有 27 个 `.rs`、24 个 `.py`、21 个 `.txt`、18 个 `.json`、2 个 `.yaml` 和 1 个 `.sh`；这些记录不能统一当作测试函数。

50 条分层抽查中，12 条需要细化或修正现有路由，8 条有直接判定但缺调用—判定边，10 条需要追踪 helper/callback，5 条需要解析继承，4 条需要 Rust 解析，3 条尚未定位判定，2 条须检查 pytest 收集，1 条是数据文件；另有 5 条现有路由与可见源码一致。这是按问题类别定额抽样，**不是**总体错误率估计。

有几处路由误分类可直接从源码核查。SGLang `test/registered/unit/constrained/test_grammar_manager.py:672` 的 `_make_req` 只负责准备请求，测试在 679 行实际调用 `mgr.process_req_with_grammar`，随后检查结果；把辅助函数当作唯一目标会漏掉产品调用。vLLM `tests/entrypoints/cohere/test_chat_v2.py:133` 通过 fixture 得到 `httpx_client`，在 136 行向 `/cohere/v2/chat` 发出流式 POST，并检查状态码与事件；这里“fixture 注入”描述客户端来源，“HTTP 端点”描述验证目标，二者应保留为不同字段。全量索引中至少 12 条标为 fixture 注入的记录，其路由证据已经明确包含 `.get/.post/.request/.stream` 等 HTTP 方法；这是可直接复查的交叉标签候选，不代表其余 fixture 路由也错。vLLM `tests/v1/kv_offload/tiering/p2p/test_manager.py:1338` 的 `mgr.shutdown()` 及其后的断言没有被“其他／未识别调用”路由捕获。相反，`tests/v1/metrics/test_histogram_buckets.py:215` 直接比较静态声明集合，本来就不需要运行时产品调用，应保留“声明性质检查”类别。

文件与继承记录也暴露了索引粒度问题。SGLang `test/registered/backends/test_flashinfer_trtllm_gen_moe_backend.py:298` 和 `test/registered/quant/test_fp8_blockwise_gemm.py:127` 指向的是子类声明，实际测试体来自基类，适合进入 MRO 队列。SGLang `sgl-kernel/python/sgl_kernel/test_utils.py:1` 是数据构造辅助模块，而 vLLM `rust/src/chat/src/renderer/deepseek_v41/fixtures/test_output_2.txt:1` 是数据文件。两者都在文件级索引中，但均不是可直接执行的测试定义。

建议把批处理结果分成互不混淆的字段：**入口来源**（fixture、helper、直接构造）、**实际执行边界**（Python 对象方法、HTTP/SDK、Rust 调用、外部进程）、**判定形式**（值比较、异常、状态、隐式不抛异常）和**关联证据**（源码行、赋值链、helper 链、运行记录）。静态规则先处理可定位的赋值和调用链，再处理 helper、继承与语言边界；不能闭合的链输出待核查原因，不推断验证性质。这样可以批量推进索引关联，同时为后续逐项语义审查留下可追溯证据。
