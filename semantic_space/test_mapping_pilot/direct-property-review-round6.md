# 调度、缓存与 KV 传输测试的源码审查（第六轮）

本轮使用固定的 vLLM 源码快照 `76a0ccc209343971c6f4b56f6380a76202428385`，审查测试输入、断言和适用的候选空间单元。进度页由 **3,158** 增至 **4,264 / 31,968**，净增 **1,106 条**有显式源码检查的测试入口。它们分布在调度、前缀缓存、KV 布局与哈希、Mooncake、NIXL、分层卸载及 P2P 会话。没有运行这些测试、GPU 模型或保留评估集。

| 固定源码文件 | 净增 | 局部检查重点 |
| --- | ---: | --- |
| `tests/v1/core/test_scheduler.py` | 131 | 请求准入、预算、抢占、分段 Prefill、编码器缓存和恢复；未把只报告统计但无对应候选单元的测试强行归入调度进度 |
| `tests/v1/core/test_prefix_caching.py` | 128 | 命中长度、缓存块状态、HiSparse 页驻留、淘汰和事件；未把缓存状态检查说成完整模型结果等价 |
| `tests/v1/core/test_kv_cache_utils.py` | 67 | 哈希隔离、块队列、缓存规格/布局、命中统计与配置校验 |
| `test_mooncake_store_worker.py` | 170 | Store TP 命名空间、分片/组投影、发送和接收边界、失败与资源生命周期；跳过无对应性质的日志与指标检查 |
| `offloading_connector/test_scheduler.py` | 155 | 外部命中、滑动窗口、卸载键、待完成传输及存储围栏；延迟计时检查未硬归入命中数量 |
| `p2p/test_sessions.py` | 144 | 握手、查询、传输完成、取消、断连和在途资源；无显式断言的入口未计入 |
| `p2p/test_manager.py` | 114 | 请求绑定、会话管理、端点、关闭等待和保留传输；修正了将 `reports` 误识别为端口 `port` 的风险，跳过无对应单元的纯延迟指标 |
| `test_nixl_connector.py` | 74 | KV 层/分片描述、握手失败与恢复、注册和资源清理；跳过纯日志与指标检查 |
| `test_nixl_push_connector.py` | 71 | Push 注册、通知、失效租约、分片配对和失败门控 |
| `test_mooncake_store_scheduler.py` | 52 | 命中边界、保存/加载时机、组投影、块固定与多 rank 完成 |

[`build_core_direct_cases.py`](build_core_direct_cases.py) 固定这十份文件的 SHA-256 和每份文件的选中数量，并按具体测试名与检查内容将局部性质对应到现有单元；它显式排除旧台账已绿的入口、无检查语句的测试，以及缺少合适对应单元的统计/日志测试。[`direct-property-crosswalk.tsv`](direct-property-crosswalk.tsv) 保存每条的测试定义行、直接断言、断言源码行和适用条件。映射构建器与进度页检查键不重复、源码行一致且单元存在。

**绿色的边界：**这里确认的是固定源码测试定义中的局部状态或协议性质。KV 分片描述、缓存事件和调度结果的断言不能替代真实张量内容、完整模型数值或端到端请求验证；测试也未在本轮执行。
