# vLLM 运行与传输检查源码核对
本批核对 8 个文件，形成 235 项新增逐入口映射，排除索引中已确认的 6 项。每项保留独立的调用、断言、参数化、定义内容和 SHA256；未执行这些测试。语义归属依据实际计算对象与断言确定，AST 用来导出证据。

Mooncake coordinator 的 39 项检查针对人工哈希集合上的命中长度、滑动窗口与多缓存组装载/保存掩码。它们未执行 KV 传输。Mooncake connector 的 48 项检查区分了包装调用转发、事件聚合、异步查询、控制消息与真实 BlockPool 引用计数。查询使用真实线程池和人工返回的 socket 数据，不能据此标记实际网络数据保持。

NIXL 描述符几何的 28 项检查以真实 worker 构建地址关系，但 _RecordingNixl.transfer 为空操作。大多数检查的是地址、长度与 token 对应；4 项 sweep 经 _run_hetero_case 检查辅助函数断言，证据另行链接。test_real_model_layout_pull_copies_every_page 用 ctypes.memmove 执行记录描述符，在 CPU 内存中比较每页并检查请求外字节；其“真实 model layout”指缓存布局，并未执行模型或真实 RDMA。

Worker 的 75 项检查区分微批/PCP 布局切片、buffer 所有权、图执行选择、缓存物理布局与内容复制。DP 决策测试用预置矩阵替代 collective。CUDA toy-model 两项用明确算式比较执行结果；异常测试仅保证测试所构造失败顺序的返回，源码明确仍有其他时序可能挂起。未把这些入口当作完整模型或真实多进程通信正确性。

传输统计检查以人工观测值和真实聚合/Prometheus 逻辑为判据；它们验证计数、快照、重置、标签及序列化，不是实际性能测试。MultiConnector 集成案例运行模型并比较两个本地目录内容；其余大多使用 mock 子组件，检查转发参数与结果归属。

有两项测试在函数内复制 LookupKeyServer RESET handler 分支，断言只约束这段测试内逻辑，并未调用生产 process_request。这两项记录到 excluded-reviewed.json，不计入生产路径已确认映射。无显式断言的配置 smoke 入口也未加入本批。

提出 8 个细化验证单元，均明确 framework_scope=[vllm]，保留产品源码锚点与哈希。它们暂为扩展建议，需主流程合并空间后再导入相关映射。它们不宣称已获得 SGLang 对应证据。
