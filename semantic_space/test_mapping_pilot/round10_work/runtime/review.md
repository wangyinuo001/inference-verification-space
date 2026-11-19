# 本批源码归类说明
已核对 6 个文件的测试函数、直接断言与相关辅助函数，新增 229 个入口映射，排除已确认的 4 个入口。此次没有执行上游测试。
CPU offload manager 55 项检查存储接收、LRU/ARC 队列、引用和淘汰、事件与统计；factory 40 项检查配置生效、容量换算、worker 构造参数和拒绝条件；backpressure 50 项检查背压状态机、限流决定及统计。上述状态或调用断言不等同于真实缓存数据传输。
partial prefix cache 51 项检查混合 attention/Mamba 缓存块、hash 边界、checkpoint 保留、CoW 计划和延迟释放。其中 KVCacheBlockCopy 只是要执行的复制计划；归类不声称 GPU 已完成数据复制。
事件文件 13 项使用真实本地 ZMQ socket。辅助类虽然名为 MockSubscriber，内部实际创建 SUB/DEALER，故不能仅按名字视为 mock。高流量断言只要求至少 90% 的消息在时间窗口内到达；没有归为无损交付。
打包传输文件新增 20 项。mock broadcast roundtrip 只核对打包表示和本地复制；独立子进程 IPC roundtrip 才使用真实 CUDA IPC。异步消费者测试用 GPU sleep、事件、地址和 sentinel 验证等待与存储生命周期，属于独立性质。仅检查 metadata/空输入的入口未写成数值 roundtrip。
新增单元建议在 proposed-units.json，需与统一空间合并检查后再确认对应 93 项。其余 136 项使用现有单元。缓存统计、接收去重、背压、传输打包和异步缓冲不能统称缓存正确性；它们的输入与判定对象不同。
每项的源码片段、范围、SHA256、全部断言位置在 evidence.json。每个 crosswalk 的 oracle_source 指向实际断言（或已查看的 verify/check 辅助函数）。同一性质下的概括描述不表示单个入口检查该性质的全部条件，具体范围以其条件与断言为准。
未纳入两项：test_shared_base 仅检查类继承，test_null_publisher 仅调用无异常；不把它们提升为系统功能验证已完成。
