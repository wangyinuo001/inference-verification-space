## 共享内存槽位与读者状态

| # | 共同需求 | vLLM（固定快照） | SGLang 0.5.14（固定快照） | 条件差异 |
|---|---|---|---|---|
| 1 | 被读者持有的数据不被新的写入覆盖；容量不足时不分配 | `vllm/distributed/device_communicators/shm_object_storage.py:218-318` 的 `SingleWriterShmRingBuffer` 通过物理区间和单调 ID 检查回绕、容量，再由 `is_free_fn` 决定可回收数据 | `python/sglang/srt/distributed/device_communicators/shm_broadcast.py:32-85,339-432` 的 `ShmRingBuffer` 使用固定槽、写标志与每个读者的完成标志，写者等待全部读者完成 | vLLM 对象存储分配变长区间，SGLang 消息环使用固定大小槽；不能按相同物理地址或队列索引比较 |
| 2 | 写入完成后才能交付原值，全部读者完成后才允许复用 | `vllm/distributed/device_communicators/shm_broadcast.py:732-804,920-949` 在读者完成前阻止写者复用槽位，并在发布和读者完成处使用内存栅栏 | `python/sglang/srt/distributed/device_communicators/shm_broadcast.py:339-432` 在复位读标志后设置已写标志，读者读完后标记完成 | vLLM 源码明确使用内存栅栏与可选超时；SGLang 的标志顺序由代码注释说明，验证时需要按各自的发布机制观察 |

旧候选空间的 `shared_resource.buffer_slot_integrity` 还记录对象句柄与代次等更细的义务。新单元只抽取两套实现都需要的槽位不覆盖与读者完成条件；句柄身份不由该单元替代。
