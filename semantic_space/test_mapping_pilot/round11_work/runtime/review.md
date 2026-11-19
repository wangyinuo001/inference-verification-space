# 运行与缓存测试源码核对
本批核对 7 个文件、282 个尚未确认的源码级测试入口。未执行 GPU、上游测试或导入框架。crosswalk 的性质限定为单元名称和该入口实际断言，避免把同一组其他测试的检查写成此项保证；oracle_source 为可直接解析的 path:line，其余断言、调用、源码片段和辅助函数保存在 evidence.json。
EC session 的数据、传输和连接接口使用 MagicMock；它检查 ACK/NACK、版本与会话身份、超时隔离、retry/tombstone 及 release 调用。EmbeddingCache 文件是纯整数账本，真实线程测试检查操作无异常与最终引用状态，不检查 embedding 数值。两者不归为模型向量正确性。
Offload metrics 用人工统计记录检查累计、归约、label 和 metric 调用，Prometheus 部分使用 _FakeMetric。这里未测服务性能或真实带宽。Object store 使用 MockNixlAgent，以对象键集合和 PROC/DONE 模拟作业；名称为 store_then_load_roundtrip 的入口只核对 JobResult 和时间字段，没有值比较，不归为传输内容完整性。
FS tier 使用临时文件与实际文件 IO。含 torch.allclose/f.read 比较的入口关联内容完整性；其余分别检查存在性、失败结果、事件和线程池关闭。异步文件探测失败后 per-request verdict 需要与文件是否仍存在分别判定。
Tiering manager 检查分层键晋升/下沉、共享键去重、请求级延迟清理及 pin；ExampleSecondaryTierManager 是内存里的键/作业示例实现。它对流程与管理状态有直接断言，没有远端实际数据证据。
Worker utils 按每项实际断言拆成布局、层绑定、主机注册、镜像次序与角色、复制内容、容量和统计。mock stream 的 wait/record 调用说明依赖提交，不声称完整 GPU 并发执行已验证；真实张量读取比较另外记录。
新增单元建议 7 个在 proposed-units.json，均包含源码位置、条件维度、观察量和判定范围。尚未纳入两项仅执行无显式断言的测试：EC cancel-valid 与 OBJ shutdown-idempotent。它们可记录为烟测，但本批不冒充性质已核对。
