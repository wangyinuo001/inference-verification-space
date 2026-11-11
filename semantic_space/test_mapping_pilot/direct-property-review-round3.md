# 普通测试断言审查与空间对照（第三轮）

固定源码：vLLM `76a0ccc209343971c6f4b56f6380a76202428385`；SGLang `49e384ce9d304648e9959666ecb8ce8cd98d0deb`。本轮只审阅源码与局部断言，没有运行模型、服务或 GPU 测试，也没有查看保留评估集。

本轮净增 **121** 条绿色入口，进度页从 1,893 增至 **2,014 / 31,968**。其中 81 条是本轮直接回读测试函数后登记，40 条是此前已有 `source_property_confirmed` 源码审查、但尚未对照候选单元的入口。三份交叉映射及已有逐项审查按 `(snapshot, path, test_name, line)` 去重；本轮 121 条与既有绿色入口不重合。

| 来源 | 新增 | 已核对的局部性质 | 空间边界 |
| --- | ---: | --- | --- |
| vLLM `tests/test_config.py` | 45 | 配置默认值、派生字段、模型类型、运行模式、能力约束及非法配置拒绝 | 只检查配置对象和局部能力判断，不证明实际启动或运行 |
| vLLM `tests/test_envs.py` | 28 | 标量、列表、集合环境选项的默认、大小写、空项、重复项和可调用选项解析 | 只检查环境选项解析，不证明所有服务配置最终生效 |
| SGLang `test_mem_cache_utils.py` | 8 | 淘汰策略名称选择相应策略类；非法名称报错 | 只检查策略构造，不证明运行时淘汰顺序 |
| SGLang `test_serving_chat.py` | 22 | 推理开关与历史工具输入绑定、工具 schema、required 解析和流式参数补发 | 只检查测试函数列出的入口/输出局部关系 |
| SGLang `test_serving_responses.py` | 10 | Responses 输入归一化、非法输入和工具输出项 | 只检查 mock/局部请求转换，不证明端到端交付 |
| SGLang `test_network_address.py` | 8 | 分布式启动地址与服务 URL 的 IPv6 派生及非法配置拒绝 | 不将一般网络解析/格式化辅助测试算入这些配置单元 |

直接审查的 81 条保存在 [`direct-property-crosswalk.tsv`](direct-property-crosswalk.tsv)，逐条有源码断言行、性质和适用边界；[`build_direct_property_crosswalk.py`](build_direct_property_crosswalk.py) 在固定源码快照上重新定位断言并验证测试定义行。原先已确认的 40 条由 [`build_confirmed_crosswalk.py`](build_confirmed_crosswalk.py) 从 `master-review-index.tsv` 的 `source_property_confirmed` 记录读取原始性质与断言位置。交互页同步展示这两批普通测试的局部检查。

审查时还读了 SGLang 同一缓存文件中的哈希转换、分页哈希和拆分测试。它们的断言虽然明确，却不等于当前 `cache.hash_order_sensitivity` 或 `cache.hash_bigram_equivalence` 的性质，因此未计入绿色。vLLM Cohere renderer 文件有 54 条直接关联；初读其前半部分后发现当前聊天模板 token 化单元不能直接承载这些渲染配置转换检查，因此本轮未将该文件标绿。静态关联、源码已读和进入候选单元是三件不同的事。
