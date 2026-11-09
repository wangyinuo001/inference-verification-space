# 缓存事件与输出解析的独立验证规格

本规格依据固定版本的 vLLM 与 SGLang 源码及已审阅测试，定义验证仓库可以自行实现的检查。测试输入、状态、观察值与判定规则由性质确定；现有测试仅提供实现依据。逐项审阅证据保存在 `semantic_space/` 下三个审查文件族，并收录在总台账中。

## 1. 缓存淘汰策略

输入包括候选缓存项、最近访问时间、创建时间、命中次数、显式优先级和策略配置。对于分段 LRU，还应覆盖命中次数低于、等于和高于保护阈值的情况。先检查有效策略与默认或显式阈值，再检查策略比较键及由它决定的候选次序。实际淘汰另需观察可淘汰集合、受保护项、释放结果和可用容量，不能用单独的排序键测试代替。SGLang 的 `get_priority` 测试使用受控模拟节点；vLLM 的空闲块队列以块为单位维护 LRU，等时间下还考虑链尾位置。这些差异作为策略及状态条件保留。

源码：[SGLang 策略](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/evict_policy.py)、[SGLang 实际淘汰](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py)、[vLLM 空闲块队列](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_utils.py)。

## 2. 缓存事件

构造缓存存储、移除、重试、提升、跨层迁移及清空的状态序列，记录源请求 token、块哈希链、KV 组、存储层、位置与所有权。每一步分别比较发布的事件类型、顺序和载荷与真实缓存变化是否对应。部分尾块和不同 KV 组需要按其物理块粒度解释；关闭自描述模式时，检查占位事件的字段规则。对于待完成写入、同键重试和多层驻留，检查事件元数据在需要时保留，在最后相关状态释放或重置后清除。分层背压配置另比较顶层默认值与单层覆盖的有效结果。SGLang 使用 radix 节点生成 store/remove/clear 事件，vLLM 还处理 pending 操作及跨层事件；共同义务是状态变化与对外事件对应，具体字段由条件限定。

源码：[SGLang 事件生成](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/events.py)、[vLLM 事件跟踪](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/kv_transfer/kv_connector/v1/offloading/events.py)。

## 3. 结构化输出解析

以声明的输出语法、通道和分块方式构造完整与不完整生成文本。词法检查比较结构标记的类型和原文字符区间，以及跨块保留的未完成前缀；语义检查比较推理、正文和工具调用片段产生的类别、内容与顺序；增量检查比较每块已输出内容和剩余缓冲区，并在终止后比较合并内容；格式检查覆盖结构化与文本回退两种输入及延迟选择。SGLang 的部分分块测试只比较拼接后的内容，未来独立测试若要保证事件边界和顺序，需要分别添加断言。vLLM 的 Harmony 测试直接检查解析出的 reasoning、content、tool calls 和增量结果，具体输出表示不同，性质可以按通道内容和分块关系统一。

源码：[SGLang 解析器](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/parser/harmony_parser.py)、[vLLM 解析器](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/parser/harmony.py)。

## 4. 判定边界与补测

审阅的 SGLang Harmony 文件有一项断言仅检查 `len(events) >= 0`，不能判断独立结构标记是否被过滤；另有一项仅确认返回对象为列表。独立测试需要检查事件内容及剩余缓冲区。五项只检查内部记录或策略对象初始化字段，可作为实现层证据，暂不形成请求行为义务。事件内容、缓存真实释放、配置解析分别保留为不同验证结果，供后续空间剪枝使用。
