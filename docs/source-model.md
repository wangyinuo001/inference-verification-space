# 推理系统 Verification Space：源码调研与分层模型

2026年11月9日

本文从 vLLM 与 SGLang 的源码出发，整理推理系统需要验证的功能、行为与性质，并据此提出分层 Verification Space。分析对象是两个固定快照：vLLM `76a0ccc`（[`vllm-20261105`](https://github.com/vllm-project/vllm/tree/76a0ccc209343971c6f4b56f6380a76202428385)）与 SGLang 0.5.14（[`sglang-v0.5.14-complete`](https://github.com/sgl-project/sglang/tree/49e384ce9d304648e9959666ecb8ce8cd98d0deb)）。结构化模型与校验脚本在 [`source_model/`](../source_model/)，逐单元列表见 [`units.md`](../source_model/units.md)。

本文记录 11 月 9 日的 71 单元版本。最新结构化模型补充了启动配置、缓存淘汰次序及共享槽位等性质，共 76 单元；以 [`space_model.json`](../source_model/space_model.json) 和 [`units.md`](../source_model/units.md) 为准。

## 1. 调研方式

此前的工作从现有测试出发：逐条阅读测试入口，判断它检查了什么，再归入空间。这条路线要处理三万余个入口，进展慢；而且得到的是“现有测试检查了什么”，不是“系统需要保证什么”。

本轮改为从源码出发。沿一次请求的执行路径分成六条调研线：请求输入输出、调度与生命周期、KV 缓存管理、模型执行与采样、分布式与权重加载、跨流程功能（推测解码、LoRA、多模态、输出解析）。每条线在两个框架中找到对应实现，并为每个机制记录：

- 两边的源码位置（文件、行号、函数）；
- 该机制持有或修改的状态；
- 依据代码可以推断的错误形式；
- 与框架无关的可验证性质及判定方式（参考实现、变形关系或统计检验）；
- 性质依赖的输入、配置与运行状态；
- 两个框架间必须保留为验证条件的语义差异。

六条线共得到 99 条记录，原始报告保存在 [`traces/`](../source_model/traces/)。调研报告中共 328 处 `文件:行号` 引用；建模时按确切路径重新核对，发现 3 处路径或行号有误并已改正，模型目前引用的 222 处源码位置都由 [`check_model.py`](../source_model/check_model.py) 核对。决定空间结构的 7 条关键差异另行人工回读源码确认（见 §6）。现有测试不参与分类，只在后续用于检查缺口。

## 2. 两个框架的执行路径对照

| 阶段 | vLLM | SGLang | 主要可验证内容 |
|---|---|---|---|
| S1 入口与规范化 | OpenAI serving → `AsyncLLM`（v1/engine/async_llm.py） | serving_chat → `TokenizerManager` | 合法请求恰好转为一个内部请求；非法请求在调度前被拒 |
| S2 渲染与分词 | renderer、`render_chat_request` | `_apply_jinja_template` / conversation 模板 | 模板边界、文本与 token 一致、多模态占位对齐 |
| S3 调度 | `Scheduler.schedule`、`RequestStatus` 枚举 | `Scheduler.get_next_batch_to_run`、`SchedulePolicy` | 预算、顺序、分块、抢占、取消、结束 |
| S4 跨步状态 | `BlockPool` + `KVCacheManager`（块哈希链） | `RadixCache` + `ReqToTokenPool` + 分页分配器 | 守恒、所有权、隔离、驱逐、释放 |
| S5 步绑定 | `GPUModelRunner._prepare_inputs`、注意力元数据构建器 | `ForwardBatch.init_new`、`ForwardMode`、CUDA 图 runner | 位置、行边界、写入槽位、图填充 |
| S6 数值计算 | model_executor/layers、attention backends | layers、attention backends | 与参考实现在容差内一致 |
| S7 采样 | v1/sample/sampler.py、rejection_sampler | layers/sampler.py、speculative/ | 处理顺序、过滤、种子、logprob、推测接受 |
| S8 输出组装 | `OutputProcessor`、`IncrementalDetokenizer` | `DetokenizerManager`、TokenizerManager 回包 | 增量解码、停止、流式/非流式一致、用量、结束原因 |
| S9 拓扑 | parallel_state、linear.weight_loader、kv_transfer | parallel_state、linear.py、disaggregation/、data_parallel_controller | 组划分、分片加载、通信、并行等价、PD 传输 |

两个框架的进程边界不同：vLLM 将 API Server、Engine Core、Worker 分开；SGLang 由 TokenizerManager、Scheduler、DetokenizerManager 三个进程组成。进程边界决定状态归属，但不改变需要验证的功能语义，所以这里按功能阶段对齐，不按进程对齐。

## 3. 统一抽象：验证单元

每个验证单元描述一条框架无关的性质，包含：性质、判定方式、适用条件、依赖的下层单元、组合缺口、对应的执行扰动、两边源码位置、框架差异。共 71 个单元，其中 64 个两边都有实现（common），7 个只在部分架构或开关下出现（extension：混合模型状态、多级缓存、缓存事件、专家并行、PD 分离、在线权重更新、卸载恢复）。

去掉类名与接口后，99 条记录中的性质归为六类：

| 类型 | 含义 | 例子 |
|---|---|---|
| 数值保真 value | 与参考实现在声明容差内一致 | 注意力后端对 FP32 参考；TP 分片拼接等于检查点 |
| 确定性契约 contract | 输入到内部表示/输出的映射满足规格 | 前缀命中上界 hit ≤ n−1；行边界为累积长度 |
| 资源守恒与所有权 resource | 获取/共享/释放恰好一次，不同请求不串用 | 已分配+空闲=总量；不写入被他人引用的块 |
| 协议与生命周期 protocol | 状态迁移、顺序、恰好一次、终止 | 取消幂等且最终清理；每请求恰好一个终止输出 |
| 执行等价 equivalence | 请求输出在某种执行扰动下不变 | 缓存命中与冷启动、分块与整体、图与即时 |
| 分布 distribution | 随机采样服从目标分布 | 推测采样保持目标分布 |

其中**执行等价**最值得单独处理。推理系统的大量优化（前缀缓存、分块 prefill、抢占、重叠调度、CUDA 图、推测解码、并行、PD 分离、编码缓存、卸载）有一个共同点：它们在不改变请求语义的前提下改变执行方式。两个框架的实现机制差别很大，但要求相同：请求结果在该扰动下不变。因此空间中用一个显式维度记录“扰动”（15 种），每个等价单元对应一种扰动，并依赖实现该扰动的机制的局部性质。

## 4. 分层依据：判定边界

### 4.1 层次定义

层次按**判定一条性质所需的最小真实组件集合**划分，而不是按代码所在目录划分：

| 层 | 名称 | 需要真实运行的部分 | 可替换为伪实现的部分 |
|---|---|---|---|
| B0 | 纯函数 | 单个函数或算子 | 其余全部 |
| B1 | 单组件状态机 | 一个带状态的组件（分配器、基数树、调度器、增量解码器、解析器） | 模型前向、设备 |
| B2 | 单步引擎 | 调度结果、状态与一次真实前向（含图与填充） | 多步与多请求语义 |
| B3 | 单请求多步 | 一个请求在单个实例上的完整生成 | 并发干扰 |
| B4 | 多请求并发 | 同一实例上多个请求的批组成与相互影响 | 多进程拓扑 |
| B5 | 多进程拓扑 | TP/PP/DP/EP/PD 进程与通信 | 无 |

这样分层有三个用途：层越高，测试越贵；一条性质落在哪层，独立仓库就该在哪层实现它（例如 B1 性质可用伪模型在 CPU 上跑）；上层失败时，可以先查它依赖的下层单元。

### 4.2 为什么不按算子—模块—模型—系统划分

四层目录按代码位置分类。源码表明，同一个组件里的不同性质，判定所需的边界可能相差很远：

- **前缀缓存**属于缓存模块。但“键能区分 LoRA、多模态和 salt”只需基数树或块哈希本身（B1）；“命中后输出与冷启动相同”则需要单请求完整生成（B3），因为复用的 KV 是在另一批组成、另一内核路径下算出的。
- **进程组划分与分片加载**通常归入“分布式系统”层。但给定 world size 和并行配置后，组成员是纯函数（B0，[vLLM parallel_state.py:2085-2177](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/parallel_state.py#L2085-L2177)）；分片拼接回检查点只需在单进程中模拟各秩加载（B1），不需要多卡。
- **采样种子**属于采样算子。但“同种子输出不随同批其他请求变化”至少需要两个请求同批执行（B4）。
- **CUDA 图填充**一般归到执行器模块，其性质需要一次真实前向及哨兵 KV（B2）。

模型统计也说明阶段与层次不一致：

```
              B0  B1  B2  B3  B4  B5
S4_state       .  11   4   2   .   .
S8_output      2   3   .   6   .   .
S9_topology    1   1   .   .   .   5
```

S4 的 17 个单元跨 B1—B3，S9 横跨 B0、B1、B5。若按代码位置分层，同一层会混入成本与判定方式完全不同的检查。

### 4.3 维度之间的关系

- **阶段**（9）与**判定边界**（6）互相独立，见上表。
- **性质类型**与边界相关但不确定：资源类集中在 B1/B2（16/16），执行等价集中在 B3 及以上（12/16），数值类主要在 B0。这符合直觉：所有权可以在单组件内判定，等价关系必须观察请求结果。
- **条件维度**（28）不是自由组合。源码中已经存在硬约束，例如 SGLang HiRadixCache 不支持缓存事件（[radix_cache_cpp.py:45-47](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache_cpp.py#L45-L47)）；推测解码与 DP attention 同开时 SGLang 不混合 prefill/decode；prompt logprob 会收紧前缀命中上界。这些约束单独列出（K1—K5），后续剪枝必须先排除不可行组合。

## 5. 跨层依赖与组合缺口

一个上层单元的依赖是这样一组下层单元：只要其中任何一个不成立，上层性质就失去保障。但反过来，所有依赖都成立也不足以推出上层。对每个请求级等价单元，模型都写明**组合缺口**：下层性质无法覆盖、只能在本层观察的部分。15 个单元有明确缺口，例如：

| 上层单元 | 依赖的下层单元 | 组合缺口 |
|---|---|---|
| eq.prefix_hit（B3） | kv.key_completeness、kv.hit_extent、kv.no_shared_write、kv.ref_protect、sched.prefix_bound、bind.slot_mapping、op.attention | 键与映射正确只保证读到同一输入算出的 KV；这些 KV 在另一批组成、另一内核路径下算出，数值能否使结果一致只能在请求级观察 |
| eq.preemption（B3） | sched.preempt_state、kv.release_terminal、eq.prefix_hit | 调度计数正确时，异步在途输出仍可能被重复使用（vLLM [request.py:166-185](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L166-L185)） |
| eq.batch_invariance（B4） | bind.slot_reset、bind.graph_padding、samp.processor_order、op.moe | 批大小改变归约顺序与内核选择，算子各自满足容差也推不出批不变 |
| bind.graph_padding（B2） | bind.boundaries、bind.slot_mapping | 逐行元数据正确，不排除填充行经共享缓冲区写入 KV |
| out.grammar_valid（B3） | grammar.state、samp.grammar_mask | 推测 token 被拒后语法状态要回滚（vLLM [backend_types.py:64-72](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/structured_output/backend_types.py#L64-L72)），单步掩码正确不保证整段合法 |
| out.stream_parity（B3） | out.detok、out.stop、out.parse_incremental、out.usage | 两个框架的流式与非流式是两条独立组装路径 |

组合缺口回答了“哪些上层单元不能因为下层都通过就省掉”，也是后续剪枝最需要保留的部分。

## 6. 必须保留的框架差异

抽象去掉了类与接口，但下列差异会改变执行路径或可测条件，在空间中保留为条件取值或扩展单元。前 7 条已人工回读源码确认。

| 差异 | vLLM | SGLang | 在空间中的处理 |
|---|---|---|---|
| 前缀命中粒度 | 满块命中，非对齐部分命中时预留写时复制块 | 基数树按 token 匹配，再按 page_size 截断（[radix_cache.py:395-431](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L395-L431)） | 条件 `cache_granularity`、`page_size` |
| 命中上界 | min(n−1, prompt_logprob_start, 0 若跳过缓存)（[request.py:352-377](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L352-L377)） | n−1，返回 logprob 时再取 min(·, logprob_start_len)（[schedule_batch.py:1223-1228](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L1223-L1228)） | 共同契约 sched.prefix_bound，条件 `hit_bound` |
| 前缀键的身份编码 | 结构化 extra-key：LoRA、多模态哈希、首块 cache_salt、prompt embeds（[kv_cache_utils.py:594-676](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_utils.py#L594-L676)） | `extra_key` 字符串后直接拼接 `lora_id`（[schedule_batch.py:776-783](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L776-L783)）；多模态占位 token 替换为 `偏移 + hash % 2^30`（[schedule_batch.py:144-146](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L144-L146)） | kv.key_completeness 需按两种编码分别构造碰撞用例 |
| 请求生命周期表示 | `RequestStatus` 枚举，PREEMPTED 之后都算结束（[request.py:422-446](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L422-L446)） | 队列/批成员关系与 `finished_reason` | 检查重新入队与清理语义，不比较状态值 |
| 分块中取消 | EngineCore 在步边界处理取消队列 | 记录到 `_pending_chunked_abort_req` 延迟处理（[scheduler.py:2470-2488](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2470-L2488)） | 只要求最终清理与无后续输出 |
| 多个停止串 | 在增量解码中处理 | 只处理一个匹配串（源码 TODO），并假定最后一个 token 是停止 token（[detokenizer_manager.py:175-199](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/detokenizer_manager.py#L175-L199)） | 条件 `stop_config=multi_stop_string` |
| 缓存事件 | 块存入/移除事件携带组、介质、extra keys | HiRadixCache 断言不支持 | 扩展单元 kv.events + 约束 K1 |
| 调度顺序 | 先运行请求，后等待请求 | 有新 prefill 批时先跑 prefill | 条件 `schedule_order` |
| logprob 来源 | 显式区分 raw 与处理后 | 环境变量选择是否返回原始值 | 条件 `logprob_mode` |
| 推理解析 | 基于 token ID | 基于文本片段 | 两者都要枚举切分点 |
| 推测解码形态 | draft model、n-gram 等 proposer | EAGLE 链/树、外部 n-gram 语料 | 条件 `spec_method` |

第三条中的两点只来自源码阅读，尚未运行验证，不能当作已确认缺陷：（1）用户提供的 `extra_key` 若恰好等于另一请求的 salt 加其 `lora_id`，两者的前缀键相同，而 `lora_id` 是由名称和路径确定的 uuid5；（2）多模态身份在前缀键中只保留 30 位。它们说明 kv.key_completeness 的测试需要针对具体编码构造反例，而不能只检查“不同图片不命中”。

## 7. 对后续剪枝的支持

模型提供一个范围查询（[`check_model.py --changed`](../source_model/check_model.py)）：先找出源码位置落在改动文件中的单元，再沿依赖反向找出所有上层单元。示例：

- 改动 SGLang `mem_cache/radix_cache.py`：直接命中 10 个单元（9 个 B1/B2 缓存单元与 eq.prefix_hit），向上影响 10 个单元，包括分块、抢占、重叠调度、卸载恢复、批不变、LoRA 隔离与 PD 传输等价（[scope-radix_cache.txt](../source_model/scope-radix_cache.txt)）。
- 改动 vLLM `v1/engine/detokenizer.py` 与 `v1/attention/backends/flash_attn.py`：前者影响 out.detok、out.stop 及其上层的结束原因和流式一致；后者目前没有单元引用，查询输出“范围未知”而不是空集（[scope-detok.txt](../source_model/scope-detok.txt)）。这说明各后端实现文件的映射还需补齐。

这个查询只确定“哪些验证需求可能受影响”，不是测试选择方法。真正剪枝还需要用单元的条件维度和约束进一步收窄，并决定组合缺口单元能否用较低层检查替代。这些都留到空间稳定之后。

## 8. 当前局限与下一步

- 单元来自六条调研线的 99 条记录，覆盖了代表性路径；嵌入/池化请求、量化格式细节、各注意力后端的独立实现、编译缓存、观测与指标尚未展开。
- 源码位置是行区间级证据；“错误形式”是依据代码推断，尚未注入验证。
- 判定方式和容差策略只给出类型，具体容差需在独立测试仓库中按 dtype 和后端确定。
- 下一步：（1）补齐扩展阶段和后端实现文件的单元映射，使范围查询不再出现大面积“未知”；（2）对 kv.key_completeness、out.stop、eq.prefix_hit、eq.batch_invariance 四个单元，在两个框架上各写一个最小可运行检查，检验单元定义能否直接转成独立测试；（3）用已有测试台账抽样核对，只用来发现空间缺口，不再逐条审查。
