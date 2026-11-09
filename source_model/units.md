# 验证单元表

由 `render_units.py` 从 `space_model.json` 生成，请勿手工编辑。源码链接指向固定 commit：vLLM `76a0ccc`、SGLang `49e384c`。

| 层 | 名称 | 需要真实运行的部分 | 单元数 |
|---|---|---|---|
| B0 | 纯函数 | 单个函数/算子，无系统状态 | 12 |
| B1 | 单组件状态机 | 一个带状态的组件（分配器、基数树、调度器、增量解码器、解析器） | 26 |
| B2 | 单步引擎 | 调度结果+状态+一次真实前向（含图/填充） | 11 |
| B3 | 单请求多步 | 单引擎实例上一个请求的完整生成 | 13 |
| B4 | 多请求并发 | 同一实例上多个请求的批组成与相互干扰 | 2 |
| B5 | 多进程拓扑 | TP/PP/DP/EP/PD 多进程与通信 | 5 |

## B0 纯函数

### `op.attention`

各后端在 prefill/decode/mixed 与分页布局下，注意力输出与数值稳定参考实现在容差内一致。

- 阶段：注意力、线性/量化、归一化、MoE 等数值计算；类型：value
- 判定：FP32 参考注意力；容差按 dtype 声明。
- 条件：`attention_backend`, `dtype_quant`, `forward_mode`, `page_size`, `window`
- vLLM：[v1/attention/backend.py:386-407](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/attention/backend.py#L386-L407)；[v1/attention/backend.py:598-737](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/attention/backend.py#L598-L737)
- SGLang：[model_executor/forward_batch_info.py:438-501](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L438-L501)
- 调研记录：exec#6

### `op.layers`

归一化、旋转位置、线性与量化层输出与 FP32 参考在逐层容差内一致；不应改动的残差不被原地修改。

- 阶段：注意力、线性/量化、归一化、MoE 等数值计算；类型：value
- 判定：逐层参考 + 输入张量快照比较。
- 条件：`dtype_quant`
- vLLM：[model_executor/layers/linear.py:407-437](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/model_executor/layers/linear.py#L407-L437)
- SGLang：[layers/linear.py:190-277](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/linear.py#L190-L277)
- 调研记录：exec#10

### `op.lora_apply`

带适配器的层输出等于 W + scale·B·A 的显式参考；批内按请求选择正确的适配器。

- 阶段：注意力、线性/量化、归一化、MoE 等数值计算；类型：value
- 判定：逐层合并权重参考。
- 条件：`lora`, `dtype_quant`
- vLLM：[lora/layers/base_linear.py:162-220](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/lora/layers/base_linear.py#L162-L220)
- SGLang：[lora/layers.py:48-130](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/lora/layers.py#L48-L130)
- 调研记录：ext#7

### `op.moe`

固定路由 logits 下，专家选择与输出等于参考实现；填充行不占用专家容量。

- 阶段：注意力、线性/量化、归一化、MoE 等数值计算；类型：value
- 判定：参考 MoE；填充行注入极端值检查真实行不变。
- 条件：`dtype_quant`, `batch_shape`
- vLLM：[v1/worker/gpu_model_runner.py:700-850](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L700-L850)
- SGLang：[layers/moe/ep_moe/layer.py:181-231](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/moe/ep_moe/layer.py#L181-L231)
- 调研记录：exec#11

### `samp.filter`

贪心为精确 argmax；top-k/top-p/min-p 的支撑集与参考一致；过滤后重新归一化。

- 阶段：logits 处理、采样、logprob、推测验收；类型：distribution
- 判定：支撑集精确比较 + 大样本频率检验。
- 条件：`sampling`, `attention_backend`
- vLLM：[v1/sample/sampler.py:308-368](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/sample/sampler.py#L308-L368)
- SGLang：[layers/sampler.py:214-316](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/sampler.py#L214-L316)
- 调研记录：exec#15

### `samp.grammar_mask`

非法 token 的 logits 为 -inf；采样 token 一定被请求的语法接受。

- 阶段：logits 处理、采样、logprob、推测验收；类型：contract
- 判定：语法解析器参考 + 掩码检查。
- 条件：`structured_output`
- vLLM：[v1/structured_output/utils.py:101-194](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/structured_output/utils.py#L101-L194)；[v1/worker/gpu/structured_outputs.py:33-97](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu/structured_outputs.py#L33-L97)
- SGLang：[constrained/base_grammar_backend.py:69-80](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/constrained/base_grammar_backend.py#L69-L80)
- 调研记录：exec#18

### `samp.logprob`

返回的 logprob 与所选模式（原始/处理后）一致，与输出 token 一一对齐，top-k 数量符合请求。

- 阶段：logits 处理、采样、logprob、推测验收；类型：contract
- 判定：原始 log_softmax 参考 + 逐 token 对齐检查。
- 条件：`logprob_mode`
- 框架差异：vLLM 显式区分 raw_logprobs/raw_logits 并在处理器前捕获；SGLang 由环境变量选择返回原始 logprob。
- vLLM：[v1/sample/sampler.py:129-180](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/sample/sampler.py#L129-L180)；[v1/engine/logprobs.py:32-75](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/logprobs.py#L32-L75)
- SGLang：[layers/sampler.py:119-190](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/sampler.py#L119-L190)；[managers/tokenizer_manager.py:2106-2299](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/tokenizer_manager.py#L2106-L2299)
- 调研记录：exec#17, reqio#12

### `samp.processor_order`

logits 处理按声明顺序作用：白名单/禁用词 → 影响 argmax 的处理器 → 惩罚 → 温度与过滤；惩罚只使用本请求历史。

- 阶段：logits 处理、采样、logprob、推测验收；类型：value
- 判定：手算 logits 参考。
- 条件：`sampling`
- 框架差异：SGLang 在 _preprocess_logits 集中处理自定义处理器与 NaN 清理；处理顺序作为框架条件保留。
- vLLM：[v1/sample/sampler.py:78-180](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/sample/sampler.py#L78-L180)
- SGLang：[layers/sampler.py:93-190](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/sampler.py#L93-L190)
- 调研记录：exec#13, exec#14

### `spec.distribution`

推测采样的接受/拒绝与补偿采样使下一 token 分布等于目标模型分布；贪心是其特例。

- 阶段：logits 处理、采样、logprob、推测验收；类型：distribution
- 判定：固定目标/草稿概率的大样本检验。
- 条件：`spec_method`, `sampling`
- vLLM：[v1/sample/rejection_sampler.py:125-200](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/sample/rejection_sampler.py#L125-L200)；[v1/sample/rejection_sampler.py:589-620](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/sample/rejection_sampler.py#L589-L620)
- SGLang：[speculative/reject_sampling.py:6-60](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/speculative/reject_sampling.py#L6-L60)；[speculative/reject_sampling.py:159-200](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/speculative/reject_sampling.py#L159-L200)
- 调研记录：ext#2, ext#3

### `out.parse_reasoning`

推理与可见内容的划分能重构原文本，分隔符不泄漏。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：contract
- 判定：分隔符缺失/重复/嵌套用例。
- 条件：`parser`
- 框架差异：vLLM 推理解析基于 token ID；SGLang 基于文本。
- vLLM：[reasoning/abs_reasoning_parsers.py:118-160](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/reasoning/abs_reasoning_parsers.py#L118-L160)
- SGLang：[parser/reasoning_parser.py:62-112](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/parser/reasoning_parser.py#L62-L112)
- 调研记录：ext#15

### `out.parse_tool`

工具调用解析结果符合规范化格式；普通文本不会被识别为工具调用。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：contract
- 判定：按格式的正反例表。
- 条件：`parser`
- vLLM：[tool_parsers/abstract_tool_parser.py:201-214](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/tool_parsers/abstract_tool_parser.py#L201-L214)
- SGLang：[function_call/function_call_parser.py:118-139](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/function_call/function_call_parser.py#L118-L139)
- 调研记录：ext#13

### `topo.groups`

给定 world size 与 TP/PP/DP/EP 配置，各进程组成员集合互不重叠且符合预期排布；每个进程的组内秩一致。

- 阶段：并行拓扑、通信、权重加载/更新、PD 分离；类型：contract
- 判定：纯函数枚举：无需 GPU，对所有小规模配置计算组成员。
- 条件：`parallel`
- 框架差异：SGLang 另有 attention-TP、MoE-TP、MoE-EP、MoE-DP 独立组；成员集合需分别比较。
- vLLM：[distributed/parallel_state.py:2085-2177](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/parallel_state.py#L2085-L2177)；[distributed/parallel_state.py:2295-2318](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/parallel_state.py#L2295-L2318)
- SGLang：[distributed/parallel_state.py:1853-2127](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/distributed/parallel_state.py#L1853-L2127)
- 调研记录：dist#1, dist#2

## B1 单组件状态机

### `req.normalize`

每个被接受的 API 请求恰好产生一个内部请求；内部采样参数是 API 参数与模型/分词器配置的确定函数；非法请求在进入调度前被拒绝。

- 阶段：API 解析、校验与采样参数规范化；类型：contract
- 判定：参数表驱动：同一 API 输入两次规范化结果相同；非法组合返回错误且调度队列不变。
- 条件：`n`, `stop_config`, `logprob_mode`
- 框架差异：vLLM 在 AsyncLLM 做额外准入；SGLang 校验集中在 TokenizerManager，包括 token-ID/logprob 合法性。
- vLLM：[entrypoints/openai/chat_completion/serving.py:303-441](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/entrypoints/openai/chat_completion/serving.py#L303-L441)；[v1/engine/async_llm.py:326-399](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/async_llm.py#L326-L399)
- SGLang：[managers/tokenizer_manager.py:937-1100](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/tokenizer_manager.py#L937-L1100)；[entrypoints/openai/serving_chat.py:315-327](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/entrypoints/openai/serving_chat.py#L315-L327)
- 调研记录：reqio#1, reqio#2, reqio#3

### `mm.identity`

相同媒体与预处理配置得到相同身份；任何影响特征的差异得到不同身份。

- 阶段：对话模板、分词与多模态预处理；类型：resource
- 判定：同图异配置、异图同尺寸、条目顺序交换等成对输入比较身份。
- 条件：`multimodal`
- 框架差异：vLLM 把完整多模态哈希放入块哈希 extra keys；SGLang 用 pad_value = 偏移 + hash % 2^30 替换占位 token，再进入基数树键，身份在前缀键中只保留 30 位。
- vLLM：[multimodal/hasher.py:94-130](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/multimodal/hasher.py#L94-L130)；[multimodal/hasher.py:224-237](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/multimodal/hasher.py#L224-L237)
- SGLang：[managers/mm_utils.py:1245-1265](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/mm_utils.py#L1245-L1265)；[managers/schedule_batch.py:144-146](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L144-L146)
- 调研记录：ext#10

### `mm.placeholder`

每个多模态条目的占位 token 数与位置等于模型实际消费的特征跨度；裁剪后的保留数等于调度预留数。

- 阶段：对话模板、分词与多模态预处理；类型：contract
- 判定：构造不同尺寸/帧数输入，比较占位区间、特征张量长度与预留 token 数。
- 条件：`multimodal`
- vLLM：[multimodal/inputs.py:180-200](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/multimodal/inputs.py#L180-L200)；[multimodal/video_prune/evs.py:16-60](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/multimodal/video_prune/evs.py#L16-L60)
- SGLang：[managers/mm_utils.py:227-280](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/mm_utils.py#L227-L280)；[multimodal/evs/evs_core.py:21-60](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/multimodal/evs/evs_core.py#L21-L60)
- 调研记录：ext#9, ext#12

### `req.render`

相同消息、模板与选项得到相同渲染文本及生成起点；传入引擎的 token 序列与渲染文本在同一分词选项下一致。

- 阶段：对话模板、分词与多模态预处理；类型：contract
- 判定：对模板/工具/推理开关的取值表做逐项比对；decode(ids) 与渲染文本对照。
- 条件：`parser`
- 框架差异：SGLang 区分 Jinja 与 conversation 模板分支；vLLM 经 renderer 与 chat-template kwargs。
- vLLM：[entrypoints/openai/chat_completion/serving.py:198-286](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/entrypoints/openai/chat_completion/serving.py#L198-L286)
- SGLang：[entrypoints/openai/serving_chat.py:591-939](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/entrypoints/openai/serving_chat.py#L591-L939)；[managers/tokenizer_manager.py:576-695](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/tokenizer_manager.py#L576-L695)
- 调研记录：reqio#4, reqio#5

### `sched.abort`

取消幂等；被取消请求最终从等待、运行、分块、挂起各结构中移除，只产生一个终止结果，并释放其独占资源。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：protocol
- 判定：伪模型下在等待/运行/分块中途三种时刻取消，检查结构与资源。
- 条件：`abort_timing`, `chunk_size`, `overlap`
- 框架差异：SGLang 对分块中的请求延迟取消（_pending_chunked_abort_req）；vLLM 在 EngineCore 步边界处理取消队列。只能要求最终清理，不能要求同步生效。
- vLLM：[v1/engine/core.py:606-612](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/core.py#L606-L612)；[v1/engine/core.py:869-877](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/core.py#L869-L877)
- SGLang：[managers/scheduler.py:3774-3815](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L3774-L3815)；[managers/scheduler.py:2464-2510](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2464-L2510)
- 调研记录：sched#11, sched#12

### `sched.budget`

每步调度 token 总数不超过预算；活动请求数不超过上限；准入请求都持有所需槽位与 KV。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：resource
- 判定：伪模型下随机负载，逐步断言预算与槽位不变式。
- 条件：`chunk_size`, `cache_state`, `parallel`
- 框架差异：SGLang 准入同时受 req_to_token_pool 可用量与 PP 微批上限约束；vLLM 以标量 token 预算为主并区分 runner 槽位与准入槽位。
- vLLM：[v1/core/sched/scheduler.py:718-748](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L718-L748)；[v1/core/sched/scheduler.py:128-137](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L128-L137)；[v1/core/sched/scheduler.py:1688-1690](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L1688-L1690)
- SGLang：[managers/scheduler.py:2684-2702](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2684-L2702)
- 调研记录：sched#4, sched#8

### `sched.chunking`

各分块拼接恰好覆盖提示且无重叠；同一请求不同时处于活动批与暂存状态。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：protocol
- 判定：伪模型记录每步分块区间。
- 条件：`chunk_size`
- 框架差异：分块边界不要求跨框架一致，只要求覆盖与互斥。
- vLLM：[v1/core/sched/scheduler.py:1343-1454](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L1343-L1454)
- SGLang：[managers/scheduler.py:2464-2467](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2464-L2467)；[managers/scheduler.py:2603-2617](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2603-L2617)
- 调研记录：sched#6

### `sched.order`

等待队列出队顺序服从所配置策略；相同键时顺序确定。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：protocol
- 判定：伪模型+伪分配器：注入到达时间/优先级/前缀命中，检查出队序列。
- 条件：`scheduler_policy`
- 框架差异：SGLang 支持 LPM/DFS-weight 等缓存感知策略；vLLM 比较优先级、到达时间、请求 ID。
- vLLM：[v1/request.py:407-420](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L407-L420)；[v1/core/sched/scheduler.py:1068-1100](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L1068-L1100)
- SGLang：[managers/schedule_policy.py:107-202](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_policy.py#L107-L202)
- 调研记录：sched#2

### `sched.preempt_state`

被抢占/回撤的请求一定重新入队；其 KV 要么全部释放待重算，要么完整保留，不出现部分所有权。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：resource
- 判定：伪分配器容量压力下触发抢占，检查队列与块所有权。
- 条件：`preemption`
- 框架差异：vLLM 用 PREEMPTED 状态枚举；SGLang 通过队列/批重组表达回撤，不存在等价枚举，应检查重新入队语义而非状态值。
- vLLM：[v1/core/sched/scheduler.py:946-1010](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L946-L1010)；[v1/request.py:422-446](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L422-L446)
- SGLang：[managers/scheduler.py:2258-2262](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2258-L2262)；[managers/schedule_batch.py:2684-2761](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L2684-L2761)
- 调研记录：sched#9

### `sched.prefix_bound`

前缀命中长度满足 0 ≤ hit ≤ min(n−1, logprob 起点, 显式上限)，使最后位置和需要 logprob 的位置一定被重算。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：contract
- 判定：构造全命中请求与带 prompt logprob 请求，检查调度的已计算长度。
- 条件：`hit_bound`, `logprob_mode`, `cache_state`
- 框架差异：两者都保证 n−1；vLLM 另有 skip_reading_prefix_cache 与 prompt_logprob_start，SGLang 仅在返回 logprob 时再用 logprob_start_len 收紧。
- vLLM：[v1/request.py:352-377](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L352-L377)；[v1/core/kv_cache_manager.py:322-330](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_manager.py#L322-L330)
- SGLang：[managers/schedule_batch.py:1223-1228](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L1223-L1228)；[managers/schedule_policy.py:84-122](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_policy.py#L84-L122)
- 调研记录：sched#3

### `sched.progress`

每个未结束的运行请求在每步被调度或被显式推迟；同一批中请求不重复；已结束请求不再调度。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：protocol
- 判定：伪模型逐步检查批成员集合。
- 条件：`schedule_order`
- 框架差异：vLLM 先调度运行请求再准入等待请求；SGLang 有新 prefill 批时先跑 prefill、可跳过本步 decode。公平性与顺序必须作为条件。
- vLLM：[v1/core/sched/scheduler.py:812-1065](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L812-L1065)
- SGLang：[managers/scheduler.py:2555-2683](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2555-L2683)；[managers/schedule_batch.py:2684-2701](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L2684-L2701)
- 调研记录：sched#5, sched#13

### `grammar.state`

每个请求的语法状态只按其已接受 token 前进；位掩码行与请求一一对应。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：protocol
- 判定：多语法请求打乱批顺序，检查每行掩码与参考解析器。
- 条件：`structured_output`, `batch_shape`
- vLLM：[v1/structured_output/backend_types.py:36-80](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/structured_output/backend_types.py#L36-L80)；[v1/structured_output/__init__.py:285-300](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/structured_output/__init__.py#L285-L300)
- SGLang：[constrained/base_grammar_backend.py:42-80](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/constrained/base_grammar_backend.py#L42-L80)
- 调研记录：exec#18

### `kv.conservation`

静止时 已分配 + 空闲 = 总量；分配出的物理块/页唯一且在范围内。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：对分配器做随机分配/释放序列（属性测试）。
- 条件：`page_size`
- vLLM：[v1/core/block_pool.py:139-158](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L139-L158)；[v1/core/block_pool.py:628-660](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L628-L660)
- SGLang：[mem_cache/allocator/paged.py:102-192](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/allocator/paged.py#L102-L192)
- 调研记录：kv#1

### `kv.events`（扩展）

对外的缓存事件流与内部状态迁移一一对应：每次存入/移除恰好一个事件，事件键与查找命名空间一致。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：protocol
- 判定：记录事件流并回放出缓存集合，与实际缓存集合比较。
- 条件：`kv_events`
- 框架差异：SGLang HiRadixCache 断言不支持事件（约束 K1）。
- vLLM：[v1/core/block_pool.py:287-303](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L287-L303)；[v1/core/block_pool.py:834-860](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L834-L860)
- SGLang：[mem_cache/radix_cache.py:467-610](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L467-L610)；[mem_cache/radix_cache_cpp.py:45-47](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache_cpp.py#L45-L47)
- 调研记录：kv#14

### `kv.hit_extent`

返回的物理映射长度等于报告的命中长度，且命中粒度符合配置（块对齐或页对齐）。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：contract
- 判定：在不同 page/block 大小下插入与查询，比较长度与索引。
- 条件：`cache_granularity`, `page_size`
- 框架差异：vLLM 以满块命中；SGLang 基数树按 token 匹配后按 page_size 截断。
- vLLM：[v1/core/kv_cache_manager.py:289-353](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_manager.py#L289-L353)；[v1/core/kv_cache_coordinator.py:464-500](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_coordinator.py#L464-L500)
- SGLang：[mem_cache/radix_cache.py:358-435](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L358-L435)
- 调研记录：kv#5

### `kv.key_completeness`

前缀键区分所有影响 KV 数值的输入：token、适配器、多模态内容、cache salt、prompt embedding；键不同输入之间不得命中。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：成对输入只改变一个 KV 相关因素，检查是否命中；包括构造的命名空间拼接碰撞。
- 条件：`lora`, `multimodal`, `cache_granularity`
- 框架差异：vLLM 用结构化 extra-key 元组（LoRA、多模态哈希、首块 cache_salt、prompt embeds）；SGLang 把 lora_id 直接拼接到 extra_key 字符串后，多模态身份经截断哈希编码进 token。键的编码方式不同，需要分别构造碰撞用例。
- vLLM：[v1/core/kv_cache_utils.py:594-676](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_utils.py#L594-L676)；[v1/core/block_pool.py:262-303](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L262-L303)
- SGLang：[mem_cache/radix_cache.py:59-71](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L59-L71)；[mem_cache/radix_cache.py:311-327](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L311-L327)；[managers/schedule_batch.py:776-783](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L776-L783)
- 调研记录：kv#3, kv#4, ext#17

### `kv.mapping`

每个活动请求恰有一份逻辑到物理映射；映射行在其物理位置释放后才可复用；分块过渡不产生重复所有权。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：随机请求生命周期序列，检查映射表与分配器一致。
- 条件：`cache_granularity`, `chunk_size`
- 框架差异：vLLM 映射按块；SGLang req_to_token 按 token 槽位，即使物理分配按页。
- vLLM：[v1/core/single_type_kv_cache_manager.py:390-467](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/single_type_kv_cache_manager.py#L390-L467)
- SGLang：[mem_cache/memory_pool.py:229-290](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/memory_pool.py#L229-L290)；[mem_cache/radix_cache.py:520-610](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L520-L610)
- 调研记录：kv#2, kv#10

### `kv.no_shared_write`

请求不会写入仍被其他请求引用的物理块；部分命中时先复制或分裂。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：共享前缀后在非对齐位置继续生成，检查写入目标与引用计数。
- 条件：`cache_state`, `page_size`
- 框架差异：vLLM 对非对齐本地命中预留写时复制块；SGLang 通过节点分裂暴露精确 token 边界，物理页仍受对齐约束。
- vLLM：[v1/core/single_type_kv_cache_manager.py:173-181](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/single_type_kv_cache_manager.py#L173-L181)；[v1/core/single_type_kv_cache_manager.py:245-263](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/single_type_kv_cache_manager.py#L245-L263)
- SGLang：[mem_cache/radix_cache.py:408-435](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L408-L435)
- 调研记录：kv#6

### `kv.ref_protect`

被使用的块/节点不被驱逐；每次获取恰好对应一次释放；驱逐同时移除查找元数据。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：交错的锁定、释放、驱逐序列（属性测试），并覆盖各驱逐策略。
- 条件：`cache_state`
- 框架差异：vLLM 直接对物理块计数；SGLang 从终端节点向根传播 lock_ref，并支持 LRU/LFU/SLRU/优先级策略。
- vLLM：[v1/core/block_pool.py:767-832](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L767-L832)
- SGLang：[mem_cache/radix_cache.py:632-671](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L632-L671)；[mem_cache/evict_policy.py:16-62](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/evict_policy.py#L16-L62)
- 调研记录：kv#7, kv#8

### `kv.release_terminal`

请求结束或取消后，其独占引用归零；仍被缓存持有的前缀引用保持有效；最后一个调度步之前不释放。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：生命周期序列结束后核对分配器与缓存所有权。
- 条件：`abort_timing`, `overlap`, `window`
- 依赖：`kv.ref_protect`, `kv.mapping`
- 框架差异：vLLM 用 last_sched_seq 推迟释放；SGLang 在 cache_finished_req 中释放非缓存与未对齐尾部。
- vLLM：[v1/core/kv_cache_manager.py:650-702](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_manager.py#L650-L702)；[v1/core/sched/scheduler.py:1703-1760](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L1703-L1760)；[v1/request.py:186-189](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L186-L189)
- SGLang：[mem_cache/radix_cache.py:467-520](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L467-L520)；[managers/schedule_batch.py:2522-2553](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L2522-L2553)
- 调研记录：kv#9, sched#14, reqio#18

### `kv.reset`

重置后旧键不再命中，所有物理位置处于活动或可回收状态。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：填充缓存、重置、再查询并核对计数。
- vLLM：[v1/core/block_pool.py:860-900](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L860-L900)
- SGLang：[mem_cache/radix_cache.py:342-405](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L342-L405)
- 调研记录：kv#15

### `lora.residency`

批执行期间该批引用的适配器都驻留且槽位不被替换；驻留数不超过容量。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：适配器数超过槽位的负载下检查每批槽位与适配器身份。
- 条件：`lora`
- 框架差异：SGLang 有显式 validate_lora_batch；vLLM 由管理器缓存与 Punica 集成。
- vLLM：[lora/model_manager.py:79-120](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/lora/model_manager.py#L79-L120)；[lora/model_manager.py:340-360](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/lora/model_manager.py#L340-L360)
- SGLang：[lora/lora_manager.py:165-200](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/lora/lora_manager.py#L165-L200)；[lora/lora_manager.py:272-330](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/lora/lora_manager.py#L272-L330)
- 调研记录：ext#6, ext#8

### `out.detok`

增量输出的拼接加最终刷新等于整体解码结果；不在 UTF-8 半字符处截断或重复。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：equivalence
- 判定：对任意 token 序列逐 token 喂入，比较拼接与一次性解码。
- 条件：`stream`
- 执行扰动：增量处理 与 整体处理
- vLLM：[v1/engine/detokenizer.py:32-180](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/detokenizer.py#L32-L180)
- SGLang：[managers/detokenizer_manager.py:62-89](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/detokenizer_manager.py#L62-L89)；[managers/detokenizer_manager.py:205-404](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/detokenizer_manager.py#L205-L404)
- 调研记录：reqio#9

### `out.parse_incremental`

流式解析的最终结果等于对完整文本的一次解析；每段增量不重复输出，分隔符在任意切分点都正确。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：equivalence
- 判定：对同一文本枚举所有切分点。
- 条件：`parser`, `stream`
- 执行扰动：增量处理 与 整体处理
- 依赖：`out.parse_tool`, `out.parse_reasoning`
- vLLM：[tool_parsers/abstract_tool_parser.py:214-240](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/tool_parsers/abstract_tool_parser.py#L214-L240)；[reasoning/abs_reasoning_parsers.py:173-200](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/reasoning/abs_reasoning_parsers.py#L173-L200)
- SGLang：[function_call/function_call_parser.py:139-170](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/function_call/function_call_parser.py#L139-L170)；[parser/reasoning_parser.py:112-150](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/parser/reasoning_parser.py#L112-L150)
- 调研记录：ext#14, ext#16

### `out.stop`

命中停止条件后，返回文本按配置排除或包含停止串，且第一个生效停止之后不再输出文本；停止 token 的可见性与计数遵循配置。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：contract
- 判定：停止串跨 token 边界、多个停止串、包含停止串三类表驱动用例。
- 条件：`stop_config`, `stream`
- 框架差异：SGLang 只处理一个匹配到的停止串（源码 TODO），并假定最后一个 token 就是停止 token；vLLM 将 stop_terminated 传入增量解码器。
- vLLM：[v1/engine/detokenizer.py:98-160](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/detokenizer.py#L98-L160)；[v1/engine/detokenizer.py:224-262](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/detokenizer.py#L224-L262)
- SGLang：[managers/detokenizer_manager.py:169-199](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/detokenizer_manager.py#L169-L199)
- 调研记录：reqio#10, reqio#11

### `topo.shard_load`

按秩加载的列/行并行、融合 QKV 与量化权重分片拼接（反量化后）等于检查点张量。

- 阶段：并行拓扑、通信、权重加载/更新、PD 分离；类型：value
- 判定：模拟各秩加载同一检查点，无需通信。
- 条件：`parallel`, `dtype_quant`
- 框架差异：SGLang 另行处理 BitsAndBytes 4bit 偏移；vLLM 处理 Marlin/块缩放偏移。
- vLLM：[model_executor/layers/linear.py:439-619](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/model_executor/layers/linear.py#L439-L619)；[model_executor/layers/linear.py:745-966](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/model_executor/layers/linear.py#L745-L966)；[model_executor/layers/linear.py:74-122](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/model_executor/layers/linear.py#L74-L122)
- SGLang：[layers/linear.py:291-466](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/linear.py#L291-L466)；[layers/linear.py:489-712](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/linear.py#L489-L712)；[layers/linear.py:85-128](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/linear.py#L85-L128)
- 调研记录：dist#5, dist#6, dist#7, dist#8

## B2 单步引擎

### `sched.mode_mix`

批元数据中每个请求的计算模式与其真实进度一致；禁止混合时不出现 prefill/decode 混合批。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：contract
- 判定：单步检查 forward mode 与各请求已计算/调度 token 数。
- 条件：`forward_mode`, `spec_method`, `parallel`
- 框架差异：SGLang 有显式 MIXED/TARGET_VERIFY/SPLIT_PREFILL 模式；vLLM 由已计算与提示长度推导是否在 prefill。
- vLLM：[v1/worker/gpu_model_runner.py:2418-2429](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L2418-L2429)
- SGLang：[model_executor/forward_batch_info.py:60-145](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L60-L145)；[managers/scheduler.py:2621-2676](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2621-L2676)
- 调研记录：sched#7, exec#5

### `kv.hybrid_boundary`（扩展）

每个复用边界上所有带状态的层组（注意力、Mamba 等）都有有效状态。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：contract
- 判定：混合模型上比较不同命中边界的输出与无缓存输出。
- 条件：`window`, `cache_state`
- vLLM：[v1/core/single_type_kv_cache_manager.py:311-345](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/single_type_kv_cache_manager.py#L311-L345)；[v1/core/kv_cache_coordinator.py:943-990](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_coordinator.py#L943-L990)
- SGLang：[mem_cache/memory_pool.py:646-884](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/memory_pool.py#L646-L884)
- 调研记录：kv#12

### `kv.tier_coherence`（扩展）

多级缓存中每个分配恰由一层持有；提升/恢复的值等于原 KV。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：强制卸载后恢复，比较 KV 张量与所有权计数。
- 条件：`cache_state`
- 框架差异：vLLM 以按介质划分的块池/连接器表示层级；SGLang 把主机副本与引用嵌入基数树节点。
- vLLM：[v1/core/block_pool.py:222-303](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L222-L303)
- SGLang：[mem_cache/radix_cache.py:235-285](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L235-L285)；[mem_cache/hiradix_cache.py:1008-1113](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/hiradix_cache.py#L1008-L1113)
- 调研记录：kv#13

### `kv.window_recycle`

窗口外的物理位置只有在没有核函数还会读取时才被回收。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：滑动窗口模型上把被回收位置写入哨兵值，检查输出不变。
- 条件：`window`
- 框架差异：vLLM 保留块对象并用空块屏蔽；SGLang 按窗口回收 token 槽位，逻辑映射可能不连续。
- vLLM：[v1/core/block_pool.py:253-303](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L253-L303)；[v1/core/single_type_kv_cache_manager.py:219-263](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/single_type_kv_cache_manager.py#L219-L263)
- SGLang：[mem_cache/allocator/swa.py:309-341](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/allocator/swa.py#L309-L341)
- 调研记录：kv#11

### `spec.kv_rollback`

每次验证后已分配 KV 长度等于 提示 + 已接受 token，被拒绝草稿的 KV 被回收或不可见。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：resource
- 判定：多轮验证后核对序列长度、空闲块数，并重复运行检查泄漏。
- 条件：`spec_method`
- vLLM：[v1/core/kv_cache_manager.py:401-460](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_manager.py#L401-L460)；[v1/sample/rejection_sampler.py:415-430](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/sample/rejection_sampler.py#L415-L430)
- SGLang：[speculative/eagle_worker_v2.py:201-230](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/speculative/eagle_worker_v2.py#L201-L230)；[speculative/ngram_worker.py:201-230](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/speculative/ngram_worker.py#L201-L230)
- 调研记录：ext#18, ext#2

### `bind.boundaries`

行边界为累积长度 [0,a,a+b,...]，填充尾部保持非减常数。

- 阶段：批次到张量与元数据的绑定、图捕获与填充；类型：contract
- 判定：元数据不变式检查。
- 条件：`batch_shape`, `graph`
- vLLM：[v1/worker/gpu_model_runner.py:2125-2131](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L2125-L2131)
- SGLang：[model_executor/forward_batch_info.py:822-829](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L822-L829)
- 调研记录：exec#3

### `bind.graph_padding`

图重放对真实行的输出等于即时执行；填充行不写 KV、不影响真实行。

- 阶段：批次到张量与元数据的绑定、图捕获与填充；类型：equivalence
- 判定：哨兵 KV/logit 行 + 图与即时结果比较，覆盖桶边界。
- 条件：`graph`, `batch_shape`
- 执行扰动：CUDA 图重放/填充 与 即时执行
- 依赖：`bind.boundaries`, `bind.slot_mapping`
- 组合缺口：逐行元数据正确不足以排除填充行通过共享缓冲区写入 KV；需要哨兵观察。
- vLLM：[v1/attention/backend.py:726-737](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/attention/backend.py#L726-L737)；[v1/worker/gpu_model_runner.py:2368-2469](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L2368-L2469)
- SGLang：[model_executor/runner/base_cuda_graph_runner.py:61-158](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py#L61-L158)；[model_executor/forward_batch_info.py:1314-1365](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L1314-L1365)
- 调研记录：exec#7

### `bind.positions`

展平的 token 与位置与每个请求的已计算进度一致（含分块偏移、多维 RoPE）。

- 阶段：批次到张量与元数据的绑定、图捕获与填充；类型：contract
- 判定：构造分块/混合批，比较位置张量与参考计算。
- 条件：`chunk_size`, `forward_mode`, `multimodal`
- vLLM：[v1/worker/gpu_model_runner.py:1832-1885](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L1832-L1885)；[v1/worker/gpu_model_runner.py:2016-2140](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L2016-L2140)
- SGLang：[model_executor/forward_batch_info.py:778-849](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L778-L849)
- 调研记录：exec#2

### `bind.slot_mapping`

本步 KV 写入位置都属于对应请求，且与逻辑映射一致。

- 阶段：批次到张量与元数据的绑定、图捕获与填充；类型：resource
- 判定：以哨兵 KV 池执行一步，检查被写位置集合。
- 条件：`page_size`, `cache_state`
- 依赖：`kv.mapping`
- vLLM：[v1/worker/gpu_model_runner.py:2248-2252](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L2248-L2252)；[v1/worker/gpu_model_runner.py:2414-2469](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L2414-L2469)
- SGLang：[model_executor/forward_batch_info.py:325-350](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L325-L350)；[model_executor/forward_batch_info.py:674-680](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L674-L680)
- 调研记录：exec#4

### `bind.slot_reset`

复用的批内请求槽不残留上一个请求的采样参数、惩罚历史、掩码或随机数生成器。

- 阶段：批次到张量与元数据的绑定、图捕获与填充；类型：resource
- 判定：先放入特殊参数请求再替换，检查新请求行的状态。
- 条件：`batch_shape`
- 框架差异：vLLM 在 InputBatch 中持有异构采样状态；SGLang 从 ScheduleBatch 借用核心字段到 ForwardBatch。
- vLLM：[v1/worker/gpu_input_batch.py:196-282](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_input_batch.py#L196-L282)
- SGLang：[model_executor/forward_batch_info.py:606-691](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/forward_batch_info.py#L606-L691)
- 调研记录：exec#1

### `op.compiled`

编译执行与即时执行在桶边界两侧的输出与副作用一致。

- 阶段：注意力、线性/量化、归一化、MoE 等数值计算；类型：equivalence
- 判定：仅改变批形状、不改变真实请求的变形测试。
- 条件：`graph`, `batch_shape`
- 执行扰动：编译执行 与 即时执行
- vLLM：[compilation/backends.py:1-60](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/compilation/backends.py#L1-L60)
- SGLang：[model_executor/cpu_graph_runner.py:88-113](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/cpu_graph_runner.py#L88-L113)
- 调研记录：exec#12

## B3 单请求多步

### `eq.mm_cache`

多模态编码缓存命中与不命中时，嵌入与输出相同；不同媒体不共享前缀 KV。

- 阶段：对话模板、分词与多模态预处理；类型：equivalence
- 判定：同文本异图、同图异配置的冷/热执行。
- 条件：`multimodal`, `cache_state`
- 执行扰动：编码缓存命中 与 不命中
- 依赖：`mm.identity`, `mm.placeholder`, `kv.key_completeness`
- 组合缺口：身份、占位与键各自正确时，缓存条目的 dtype/设备/偏移复用错误只表现为嵌入数值差异。
- vLLM：[multimodal/processing/processor.py:59-90](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/multimodal/processing/processor.py#L59-L90)
- SGLang：[managers/mm_utils.py:82-120](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/mm_utils.py#L82-L120)
- 调研记录：ext#11, ext#17

### `eq.overlap`

开启异步/重叠调度时，每个请求的输出与同步调度相同。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：equivalence
- 判定：同负载开关比较。
- 条件：`overlap`
- 执行扰动：异步/重叠调度 与 同步
- 依赖：`out.exactly_once`, `kv.release_terminal`
- 组合缺口：重叠执行改变状态读写时序，属于不能由单步局部性质推出的时序依赖。
- vLLM：[v1/engine/core.py:709-869](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/core.py#L709-L869)
- SGLang：[managers/scheduler.py:1505-1594](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L1505-L1594)；[managers/scheduler.py:3085-3145](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L3085-L3145)
- 调研记录：sched#10

### `eq.preemption`

被抢占/回撤后恢复的请求，在贪心下与不中断执行输出相同且结束原因相同。

- 阶段：准入、批组成、分块、抢占、取消与结束；类型：equivalence
- 判定：通过缩小 KV 容量强制抢占的变形测试。
- 条件：`preemption`, `overlap`, `cache_state`
- 执行扰动：抢占/回撤后恢复 与 不中断
- 依赖：`sched.preempt_state`, `kv.release_terminal`, `eq.prefix_hit`
- 组合缺口：调度器状态与 KV 计数正确时，异步在途输出仍可能被重复使用；重算路径又依赖前缀命中等价。
- vLLM：[v1/core/sched/scheduler.py:946-1010](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/sched/scheduler.py#L946-L1010)；[v1/request.py:166-185](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L166-L185)
- SGLang：[managers/scheduler.py:2555-2676](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2555-L2676)；[managers/schedule_batch.py:724-759](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L724-L759)
- 调研记录：sched#9, sched#15

### `eq.prefix_hit`

前缀缓存命中时的输出与无缓存执行等价（贪心 token 相同；prompt logprob 在容差内）。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：equivalence
- 判定：同一请求冷/热两次执行，覆盖部分命中、页边界与 prompt logprob。
- 条件：`cache_state`, `cache_granularity`, `page_size`, `logprob_mode`, `lora`, `multimodal`
- 执行扰动：前缀缓存命中 与 冷启动
- 依赖：`kv.key_completeness`, `kv.hit_extent`, `kv.no_shared_write`, `kv.ref_protect`, `sched.prefix_bound`, `bind.slot_mapping`, `op.attention`
- 组合缺口：键完整与映射正确只保证读到“同一输入”算出的 KV；该 KV 是由另一批组成、另一内核路径算出的，数值是否足以使结果一致只能在请求级观察。
- vLLM：[v1/core/kv_cache_manager.py:289-353](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/kv_cache_manager.py#L289-L353)
- SGLang：[mem_cache/radix_cache.py:358-435](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L358-L435)
- 调研记录：kv#3, kv#5, sched#3

### `eq.tier_restore`（扩展）

KV 被卸载到低层后恢复命中，输出与未卸载时相同。

- 阶段：KV/前缀缓存、适配器槽位、语法状态等跨步状态；类型：equivalence
- 判定：强制卸载后的冷/热比较。
- 条件：`cache_state`
- 执行扰动：卸载恢复 与 常驻
- 依赖：`kv.tier_coherence`, `eq.prefix_hit`
- 组合缺口：异步写回尚未完成时设备副本已被驱逐，属于时序问题，单独的所有权与数值检查都不能覆盖。
- vLLM：[v1/core/block_pool.py:222-303](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/core/block_pool.py#L222-L303)
- SGLang：[mem_cache/hiradix_cache.py:1008-1113](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/hiradix_cache.py#L1008-L1113)
- 调研记录：kv#13

### `eq.chunked_prefill`

分块与不分块 prefill 得到等价的最终 logits/KV，从而在贪心下输出相同。

- 阶段：注意力、线性/量化、归一化、MoE 等数值计算；类型：equivalence
- 判定：变形测试：只改变 chunk 大小。
- 条件：`chunk_size`, `attention_backend`, `dtype_quant`
- 执行扰动：分块 prefill 与 整体 prefill
- 依赖：`sched.chunking`, `bind.positions`, `bind.slot_mapping`, `op.attention`
- 组合缺口：分块使注意力分别走 extend 与 prefill 内核；逐算子各自在容差内不保证累积后贪心 token 相同，需在请求级判定并声明容差策略。
- vLLM：[v1/worker/gpu_model_runner.py:2643-2680](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L2643-L2680)
- SGLang：[model_executor/model_runner.py:2869-2894](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/model_executor/model_runner.py#L2869-L2894)
- 调研记录：exec#9, sched#6

### `eq.spec_greedy`

贪心推测解码的输出与不开推测的贪心输出相同。

- 阶段：logits 处理、采样、logprob、推测验收；类型：equivalence
- 判定：开关推测的变形测试，覆盖链式、树式与 n-gram。
- 条件：`spec_method`, `graph`
- 执行扰动：推测解码 与 普通解码
- 依赖：`spec.distribution`, `spec.kv_rollback`, `bind.positions`
- 组合缺口：接受判定、KV 回滚与位置各自正确时，树式草稿的位置/掩码组合错误只在请求级可见。
- 框架差异：SGLang EAGLE 可形成树式草稿并维护独立草稿 worker 与图缓冲；树宽/深是独立条件。
- vLLM：[v1/spec_decode/draft_model.py:20-90](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/spec_decode/draft_model.py#L20-L90)；[v1/spec_decode/ngram_proposer.py:1-40](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/spec_decode/ngram_proposer.py#L1-L40)
- SGLang：[speculative/eagle_worker_v2.py:268-300](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/speculative/eagle_worker_v2.py#L268-L300)；[speculative/eagle_worker_v2.py:469-600](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/speculative/eagle_worker_v2.py#L469-L600)；[speculative/ngram_worker.py:131-160](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/speculative/ngram_worker.py#L131-L160)
- 调研记录：ext#1, ext#4, ext#5

### `out.exactly_once`

每个生成 token 按序恰好处理一次；每个请求恰好一个终止输出；被接受的请求不会无声丢失。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：protocol
- 判定：注入调度扰动后检查输出序列号与终止事件。
- 条件：`overlap`, `abort_timing`
- 依赖：`sched.progress`, `sched.abort`
- 组合缺口：调度器局部计数正确时，异步步之间的过期输出仍可能被重复或遗漏处理（vLLM request.py:166-185 的在途占位）。
- vLLM：[v1/engine/output_processor.py:731-920](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/output_processor.py#L731-L920)；[v1/engine/core_client.py:191-200](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/core_client.py#L191-L200)
- SGLang：[managers/tokenizer_manager.py:1839-1947](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/tokenizer_manager.py#L1839-L1947)；[managers/detokenizer_manager.py:269-404](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/detokenizer_manager.py#L269-L404)
- 调研记录：reqio#6, reqio#8, sched#10

### `out.finish_reason`

每个终止的输出选项恰有一个结束原因，且与实际终止原因（停止、长度、取消、错误）和最后输出一致。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：contract
- 判定：按终止原因构造请求，检查原因与最后文本/token。
- 条件：`stop_config`, `abort_timing`
- 依赖：`out.stop`, `sched.abort`
- vLLM：[v1/engine/output_processor.py:425-532](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/output_processor.py#L425-L532)
- SGLang：[managers/tokenizer_manager.py:1381-1424](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/tokenizer_manager.py#L1381-L1424)
- 调研记录：reqio#14

### `out.grammar_valid`

受语法约束的请求，其完整输出被该语法接受（含推测解码与异步语法编译）。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：contract
- 判定：用参考解析器检查完整输出；开关推测解码各执行一次。
- 条件：`structured_output`, `spec_method`, `batch_shape`
- 依赖：`grammar.state`, `samp.grammar_mask`
- 组合缺口：单步掩码正确时，推测 token 被拒绝后语法状态的回滚（vLLM backend_types.rollback）与异步编译完成时机决定整段输出是否合法。
- vLLM：[v1/structured_output/backend_types.py:64-72](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/structured_output/backend_types.py#L64-L72)；[v1/structured_output/__init__.py:285-300](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/structured_output/__init__.py#L285-L300)
- SGLang：[constrained/base_grammar_backend.py:42-80](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/constrained/base_grammar_backend.py#L42-L80)；[constrained/grammar_manager.py:1-60](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/constrained/grammar_manager.py#L1-L60)
- 调研记录：exec#18

### `out.n_fanout`

n>1 时返回恰好 n 个带正确索引的选项；所有选项结束后才结束聚合。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：protocol
- 判定：n>1 流式与非流式检查索引与结束时机。
- 条件：`n`, `stream`
- vLLM：[v1/engine/output_processor.py:70-84](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/output_processor.py#L70-L84)；[v1/engine/output_processor.py:184-257](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/output_processor.py#L184-L257)
- SGLang：[managers/tokenizer_manager.py:1839-1950](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/tokenizer_manager.py#L1839-L1950)
- 调研记录：reqio#7

### `out.stream_parity`

同一请求流式各选项增量的拼接，与非流式最终响应在文本、工具调用、logprob、用量上等价。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：equivalence
- 判定：同种子同配置两次执行比较。
- 条件：`stream`, `n`, `parser`, `stop_config`
- 执行扰动：流式 与 非流式组装
- 依赖：`out.detok`, `out.stop`, `out.parse_incremental`, `out.usage`
- 组合缺口：两个框架流式与非流式是两条独立代码路径；局部解码与解析正确不能推出两条路径的组装一致。
- vLLM：[entrypoints/openai/chat_completion/stream_processor.py:17-52](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/entrypoints/openai/chat_completion/stream_processor.py#L17-L52)；[entrypoints/openai/chat_completion/serving.py:966-1200](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/entrypoints/openai/chat_completion/serving.py#L966-L1200)
- SGLang：[entrypoints/openai/serving_chat.py:339-428](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/entrypoints/openai/serving_chat.py#L339-L428)；[entrypoints/openai/serving_chat.py:1207-1376](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/entrypoints/openai/serving_chat.py#L1207-L1376)
- 调研记录：reqio#13, reqio#17

### `out.usage`

prompt + completion = total；流式最终用量等于非流式用量；n 与停止 token 的计数一致。

- 阶段：增量解码、停止、流式组装、解析、用量；类型：contract
- 判定：按配置组合核对计数。
- 条件：`n`, `stream`, `stop_config`, `cache_state`
- vLLM：[v1/engine/output_processor.py:257-277](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/engine/output_processor.py#L257-L277)
- SGLang：[managers/tokenizer_manager.py:2363-2426](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/tokenizer_manager.py#L2363-L2426)
- 调研记录：reqio#16

## B4 多请求并发

### `eq.lora_isolation`

不同适配器的请求同批执行，与各自单独执行结果相同；前缀缓存不跨适配器命中。

- 阶段：注意力、线性/量化、归一化、MoE 等数值计算；类型：equivalence
- 判定：混合适配器批与单独执行比较。
- 条件：`lora`, `cache_state`
- 执行扰动：混合适配器批 与 单独执行
- 依赖：`lora.residency`, `op.lora_apply`, `kv.key_completeness`
- 组合缺口：逐层 W+BA 正确与槽位驻留各自成立时，批内 token 到适配器索引的对应关系（含分块与图填充行）只在混合批的请求输出中可见。
- vLLM：[lora/model_manager.py:194-219](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/lora/model_manager.py#L194-L219)
- SGLang：[lora/lora_manager.py:272-330](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/lora/lora_manager.py#L272-L330)
- 调研记录：ext#8

### `eq.batch_invariance`

请求的输出不随同批其他请求的存在、顺序或填充而改变（贪心或固定种子）。

- 阶段：logits 处理、采样、logprob、推测验收；类型：equivalence
- 判定：单请求、批量、打乱、填充四种执行比较。
- 条件：`batch_shape`, `sampling`, `attention_backend`, `dtype_quant`
- 执行扰动：单请求、批量、乱序、填充
- 依赖：`bind.slot_reset`, `bind.graph_padding`, `samp.processor_order`, `op.moe`
- 组合缺口：批大小改变归约顺序与内核选择；即使每个算子各自符合容差，批不变性也不能由算子容差推出，除非启用批不变内核。
- 框架差异：SGLang 使用基于位置的种子并要求 sampling_seed 才保证确定性 logprob 采样；vLLM 使用逐请求生成器。
- vLLM：[v1/sample/sampler.py:1-70](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/sample/sampler.py#L1-L70)；[v1/worker/gpu_input_batch.py:250-279](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_input_batch.py#L250-L279)
- SGLang：[layers/sampler.py:274-301](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/sampler.py#L274-L301)
- 调研记录：exec#1, exec#16

## B5 多进程拓扑

### `topo.collective`

自定义 all-reduce、PyNCCL、对称内存等通信路径与标准集合通信结果一致，图捕获内外均成立。

- 阶段：并行拓扑、通信、权重加载/更新、PD 分离；类型：value
- 判定：随机张量对照 torch.distributed 参考。
- 条件：`parallel`, `graph`
- vLLM：[distributed/parallel_state.py:730-803](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/parallel_state.py#L730-L803)；[distributed/device_communicators/pynccl.py:29-181](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/device_communicators/pynccl.py#L29-L181)
- SGLang：[distributed/communication_op.py:18-84](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/distributed/communication_op.py#L18-L84)；[distributed/device_communicators/quick_all_reduce.py:52-266](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/distributed/device_communicators/quick_all_reduce.py#L52-L266)
- 调研记录：dist#3, dist#4

### `topo.ep_dispatch`（扩展）

专家并行的分发与合并保持 token 数量与顺序；输出等于复制专家参考；各秩专家映射一致。

- 阶段：并行拓扑、通信、权重加载/更新、PD 分离；类型：value
- 判定：固定路由的参考比较 + 映射一致性检查。
- 条件：`parallel`
- 依赖：`op.moe`
- vLLM：[model_executor/layers/fused_moe/modular_kernel.py:288-480](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/model_executor/layers/fused_moe/modular_kernel.py#L288-L480)
- SGLang：[layers/moe/ep_moe/layer.py:181-231](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/layers/moe/ep_moe/layer.py#L181-L231)；[eplb/lplb_solver.py:32-61](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/eplb/lplb_solver.py#L32-L61)
- 调研记录：dist#12, dist#13

### `topo.parallel_equiv`

TP/PP/DP/EP 配置下的请求输出与单卡配置在容差内一致，请求不丢失不重复。

- 阶段：并行拓扑、通信、权重加载/更新、PD 分离；类型：equivalence
- 判定：同一请求集在不同并行配置下比较。
- 条件：`parallel`, `dtype_quant`
- 执行扰动：多卡并行配置 与 单卡
- 依赖：`topo.groups`, `topo.shard_load`, `topo.collective`
- 组合缺口：分片与通信各自正确时，流水级间微批顺序、DP 路由与注意力 TP 归一化仍可能出错（gpu_model_runner/model_runner 的 DP/TP 填充计数）。
- vLLM：[distributed/parallel_state.py:2085-2177](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/parallel_state.py#L2085-L2177)；[config/parallel.py:913-969](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/config/parallel.py#L913-L969)
- SGLang：[managers/data_parallel_controller.py:125-181](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/data_parallel_controller.py#L125-L181)；[managers/scheduler_pp_mixin.py:806-869](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler_pp_mixin.py#L806-L869)
- 调研记录：dist#1, dist#9, dist#10, dist#11

### `topo.pd_transfer`（扩展）

PD 分离中解码端收到的 KV 等于本地计算的 KV；传输失败被检测且不被消费。

- 阶段：并行拓扑、通信、权重加载/更新、PD 分离；类型：equivalence
- 判定：同请求本地/分离执行比较 + 故障注入。
- 条件：`parallel`, `page_size`
- 执行扰动：PD 分离 与 本地执行
- 依赖：`kv.mapping`, `eq.prefix_hit`
- 组合缺口：两端各自的布局正确时，握手把哪个请求的哪些页发给谁、超时是否被当作成功，只在端到端传输中可见。
- vLLM：[distributed/kv_transfer/kv_transfer_state.py:19-93](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/distributed/kv_transfer/kv_transfer_state.py#L19-L93)
- SGLang：[disaggregation/base/conn.py:97-213](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/disaggregation/base/conn.py#L97-L213)
- 调研记录：dist#14

### `topo.weight_update`（扩展）

在线权重更新完成后所有秩参数等于给定张量；没有请求观察到新旧混合版本。

- 阶段：并行拓扑、通信、权重加载/更新、PD 分离；类型：protocol
- 判定：更新前后输出对照 + 更新期间并发请求检查。
- 条件：`parallel`
- vLLM：[v1/worker/gpu_model_runner.py:1-40](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/worker/gpu_model_runner.py#L1-L40)
- SGLang：[weight_sync/utils.py:14-101](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/weight_sync/utils.py#L14-L101)；[weight_sync/tensor_bucket.py:19-90](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/weight_sync/tensor_bucket.py#L19-L90)
- 调研记录：dist#15

## 条件约束

- K1：kv_events=on 与 SGLang HiRadixCache 不可同时启用（[mem_cache/radix_cache_cpp.py:45-47](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache_cpp.py#L45-L47)）
- K2：SGLang 在推测解码与 DP attention 同时启用时不进行 prefill/decode 混合（[managers/scheduler.py:2641-2654](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/scheduler.py#L2641-L2654)）
- K3：prompt logprob 请求使前缀命中上界收紧到 logprob 起点（[v1/request.py:364-377](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/vllm/v1/request.py#L364-L377)）
- K4：SGLang 前缀命中上界为 input_len-1，返回 logprob 时再受 logprob_start_len 约束（[managers/schedule_batch.py:1223-1228](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/managers/schedule_batch.py#L1223-L1228)）
- K5：SGLang 前缀匹配键按 page_size 截断，page_size>1 时命中长度只能为页对齐（[mem_cache/radix_cache.py:395-431](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/srt/mem_cache/radix_cache.py#L395-L431)）
