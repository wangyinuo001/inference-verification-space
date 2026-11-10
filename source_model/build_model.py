"""Build the source-derived Verification Space model (space_model.json).

Every unit is abstracted from the source traces in ``traces/``; ``trace``
fields point to the record numbers there. Sites use ``v:`` (relative to the
vLLM package root) and ``s:`` (relative to ``sglang/srt``).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Paths are relative to a ``reference_repos`` directory holding the pinned
# checkouts; commits are recorded so links can be rebuilt on GitHub.
SNAPSHOTS = {
    "vllm": {"path": "vllm-20261105/vllm", "repo": "vllm-project/vllm",
             "commit": "76a0ccc209343971c6f4b56f6380a76202428385", "repo_prefix": "vllm"},
    "sglang": {"path": "sglang-v0.5.14-complete/python/sglang/srt", "repo": "sgl-project/sglang",
               "commit": "49e384ce9d304648e9959666ecb8ce8cd98d0deb", "repo_prefix": "python/sglang/srt"},
}

STAGES = {
    "S0_preflight": "服务启动前的配置解析、默认值确定与能力检查",
    "S1_ingress": "API 解析、校验与采样参数规范化",
    "S2_render": "对话模板、分词与多模态预处理",
    "S3_schedule": "准入、批组成、分块、抢占、取消与结束",
    "S4_state": "KV/前缀缓存、适配器槽位、语法状态等跨步状态",
    "S5_binding": "批次到张量与元数据的绑定、图捕获与填充",
    "S6_compute": "注意力、线性/量化、归一化、MoE 等数值计算",
    "S7_sampling": "logits 处理、采样、logprob、推测验收",
    "S8_output": "增量解码、停止、流式组装、解析、用量",
    "S9_topology": "并行拓扑、通信、权重加载/更新、PD 分离",
}

KINDS = {
    "value": "数值保真：与参考实现在声明容差内一致",
    "contract": "确定性契约：输入到内部表示/输出的映射满足规格",
    "resource": "资源守恒与所有权：获取/共享/释放恰好一次，隔离不串用",
    "protocol": "协议与生命周期：状态迁移、顺序、恰好一次、终止",
    "equivalence": "执行等价：请求结果在某种执行扰动下保持不变",
    "distribution": "分布性质：随机采样结果服从目标分布",
}

BOUNDARIES = {
    "B0": {"name": "纯函数", "real": "单个函数/算子，无系统状态", "fake": "全部其他组件"},
    "B1": {"name": "单组件状态机", "real": "一个带状态的组件（分配器、基数树、调度器、增量解码器、解析器）", "fake": "模型前向、设备"},
    "B2": {"name": "单步引擎", "real": "调度结果+状态+一次真实前向（含图/填充）", "fake": "多步、多请求语义"},
    "B3": {"name": "单请求多步", "real": "单引擎实例上一个请求的完整生成", "fake": "并发干扰"},
    "B4": {"name": "多请求并发", "real": "同一实例上多个请求的批组成与相互干扰", "fake": "多进程拓扑"},
    "B5": {"name": "多进程拓扑", "real": "TP/PP/DP/EP/PD 多进程与通信", "fake": "无"},
}

CONDITIONS = {
    "config_source": ["structured", "cli", "file"],
    "config_resolution": ["explicit", "default", "alias"],
    "capability": ["model", "hardware", "dtype", "parallel", "backend"],
    "shared_buffer_kind": ["object_store", "message_ring"],
    "reader_count": ["one", "multiple"],
    "buffer_wraparound": ["no", "yes"],
    "cache_granularity": ["block_full_with_cow", "token_radix_page_aligned"],
    "page_size": ["1", ">1"],
    "cache_state": ["cold", "partial_hit", "full_hit", "evicted", "offloaded_restore"],
    "eviction_policy": ["queue_order", "lru", "lfu", "fifo", "mru", "filo", "priority", "slru"],
    "hit_bound": ["n_minus_1", "prompt_logprob_start"],
    "forward_mode": ["prefill", "extend", "decode", "mixed", "target_verify"],
    "chunk_size": ["none", "smaller_than_prompt"],
    "attention_backend": ["flash_attn", "flashinfer", "triton", "other"],
    "dtype_quant": ["bf16", "fp16", "fp8", "int4_packed"],
    "graph": ["eager", "cuda_graph_padded", "torch_compile"],
    "batch_shape": ["single", "batched", "permuted", "padded"],
    "sampling": ["greedy", "seeded_random", "random"],
    "logprob_mode": ["none", "raw", "processed", "prompt_logprobs"],
    "stop_config": ["eos", "stop_token", "single_stop_string", "multi_stop_string", "include_stop"],
    "stream": ["off", "on"],
    "n": ["1", ">1"],
    "scheduler_policy": ["fcfs", "priority", "cache_aware"],
    "schedule_order": ["running_first", "prefill_first"],
    "preemption": ["none", "recompute_or_retract"],
    "overlap": ["sync", "async_or_overlap"],
    "abort_timing": ["waiting", "running", "mid_chunk"],
    "spec_method": ["none", "draft_model", "eagle_chain", "eagle_tree", "ngram"],
    "lora": ["none", "single", "mixed_in_batch"],
    "multimodal": ["none", "image", "video_pruned"],
    "structured_output": ["none", "grammar"],
    "parser": ["none", "tool", "reasoning"],
    "window": ["full", "sliding", "hybrid_mamba"],
    "parallel": ["tp1", "tp", "pp", "dp_attention", "ep", "pd_disagg"],
    "kv_events": ["off", "on"],
}

# Constraints between conditions that the code enforces (not free products).
CONSTRAINTS = [
    {"id": "K1", "rule": "kv_events=on 与 SGLang HiRadixCache 不可同时启用",
     "site": "s:mem_cache/radix_cache_cpp.py:45-47"},
    {"id": "K2", "rule": "SGLang 在推测解码与 DP attention 同时启用时不进行 prefill/decode 混合",
     "site": "s:managers/scheduler.py:2641-2654"},
    {"id": "K3", "rule": "prompt logprob 请求使前缀命中上界收紧到 logprob 起点",
     "site": "v:v1/request.py:364-377"},
    {"id": "K4", "rule": "SGLang 前缀命中上界为 input_len-1，返回 logprob 时再受 logprob_start_len 约束",
     "site": "s:managers/schedule_batch.py:1223-1228"},
    {"id": "K5", "rule": "SGLang 前缀匹配键按 page_size 截断，page_size>1 时命中长度只能为页对齐",
     "site": "s:mem_cache/radix_cache.py:395-431"},
]

U = []


def unit(uid, stage, kind, boundary, prop, oracle, conditions, sites, trace,
         depends=(), gap=None, divergence=None, perturbation=None, status="common"):
    U.append({
        "id": uid, "stage": stage, "kind": kind, "boundary": boundary,
        "property": prop, "oracle": oracle, "conditions": list(conditions),
        "depends_on": list(depends), "composition_gap": gap,
        "perturbation": perturbation, "divergence": divergence,
        "status": status, "sites": sites, "trace": trace,
    })


# ---- S0 deployment preflight -------------------------------------------------
unit("deploy.effective_config", "S0_preflight", "contract", "B0",
     "给定配置来源、显式值与平台条件，服务启动前的生效字段符合解析、默认值和别名规则；显式值不会被默认值覆盖。",
     "对相同输入配置构造生效配置，比较关键字段及派生值；分别覆盖显式值、默认值和别名。",
     ["config_source", "config_resolution", "parallel"],
     {"vllm": ["v:config/vllm.py:1120-1147", "v:config/vllm.py:1997-2000"],
      "sglang": ["s:server_args.py:2456-2483", "s:server_args.py:2630-2647"]},
     ["preflight#1", "preflight#2"],
     divergence="vLLM 在 VllmConfig 中按优化级别设置默认项；SGLang 的 ServerArgs 将 PD 模式映射为不同的默认负载均衡方法。")
unit("deploy.admissible", "S0_preflight", "contract", "B0",
     "服务配置在声明的模型、平台、精度、并行和后端能力范围内被接受；不满足约束的组合在启动前被拒绝并给出原因。",
     "给定目标版本与平台的能力约束表，检查合法组合的配置结果及非法组合的拒绝原因。",
     ["capability", "parallel", "dtype_quant", "graph"],
     {"vllm": ["v:config/vllm.py:2027-2045", "v:config/model.py:1559-1578"],
      "sglang": ["s:server_args.py:6630-6679"]},
     ["preflight#3", "preflight#4"],
     depends=["deploy.effective_config"],
     divergence="两套框架支持的组合与拒绝位置不同，能力约束应按具体版本及平台填入。")


# ---- S1/S2 ingress and rendering -------------------------------------------
unit("req.normalize", "S1_ingress", "contract", "B1",
     "每个被接受的 API 请求恰好产生一个内部请求；内部采样参数是 API 参数与模型/分词器配置的确定函数；非法请求在进入调度前被拒绝。",
     "参数表驱动：同一 API 输入两次规范化结果相同；非法组合返回错误且调度队列不变。",
     ["n", "stop_config", "logprob_mode"],
     {"vllm": ["v:entrypoints/openai/chat_completion/serving.py:303-441", "v:v1/engine/async_llm.py:326-399"],
      "sglang": ["s:managers/tokenizer_manager.py:937-1100", "s:entrypoints/openai/serving_chat.py:315-327"]},
     ["reqio#1", "reqio#2", "reqio#3"],
     divergence="vLLM 在 AsyncLLM 做额外准入；SGLang 校验集中在 TokenizerManager，包括 token-ID/logprob 合法性。")
unit("req.render", "S2_render", "contract", "B1",
     "相同消息、模板与选项得到相同渲染文本及生成起点；传入引擎的 token 序列与渲染文本在同一分词选项下一致。",
     "对模板/工具/推理开关的取值表做逐项比对；decode(ids) 与渲染文本对照。",
     ["parser"],
     {"vllm": ["v:entrypoints/openai/chat_completion/serving.py:198-286"],
      "sglang": ["s:entrypoints/openai/serving_chat.py:591-939", "s:managers/tokenizer_manager.py:576-695"]},
     ["reqio#4", "reqio#5"],
     divergence="SGLang 区分 Jinja 与 conversation 模板分支；vLLM 经 renderer 与 chat-template kwargs。")
unit("mm.placeholder", "S2_render", "contract", "B1",
     "每个多模态条目的占位 token 数与位置等于模型实际消费的特征跨度；裁剪后的保留数等于调度预留数。",
     "构造不同尺寸/帧数输入，比较占位区间、特征张量长度与预留 token 数。",
     ["multimodal"],
     {"vllm": ["v:multimodal/inputs.py:180-200", "v:multimodal/video_prune/evs.py:16-60"],
      "sglang": ["s:managers/mm_utils.py:227-280", "s:multimodal/evs/evs_core.py:21-60"]},
     ["ext#9", "ext#12"])
unit("mm.identity", "S2_render", "resource", "B1",
     "相同媒体与预处理配置得到相同身份；任何影响特征的差异得到不同身份。",
     "同图异配置、异图同尺寸、条目顺序交换等成对输入比较身份。",
     ["multimodal"],
     {"vllm": ["v:multimodal/hasher.py:94-130", "v:multimodal/hasher.py:224-237"],
      "sglang": ["s:managers/mm_utils.py:1245-1265", "s:managers/schedule_batch.py:144-146"]},
     ["ext#10"],
     divergence="vLLM 把完整多模态哈希放入块哈希 extra keys；SGLang 用 pad_value = 偏移 + hash % 2^30 替换占位 token，再进入基数树键，身份在前缀键中只保留 30 位。")

# ---- S3 scheduling ------------------------------------------------------------
unit("sched.order", "S3_schedule", "protocol", "B1",
     "等待队列出队顺序服从所配置策略；相同键时顺序确定。",
     "伪模型+伪分配器：注入到达时间/优先级/前缀命中，检查出队序列。",
     ["scheduler_policy"],
     {"vllm": ["v:v1/request.py:407-420", "v:v1/core/sched/scheduler.py:1068-1100"],
      "sglang": ["s:managers/schedule_policy.py:107-202"]},
     ["sched#2"],
     divergence="SGLang 支持 LPM/DFS-weight 等缓存感知策略；vLLM 比较优先级、到达时间、请求 ID。")
unit("sched.budget", "S3_schedule", "resource", "B1",
     "每步调度 token 总数不超过预算；活动请求数不超过上限；准入请求都持有所需槽位与 KV。",
     "伪模型下随机负载，逐步断言预算与槽位不变式。",
     ["chunk_size", "cache_state", "parallel"],
     {"vllm": ["v:v1/core/sched/scheduler.py:718-748", "v:v1/core/sched/scheduler.py:128-137", "v:v1/core/sched/scheduler.py:1688-1690"],
      "sglang": ["s:managers/scheduler.py:2684-2702"]},
     ["sched#4", "sched#8"],
     divergence="SGLang 准入同时受 req_to_token_pool 可用量与 PP 微批上限约束；vLLM 以标量 token 预算为主并区分 runner 槽位与准入槽位。")
unit("sched.progress", "S3_schedule", "protocol", "B1",
     "每个未结束的运行请求在每步被调度或被显式推迟；同一批中请求不重复；已结束请求不再调度。",
     "伪模型逐步检查批成员集合。",
     ["schedule_order"],
     {"vllm": ["v:v1/core/sched/scheduler.py:812-1065"],
      "sglang": ["s:managers/scheduler.py:2555-2683", "s:managers/schedule_batch.py:2684-2701"]},
     ["sched#5", "sched#13"],
     divergence="vLLM 先调度运行请求再准入等待请求；SGLang 有新 prefill 批时先跑 prefill、可跳过本步 decode。公平性与顺序必须作为条件。")
unit("sched.chunking", "S3_schedule", "protocol", "B1",
     "各分块拼接恰好覆盖提示且无重叠；同一请求不同时处于活动批与暂存状态。",
     "伪模型记录每步分块区间。",
     ["chunk_size"],
     {"vllm": ["v:v1/core/sched/scheduler.py:1343-1454"],
      "sglang": ["s:managers/scheduler.py:2464-2467", "s:managers/scheduler.py:2603-2617"]},
     ["sched#6"],
     divergence="分块边界不要求跨框架一致，只要求覆盖与互斥。")
unit("sched.prefix_bound", "S3_schedule", "contract", "B1",
     "前缀命中长度满足 0 ≤ hit ≤ min(n−1, logprob 起点, 显式上限)，使最后位置和需要 logprob 的位置一定被重算。",
     "构造全命中请求与带 prompt logprob 请求，检查调度的已计算长度。",
     ["hit_bound", "logprob_mode", "cache_state"],
     {"vllm": ["v:v1/request.py:352-377", "v:v1/core/kv_cache_manager.py:322-330"],
      "sglang": ["s:managers/schedule_batch.py:1223-1228", "s:managers/schedule_policy.py:84-122"]},
     ["sched#3"],
     divergence="两者都保证 n−1；vLLM 另有 skip_reading_prefix_cache 与 prompt_logprob_start，SGLang 仅在返回 logprob 时再用 logprob_start_len 收紧。")
unit("sched.mode_mix", "S3_schedule", "contract", "B2",
     "批元数据中每个请求的计算模式与其真实进度一致；禁止混合时不出现 prefill/decode 混合批。",
     "单步检查 forward mode 与各请求已计算/调度 token 数。",
     ["forward_mode", "spec_method", "parallel"],
     {"vllm": ["v:v1/worker/gpu_model_runner.py:2418-2429"],
      "sglang": ["s:model_executor/forward_batch_info.py:60-145", "s:managers/scheduler.py:2621-2676"]},
     ["sched#7", "exec#5"],
     divergence="SGLang 有显式 MIXED/TARGET_VERIFY/SPLIT_PREFILL 模式；vLLM 由已计算与提示长度推导是否在 prefill。")
unit("sched.preempt_state", "S3_schedule", "resource", "B1",
     "被抢占/回撤的请求一定重新入队；其 KV 要么全部释放待重算，要么完整保留，不出现部分所有权。",
     "伪分配器容量压力下触发抢占，检查队列与块所有权。",
     ["preemption"],
     {"vllm": ["v:v1/core/sched/scheduler.py:946-1010", "v:v1/request.py:422-446"],
      "sglang": ["s:managers/scheduler.py:2258-2262", "s:managers/schedule_batch.py:2684-2761"]},
     ["sched#9"],
     divergence="vLLM 用 PREEMPTED 状态枚举；SGLang 通过队列/批重组表达回撤，不存在等价枚举，应检查重新入队语义而非状态值。")
unit("sched.abort", "S3_schedule", "protocol", "B1",
     "取消幂等；被取消请求最终从等待、运行、分块、挂起各结构中移除，并释放其独占资源。",
     "伪模型下在等待/运行/分块中途取消，检查目标请求的结构与资源。",
     ["abort_timing", "chunk_size", "overlap"],
     {"vllm": ["v:v1/engine/core.py:606-612", "v:v1/engine/core.py:869-877"],
      "sglang": ["s:managers/scheduler.py:3774-3815", "s:managers/scheduler.py:2464-2510"]},
     ["sched#11", "sched#12"],
     divergence="SGLang 对分块中的请求延迟取消（_pending_chunked_abort_req）；vLLM 在 EngineCore 步边界处理取消队列。只能要求最终清理，不能要求同步生效。")
unit("sched.abort_unaffected", "S3_schedule", "protocol", "B1",
     "选择性取消一个请求时，其他已接纳请求的运行资格和调度进度保持有效。",
     "同一批中取消目标请求，逐步检查其他请求仍在队列或继续执行；有最终输出时再检查正常结束。",
     ["abort_timing", "batch_shape"],
     {"vllm": ["v:v1/core/sched/scheduler.py:3124-3165"],
      "sglang": ["s:managers/scheduler.py:3874-3885"]},
     ["sched#11", "sched#12"], depends=["sched.abort"],
     divergence="vLLM 的固定测试检查另外两个请求仍在运行队列；SGLang 的 HTTP 测试检查另外两个请求最终正常完成。")

# ---- S4 state -------------------------------------------------------------------
unit("kv.conservation", "S4_state", "resource", "B1",
     "静止时 已分配 + 空闲 = 总量；分配出的物理块/页唯一且在范围内。",
     "对分配器做随机分配/释放序列（属性测试）。",
     ["page_size"],
     {"vllm": ["v:v1/core/block_pool.py:139-158", "v:v1/core/block_pool.py:628-660"],
      "sglang": ["s:mem_cache/allocator/paged.py:102-192"]},
     ["kv#1"])
unit("kv.mapping", "S4_state", "resource", "B1",
     "每个活动请求恰有一份逻辑到物理映射；映射行在其物理位置释放后才可复用；分块过渡不产生重复所有权。",
     "随机请求生命周期序列，检查映射表与分配器一致。",
     ["cache_granularity", "chunk_size"],
     {"vllm": ["v:v1/core/single_type_kv_cache_manager.py:390-467"],
      "sglang": ["s:mem_cache/memory_pool.py:229-290", "s:mem_cache/radix_cache.py:520-610"]},
     ["kv#2", "kv#10"],
     divergence="vLLM 映射按块；SGLang req_to_token 按 token 槽位，即使物理分配按页。")
unit("kv.key_completeness", "S4_state", "resource", "B1",
     "前缀键区分所有影响 KV 数值的输入：token、适配器、多模态内容、cache salt、prompt embedding；键不同输入之间不得命中。",
     "成对输入只改变一个 KV 相关因素，检查是否命中；包括构造的命名空间拼接碰撞。",
     ["lora", "multimodal", "cache_granularity"],
     {"vllm": ["v:v1/core/kv_cache_utils.py:594-676", "v:v1/core/block_pool.py:262-303"],
      "sglang": ["s:mem_cache/radix_cache.py:59-71", "s:mem_cache/radix_cache.py:311-327", "s:managers/schedule_batch.py:776-783"]},
     ["kv#3", "kv#4", "ext#17"],
     divergence="vLLM 用结构化 extra-key 元组（LoRA、多模态哈希、首块 cache_salt、prompt embeds）；SGLang 把 lora_id 直接拼接到 extra_key 字符串后，多模态身份经截断哈希编码进 token。键的编码方式不同，需要分别构造碰撞用例。")
unit("kv.hit_extent", "S4_state", "contract", "B1",
     "返回的物理映射长度等于报告的命中长度，且命中粒度符合配置（块对齐或页对齐）。",
     "在不同 page/block 大小下插入与查询，比较长度与索引。",
     ["cache_granularity", "page_size"],
     {"vllm": ["v:v1/core/kv_cache_manager.py:289-353", "v:v1/core/kv_cache_coordinator.py:464-500"],
      "sglang": ["s:mem_cache/radix_cache.py:358-435"]},
     ["kv#5"],
     divergence="vLLM 以满块命中；SGLang 基数树按 token 匹配后按 page_size 截断。")
unit("kv.no_shared_write", "S4_state", "resource", "B1",
     "请求不会写入仍被其他请求引用的物理块；部分命中时先复制或分裂。",
     "共享前缀后在非对齐位置继续生成，检查写入目标与引用计数。",
     ["cache_state", "page_size"],
     {"vllm": ["v:v1/core/single_type_kv_cache_manager.py:173-181", "v:v1/core/single_type_kv_cache_manager.py:245-263"],
      "sglang": ["s:mem_cache/radix_cache.py:408-435"]},
     ["kv#6"],
     divergence="vLLM 对非对齐本地命中预留写时复制块；SGLang 通过节点分裂暴露精确 token 边界，物理页仍受对齐约束。")
unit("kv.ref_protect", "S4_state", "resource", "B1",
     "被使用的块/节点不被驱逐；每次获取恰好对应一次释放；驱逐同时移除查找元数据。",
     "交错的锁定、释放、驱逐序列（属性测试），检查在用块安全和引用计数。",
     ["cache_state"],
     {"vllm": ["v:v1/core/block_pool.py:767-832"],
      "sglang": ["s:mem_cache/radix_cache.py:632-671", "s:mem_cache/evict_policy.py:16-62"]},
     ["kv#7", "kv#8"],
     divergence="vLLM 直接对物理块计数；SGLang 从终端节点向根传播 lock_ref，并支持 LRU/LFU/SLRU/优先级策略。")
unit("kv.eviction_priority", "S4_state", "contract", "B1",
     "给定可回收项的访问、命中与显式优先级状态，所选策略产生规定的先后关系。",
     "构造可回收项并比较队列位置或策略键；不以策略键检查代替实际驱逐。",
     ["eviction_policy"],
     {"vllm": ["v:v1/core/block_pool.py:791-839"],
      "sglang": ["s:mem_cache/evict_policy.py:12-63"]},
     ["kv#8"],
     divergence="vLLM 用空闲队列的前插/后追加表达顺序；SGLang 为叶节点计算 LRU、LFU、FIFO、MRU、FILO、优先级或 SLRU 策略键。")
unit("kv.eviction_order", "S4_state", "contract", "B1",
     "给定可驱逐块或叶节点的访问与优先级状态，回收顺序符合所选策略；优先级相同时遵守实现定义的次序。",
     "构造有不同访问时间、命中次数和显式优先级的可驱逐项，比较实际回收次序与该实现的策略键。",
     ["eviction_policy", "cache_state"],
     {"vllm": ["v:v1/core/block_pool.py:791-839", "v:v1/core/block_pool.py:696-705"],
      "sglang": ["s:mem_cache/evict_policy.py:12-63", "s:mem_cache/radix_cache.py:558-580"]},
     ["kv#8"], depends=["kv.ref_protect", "kv.eviction_priority"],
     divergence="vLLM 通过空闲队列的前插/后追加控制可复用块次序；SGLang 从可驱逐叶节点按配置策略键构建最小堆。")
unit("kv.release_terminal", "S4_state", "resource", "B1",
     "请求结束或取消后，其独占引用归零；仍被缓存持有的前缀引用保持有效；最后一个调度步之前不释放。",
     "生命周期序列结束后核对分配器与缓存所有权。",
     ["abort_timing", "overlap", "window"],
     {"vllm": ["v:v1/core/kv_cache_manager.py:650-702", "v:v1/core/sched/scheduler.py:1703-1760", "v:v1/request.py:186-189"],
      "sglang": ["s:mem_cache/radix_cache.py:467-520", "s:managers/schedule_batch.py:2522-2553"]},
     ["kv#9", "sched#14", "reqio#18"],
     depends=["kv.ref_protect", "kv.mapping"],
     divergence="vLLM 用 last_sched_seq 推迟释放；SGLang 在 cache_finished_req 中释放非缓存与未对齐尾部。")
unit("kv.reset", "S4_state", "resource", "B1",
     "重置后旧键不再命中，所有物理位置处于活动或可回收状态。",
     "填充缓存、重置、再查询并核对计数。",
     [],
     {"vllm": ["v:v1/core/block_pool.py:860-900"], "sglang": ["s:mem_cache/radix_cache.py:342-405"]},
     ["kv#15"])
unit("kv.window_recycle", "S4_state", "resource", "B2",
     "窗口外的物理位置只有在没有核函数还会读取时才被回收。",
     "滑动窗口模型上把被回收位置写入哨兵值，检查输出不变。",
     ["window"],
     {"vllm": ["v:v1/core/block_pool.py:253-303", "v:v1/core/single_type_kv_cache_manager.py:219-263"],
      "sglang": ["s:mem_cache/allocator/swa.py:309-341"]},
     ["kv#11"],
     divergence="vLLM 保留块对象并用空块屏蔽；SGLang 按窗口回收 token 槽位，逻辑映射可能不连续。")
unit("kv.hybrid_boundary", "S4_state", "contract", "B2",
     "每个复用边界上所有带状态的层组（注意力、Mamba 等）都有有效状态。",
     "混合模型上比较不同命中边界的输出与无缓存输出。",
     ["window", "cache_state"],
     {"vllm": ["v:v1/core/single_type_kv_cache_manager.py:311-345", "v:v1/core/kv_cache_coordinator.py:943-990"],
      "sglang": ["s:mem_cache/memory_pool.py:646-884"]},
     ["kv#12"], status="extension")
unit("kv.tier_coherence", "S4_state", "resource", "B2",
     "多级缓存中每个分配恰由一层持有；提升/恢复的值等于原 KV。",
     "强制卸载后恢复，比较 KV 张量与所有权计数。",
     ["cache_state"],
     {"vllm": ["v:v1/core/block_pool.py:222-303"],
      "sglang": ["s:mem_cache/radix_cache.py:235-285", "s:mem_cache/hiradix_cache.py:1008-1113"]},
     ["kv#13"], status="extension",
     divergence="vLLM 以按介质划分的块池/连接器表示层级；SGLang 把主机副本与引用嵌入基数树节点。")
unit("kv.events", "S4_state", "protocol", "B1",
     "对外的缓存事件流与内部状态迁移一一对应：每次存入/移除恰好一个事件，事件键与查找命名空间一致。",
     "记录事件流并回放出缓存集合，与实际缓存集合比较。",
     ["kv_events"],
     {"vllm": ["v:v1/core/block_pool.py:287-303", "v:v1/core/block_pool.py:834-860"],
      "sglang": ["s:mem_cache/radix_cache.py:467-610", "s:mem_cache/radix_cache_cpp.py:45-47"]},
     ["kv#14"], status="extension",
     divergence="SGLang HiRadixCache 断言不支持事件（约束 K1）。")
unit("lora.residency", "S4_state", "resource", "B1",
     "批执行期间该批引用的适配器都驻留且槽位不被替换；驻留数不超过容量。",
     "适配器数超过槽位的负载下检查每批槽位与适配器身份。",
     ["lora"],
     {"vllm": ["v:lora/model_manager.py:79-120", "v:lora/model_manager.py:340-360"],
      "sglang": ["s:lora/lora_manager.py:165-200", "s:lora/lora_manager.py:272-330"]},
     ["ext#6", "ext#8"],
     divergence="SGLang 有显式 validate_lora_batch；vLLM 由管理器缓存与 Punica 集成。")
unit("grammar.state", "S4_state", "protocol", "B1",
     "每个请求的语法状态只按其已接受 token 前进；位掩码行与请求一一对应。",
     "多语法请求打乱批顺序，检查每行掩码与参考解析器。",
     ["structured_output", "batch_shape"],
     {"vllm": ["v:v1/structured_output/backend_types.py:36-80", "v:v1/structured_output/__init__.py:285-300"],
      "sglang": ["s:constrained/base_grammar_backend.py:42-80"]},
     ["exec#18"])
unit("spec.kv_rollback", "S4_state", "resource", "B2",
     "每次验证后已分配 KV 长度等于 提示 + 已接受 token，被拒绝草稿的 KV 被回收或不可见。",
     "多轮验证后核对序列长度、空闲块数，并重复运行检查泄漏。",
     ["spec_method"],
     {"vllm": ["v:v1/core/kv_cache_manager.py:401-460", "v:v1/sample/rejection_sampler.py:415-430"],
      "sglang": ["s:speculative/eagle_worker_v2.py:201-230", "s:speculative/ngram_worker.py:201-230"]},
     ["ext#18", "ext#2"])

# ---- S5 binding ----------------------------------------------------------------------
unit("bind.slot_reset", "S5_binding", "resource", "B2",
     "复用的批内请求槽不残留上一个请求的采样参数、惩罚历史、掩码或随机数生成器。",
     "先放入特殊参数请求再替换，检查新请求行的状态。",
     ["batch_shape"],
     {"vllm": ["v:v1/worker/gpu_input_batch.py:196-282"],
      "sglang": ["s:model_executor/forward_batch_info.py:606-691"]},
     ["exec#1"],
     divergence="vLLM 在 InputBatch 中持有异构采样状态；SGLang 从 ScheduleBatch 借用核心字段到 ForwardBatch。")
unit("bind.positions", "S5_binding", "contract", "B2",
     "展平的 token 与位置与每个请求的已计算进度一致（含分块偏移、多维 RoPE）。",
     "构造分块/混合批，比较位置张量与参考计算。",
     ["chunk_size", "forward_mode", "multimodal"],
     {"vllm": ["v:v1/worker/gpu_model_runner.py:1832-1885", "v:v1/worker/gpu_model_runner.py:2016-2140"],
      "sglang": ["s:model_executor/forward_batch_info.py:778-849"]},
     ["exec#2"])
unit("bind.boundaries", "S5_binding", "contract", "B2",
     "行边界为累积长度 [0,a,a+b,...]，填充尾部保持非减常数。",
     "元数据不变式检查。",
     ["batch_shape", "graph"],
     {"vllm": ["v:v1/worker/gpu_model_runner.py:2125-2131"],
      "sglang": ["s:model_executor/forward_batch_info.py:822-829"]},
     ["exec#3"])
unit("bind.slot_mapping", "S5_binding", "resource", "B2",
     "本步 KV 写入位置都属于对应请求，且与逻辑映射一致。",
     "以哨兵 KV 池执行一步，检查被写位置集合。",
     ["page_size", "cache_state"],
     {"vllm": ["v:v1/worker/gpu_model_runner.py:2248-2252", "v:v1/worker/gpu_model_runner.py:2414-2469"],
      "sglang": ["s:model_executor/forward_batch_info.py:325-350", "s:model_executor/forward_batch_info.py:674-680"]},
     ["exec#4"], depends=["kv.mapping"])
unit("bind.graph_padding", "S5_binding", "equivalence", "B2",
     "图重放对真实行的输出等于即时执行；填充行不写 KV、不影响真实行。",
     "哨兵 KV/logit 行 + 图与即时结果比较，覆盖桶边界。",
     ["graph", "batch_shape"],
     {"vllm": ["v:v1/attention/backend.py:726-737", "v:v1/worker/gpu_model_runner.py:2368-2469"],
      "sglang": ["s:model_executor/runner/base_cuda_graph_runner.py:61-158", "s:model_executor/forward_batch_info.py:1314-1365"]},
     ["exec#7"], depends=["bind.boundaries", "bind.slot_mapping"], perturbation="graph_vs_eager",
     gap="逐行元数据正确不足以排除填充行通过共享缓冲区写入 KV；需要哨兵观察。")

# ---- S6 compute ------------------------------------------------------------------------
unit("op.attention", "S6_compute", "value", "B0",
     "各后端在 prefill/decode/mixed 与分页布局下，注意力输出与数值稳定参考实现在容差内一致。",
     "FP32 参考注意力；容差按 dtype 声明。",
     ["attention_backend", "dtype_quant", "forward_mode", "page_size", "window"],
     {"vllm": ["v:v1/attention/backend.py:386-407", "v:v1/attention/backend.py:598-737"],
      "sglang": ["s:model_executor/forward_batch_info.py:438-501"]},
     ["exec#6"])
unit("op.layers", "S6_compute", "value", "B0",
     "归一化、旋转位置、线性与量化层输出与 FP32 参考在逐层容差内一致；不应改动的残差不被原地修改。",
     "逐层参考 + 输入张量快照比较。",
     ["dtype_quant"],
     {"vllm": ["v:model_executor/layers/linear.py:407-437"], "sglang": ["s:layers/linear.py:190-277"]},
     ["exec#10"])
unit("op.moe", "S6_compute", "value", "B0",
     "固定路由 logits 下，专家选择与输出等于参考实现；填充行不占用专家容量。",
     "参考 MoE；填充行注入极端值检查真实行不变。",
     ["dtype_quant", "batch_shape"],
     {"vllm": ["v:v1/worker/gpu_model_runner.py:700-850"], "sglang": ["s:layers/moe/ep_moe/layer.py:181-231"]},
     ["exec#11"])
unit("op.lora_apply", "S6_compute", "value", "B0",
     "带适配器的层输出等于 W + scale·B·A 的显式参考；批内按请求选择正确的适配器。",
     "逐层合并权重参考。",
     ["lora", "dtype_quant"],
     {"vllm": ["v:lora/layers/base_linear.py:162-220"], "sglang": ["s:lora/layers.py:48-130"]},
     ["ext#7"])
unit("op.compiled", "S6_compute", "equivalence", "B2",
     "编译执行与即时执行在桶边界两侧的输出与副作用一致。",
     "仅改变批形状、不改变真实请求的变形测试。",
     ["graph", "batch_shape"],
     {"vllm": ["v:compilation/backends.py:1-60"], "sglang": ["s:model_executor/cpu_graph_runner.py:88-113"]},
     ["exec#12"], perturbation="compile_vs_eager")

# ---- S7 sampling --------------------------------------------------------------------------
unit("samp.processor_order", "S7_sampling", "value", "B0",
     "logits 处理按声明顺序作用：白名单/禁用词 → 影响 argmax 的处理器 → 惩罚 → 温度与过滤；惩罚只使用本请求历史。",
     "手算 logits 参考。",
     ["sampling"],
     {"vllm": ["v:v1/sample/sampler.py:78-180"], "sglang": ["s:layers/sampler.py:93-190"]},
     ["exec#13", "exec#14"],
     divergence="SGLang 在 _preprocess_logits 集中处理自定义处理器与 NaN 清理；处理顺序作为框架条件保留。")
unit("samp.filter", "S7_sampling", "distribution", "B0",
     "贪心为精确 argmax；top-k/top-p/min-p 的支撑集与参考一致；过滤后重新归一化。",
     "支撑集精确比较 + 大样本频率检验。",
     ["sampling", "attention_backend"],
     {"vllm": ["v:v1/sample/sampler.py:308-368"], "sglang": ["s:layers/sampler.py:214-316"]},
     ["exec#15"])
unit("samp.logprob", "S7_sampling", "contract", "B0",
     "返回的 logprob 与所选模式（原始/处理后）一致，与输出 token 一一对齐，top-k 数量符合请求。",
     "原始 log_softmax 参考 + 逐 token 对齐检查。",
     ["logprob_mode"],
     {"vllm": ["v:v1/sample/sampler.py:129-180", "v:v1/engine/logprobs.py:32-75"],
      "sglang": ["s:layers/sampler.py:119-190", "s:managers/tokenizer_manager.py:2106-2299"]},
     ["exec#17", "reqio#12"],
     divergence="vLLM 显式区分 raw_logprobs/raw_logits 并在处理器前捕获；SGLang 由环境变量选择返回原始 logprob。")
unit("samp.grammar_mask", "S7_sampling", "contract", "B0",
     "非法 token 的 logits 为 -inf；采样 token 一定被请求的语法接受。",
     "语法解析器参考 + 掩码检查。",
     ["structured_output"],
     {"vllm": ["v:v1/structured_output/utils.py:101-194", "v:v1/worker/gpu/structured_outputs.py:33-97"], "sglang": ["s:constrained/base_grammar_backend.py:69-80"]},
     ["exec#18"])
unit("spec.distribution", "S7_sampling", "distribution", "B0",
     "推测采样的接受/拒绝与补偿采样使下一 token 分布等于目标模型分布；贪心是其特例。",
     "固定目标/草稿概率的大样本检验。",
     ["spec_method", "sampling"],
     {"vllm": ["v:v1/sample/rejection_sampler.py:125-200", "v:v1/sample/rejection_sampler.py:589-620"],
      "sglang": ["s:speculative/reject_sampling.py:6-60", "s:speculative/reject_sampling.py:159-200"]},
     ["ext#2", "ext#3"])

# ---- S8 output ----------------------------------------------------------------------------
unit("out.detok", "S8_output", "equivalence", "B1",
     "增量输出的拼接加最终刷新等于整体解码结果；不在 UTF-8 半字符处截断或重复。",
     "对任意 token 序列逐 token 喂入，比较拼接与一次性解码。",
     ["stream"],
     {"vllm": ["v:v1/engine/detokenizer.py:32-180"], "sglang": ["s:managers/detokenizer_manager.py:62-89", "s:managers/detokenizer_manager.py:205-404"]},
     ["reqio#9"], perturbation="incremental_vs_whole")
unit("out.stop", "S8_output", "contract", "B1",
     "命中停止条件后，返回文本按配置排除或包含停止串，且第一个生效停止之后不再输出文本；停止 token 的可见性与计数遵循配置。",
     "停止串跨 token 边界、多个停止串、包含停止串三类表驱动用例。",
     ["stop_config", "stream"],
     {"vllm": ["v:v1/engine/detokenizer.py:98-160", "v:v1/engine/detokenizer.py:224-262"],
      "sglang": ["s:managers/detokenizer_manager.py:169-199"]},
     ["reqio#10", "reqio#11"],
     divergence="SGLang 只处理一个匹配到的停止串（源码 TODO），并假定最后一个 token 就是停止 token；vLLM 将 stop_terminated 传入增量解码器。")
unit("out.parse_tool", "S8_output", "contract", "B0",
     "工具调用解析结果符合规范化格式；普通文本不会被识别为工具调用。",
     "按格式的正反例表。",
     ["parser"],
     {"vllm": ["v:tool_parsers/abstract_tool_parser.py:201-214"], "sglang": ["s:function_call/function_call_parser.py:118-139"]},
     ["ext#13"])
unit("out.parse_reasoning", "S8_output", "contract", "B0",
     "推理与可见内容的划分能重构原文本，分隔符不泄漏。",
     "分隔符缺失/重复/嵌套用例。",
     ["parser"],
     {"vllm": ["v:reasoning/abs_reasoning_parsers.py:118-160"], "sglang": ["s:parser/reasoning_parser.py:62-112"]},
     ["ext#15"],
     divergence="vLLM 推理解析基于 token ID；SGLang 基于文本。")
unit("out.parse_incremental", "S8_output", "equivalence", "B1",
     "流式解析的最终结果等于对完整文本的一次解析；每段增量不重复输出，分隔符在任意切分点都正确。",
     "对同一文本枚举所有切分点。",
     ["parser", "stream"],
     {"vllm": ["v:tool_parsers/abstract_tool_parser.py:214-240", "v:reasoning/abs_reasoning_parsers.py:173-200"],
      "sglang": ["s:function_call/function_call_parser.py:139-170", "s:parser/reasoning_parser.py:112-150"]},
     ["ext#14", "ext#16"], depends=["out.parse_tool", "out.parse_reasoning"], perturbation="incremental_vs_whole")

unit("out.grammar_valid", "S8_output", "contract", "B3",
     "受语法约束的请求，其完整输出被该语法接受（含推测解码与异步语法编译）。",
     "用参考解析器检查完整输出；开关推测解码各执行一次。",
     ["structured_output", "spec_method", "batch_shape"],
     {"vllm": ["v:v1/structured_output/backend_types.py:64-72", "v:v1/structured_output/__init__.py:285-300"],
      "sglang": ["s:constrained/base_grammar_backend.py:42-80", "s:constrained/grammar_manager.py:1-60"]},
     ["exec#18"], depends=["grammar.state", "samp.grammar_mask"],
     gap="单步掩码正确时，推测 token 被拒绝后语法状态的回滚（vLLM backend_types.rollback）与异步编译完成时机决定整段输出是否合法。")

# ---- end-to-end request semantics (B3/B4) -----------------------------------------------------
unit("out.exactly_once", "S8_output", "protocol", "B3",
     "每个生成 token 按序恰好处理一次；每个请求恰好一个终止输出；被接受的请求不会无声丢失。",
     "注入调度扰动后检查输出序列号与终止事件。",
     ["overlap", "abort_timing"],
     {"vllm": ["v:v1/engine/output_processor.py:731-920", "v:v1/engine/core_client.py:191-200"],
      "sglang": ["s:managers/tokenizer_manager.py:1839-1947", "s:managers/detokenizer_manager.py:269-404"]},
     ["reqio#6", "reqio#8", "sched#10"], depends=["sched.progress", "sched.abort"],
     gap="调度器局部计数正确时，异步步之间的过期输出仍可能被重复或遗漏处理（vLLM request.py:166-185 的在途占位）。")
unit("out.finish_reason", "S8_output", "contract", "B3",
     "每个终止的输出选项恰有一个结束原因，且与实际终止原因（停止、长度、取消、错误）和最后输出一致。",
     "按终止原因构造请求，检查原因与最后文本/token。",
     ["stop_config", "abort_timing"],
     {"vllm": ["v:v1/engine/output_processor.py:425-532"], "sglang": ["s:managers/tokenizer_manager.py:1381-1424"]},
     ["reqio#14"], depends=["out.stop", "sched.abort"])
unit("out.no_late_output", "S8_output", "protocol", "B3",
     "取消生效后到达的旧计算结果不得作为正常响应交付。",
     "使取消与已启动计算交叠，记录请求身份、取消事件和随后到达的结果，检查无迟到正常交付。",
     ["abort_timing", "overlap", "stream"],
     {"vllm": ["v:v1/engine/output_processor.py:762-767"],
      "sglang": ["s:managers/tokenizer_manager.py:1848-1858"]},
     ["reqio#14", "sched#11"], depends=["sched.abort"],
     gap="请求状态已删除的局部跳过逻辑仍需与实际取消事件、在途输出和客户端交付组成同一次序列。")
unit("out.stream_parity", "S8_output", "equivalence", "B3",
     "同一请求流式各选项增量的拼接，与非流式最终响应在文本、工具调用、logprob、用量上等价。",
     "同种子同配置两次执行比较。",
     ["stream", "n", "parser", "stop_config"],
     {"vllm": ["v:entrypoints/openai/chat_completion/stream_processor.py:17-52", "v:entrypoints/openai/chat_completion/serving.py:966-1200"],
      "sglang": ["s:entrypoints/openai/serving_chat.py:339-428", "s:entrypoints/openai/serving_chat.py:1207-1376"]},
     ["reqio#13", "reqio#17"], depends=["out.detok", "out.stop", "out.parse_incremental", "out.usage"],
     perturbation="stream_vs_nonstream",
     gap="两个框架流式与非流式是两条独立代码路径；局部解码与解析正确不能推出两条路径的组装一致。")
unit("out.n_fanout", "S8_output", "protocol", "B3",
     "n>1 时返回恰好 n 个带正确索引的选项；所有选项结束后才结束聚合。",
     "n>1 流式与非流式检查索引与结束时机。",
     ["n", "stream"],
     {"vllm": ["v:v1/engine/output_processor.py:70-84", "v:v1/engine/output_processor.py:184-257"],
      "sglang": ["s:managers/tokenizer_manager.py:1839-1950"]},
     ["reqio#7"])
unit("out.usage", "S8_output", "contract", "B3",
     "prompt + completion = total；流式最终用量等于非流式用量；n 与停止 token 的计数一致。",
     "按配置组合核对计数。",
     ["n", "stream", "stop_config", "cache_state"],
     {"vllm": ["v:v1/engine/output_processor.py:257-277"], "sglang": ["s:managers/tokenizer_manager.py:2363-2426"]},
     ["reqio#16"])

EQ_SITES_SCHED = {"vllm": ["v:v1/core/sched/scheduler.py:946-1010", "v:v1/request.py:166-185"],
                  "sglang": ["s:managers/scheduler.py:2555-2676", "s:managers/schedule_batch.py:724-759"]}

unit("eq.chunked_prefill", "S6_compute", "equivalence", "B3",
     "分块与不分块 prefill 得到等价的最终 logits/KV，从而在贪心下输出相同。",
     "变形测试：只改变 chunk 大小。",
     ["chunk_size", "attention_backend", "dtype_quant"],
     {"vllm": ["v:v1/worker/gpu_model_runner.py:2643-2680"], "sglang": ["s:model_executor/model_runner.py:2869-2894"]},
     ["exec#9", "sched#6"], depends=["sched.chunking", "bind.positions", "bind.slot_mapping", "op.attention"],
     perturbation="chunking",
     gap="分块使注意力分别走 extend 与 prefill 内核；逐算子各自在容差内不保证累积后贪心 token 相同，需在请求级判定并声明容差策略。")
unit("eq.prefix_hit", "S4_state", "equivalence", "B3",
     "前缀缓存命中时的输出与无缓存执行等价（贪心 token 相同；prompt logprob 在容差内）。",
     "同一请求冷/热两次执行，覆盖部分命中、页边界与 prompt logprob。",
     ["cache_state", "cache_granularity", "page_size", "logprob_mode", "lora", "multimodal"],
     {"vllm": ["v:v1/core/kv_cache_manager.py:289-353"], "sglang": ["s:mem_cache/radix_cache.py:358-435"]},
     ["kv#3", "kv#5", "sched#3"],
     depends=["kv.key_completeness", "kv.hit_extent", "kv.no_shared_write", "kv.ref_protect", "sched.prefix_bound", "bind.slot_mapping", "op.attention"],
     perturbation="prefix_cache",
     gap="键完整与映射正确只保证读到“同一输入”算出的 KV；该 KV 是由另一批组成、另一内核路径算出的，数值是否足以使结果一致只能在请求级观察。")
unit("eq.preemption", "S3_schedule", "equivalence", "B3",
     "被抢占/回撤后恢复的请求，在贪心下与不中断执行输出相同且结束原因相同。",
     "通过缩小 KV 容量强制抢占的变形测试。",
     ["preemption", "overlap", "cache_state"],
     EQ_SITES_SCHED, ["sched#9", "sched#15"],
     depends=["sched.preempt_state", "kv.release_terminal", "eq.prefix_hit"], perturbation="preemption",
     gap="调度器状态与 KV 计数正确时，异步在途输出仍可能被重复使用；重算路径又依赖前缀命中等价。")
unit("eq.overlap", "S3_schedule", "equivalence", "B3",
     "开启异步/重叠调度时，每个请求的输出与同步调度相同。",
     "同负载开关比较。",
     ["overlap"],
     {"vllm": ["v:v1/engine/core.py:709-869"], "sglang": ["s:managers/scheduler.py:1505-1594", "s:managers/scheduler.py:3085-3145"]},
     ["sched#10"], depends=["out.exactly_once", "kv.release_terminal"], perturbation="overlap",
     gap="重叠执行改变状态读写时序，属于不能由单步局部性质推出的时序依赖。")
unit("eq.batch_invariance", "S7_sampling", "equivalence", "B4",
     "请求的输出不随同批其他请求的存在、顺序或填充而改变（贪心或固定种子）。",
     "单请求、批量、打乱、填充四种执行比较。",
     ["batch_shape", "sampling", "attention_backend", "dtype_quant"],
     {"vllm": ["v:v1/sample/sampler.py:1-70", "v:v1/worker/gpu_input_batch.py:250-279"],
      "sglang": ["s:layers/sampler.py:274-301"]},
     ["exec#1", "exec#16"],
     depends=["bind.slot_reset", "bind.graph_padding", "samp.processor_order", "op.moe"], perturbation="batch_composition",
     divergence="SGLang 使用基于位置的种子并要求 sampling_seed 才保证确定性 logprob 采样；vLLM 使用逐请求生成器。",
     gap="批大小改变归约顺序与内核选择；即使每个算子各自符合容差，批不变性也不能由算子容差推出，除非启用批不变内核。")
unit("eq.lora_isolation", "S6_compute", "equivalence", "B4",
     "不同适配器的请求同批执行，与各自单独执行结果相同；前缀缓存不跨适配器命中。",
     "混合适配器批与单独执行比较。",
     ["lora", "cache_state"],
     {"vllm": ["v:lora/model_manager.py:194-219"], "sglang": ["s:lora/lora_manager.py:272-330"]},
     ["ext#8"], depends=["lora.residency", "op.lora_apply", "kv.key_completeness"], perturbation="adapter_mix",
     gap="逐层 W+BA 正确与槽位驻留各自成立时，批内 token 到适配器索引的对应关系（含分块与图填充行）只在混合批的请求输出中可见。")
unit("eq.mm_cache", "S2_render", "equivalence", "B3",
     "多模态编码缓存命中与不命中时，嵌入与输出相同；不同媒体不共享前缀 KV。",
     "同文本异图、同图异配置的冷/热执行。",
     ["multimodal", "cache_state"],
     {"vllm": ["v:multimodal/processing/processor.py:59-90"], "sglang": ["s:managers/mm_utils.py:82-120"]},
     ["ext#11", "ext#17"], depends=["mm.identity", "mm.placeholder", "kv.key_completeness"], perturbation="encoder_cache",
     gap="身份、占位与键各自正确时，缓存条目的 dtype/设备/偏移复用错误只表现为嵌入数值差异。")
unit("eq.spec_greedy", "S7_sampling", "equivalence", "B3",
     "贪心推测解码的输出与不开推测的贪心输出相同。",
     "开关推测的变形测试，覆盖链式、树式与 n-gram。",
     ["spec_method", "graph"],
     {"vllm": ["v:v1/spec_decode/draft_model.py:20-90", "v:v1/spec_decode/ngram_proposer.py:1-40"],
      "sglang": ["s:speculative/eagle_worker_v2.py:268-300", "s:speculative/eagle_worker_v2.py:469-600", "s:speculative/ngram_worker.py:131-160"]},
     ["ext#1", "ext#4", "ext#5"], depends=["spec.distribution", "spec.kv_rollback", "bind.positions"], perturbation="speculation",
     divergence="SGLang EAGLE 可形成树式草稿并维护独立草稿 worker 与图缓冲；树宽/深是独立条件。",
     gap="接受判定、KV 回滚与位置各自正确时，树式草稿的位置/掩码组合错误只在请求级可见。")
unit("eq.tier_restore", "S4_state", "equivalence", "B3",
     "KV 被卸载到低层后恢复命中，输出与未卸载时相同。",
     "强制卸载后的冷/热比较。",
     ["cache_state"],
     {"vllm": ["v:v1/core/block_pool.py:222-303"], "sglang": ["s:mem_cache/hiradix_cache.py:1008-1113"]},
     ["kv#13"], depends=["kv.tier_coherence", "eq.prefix_hit"], perturbation="offload", status="extension",
     gap="异步写回尚未完成时设备副本已被驱逐，属于时序问题，单独的所有权与数值检查都不能覆盖。")

# ---- S9 topology -----------------------------------------------------------------------------
unit("topo.shared_slot", "S9_topology", "resource", "B5",
     "共享内存槽位在写者发布后交付原值；仍被读者使用的槽位不被复用，全部读者完成后才可回收；回绕不覆盖有效数据。",
     "用不同容量、读者数量和读者完成顺序驱动写入/读取/回绕，逐次核对槽位身份、读回值与可复用状态。",
     ["shared_buffer_kind", "reader_count", "buffer_wraparound"],
     {"vllm": ["v:distributed/device_communicators/shm_object_storage.py:218-318",
                "v:distributed/device_communicators/shm_broadcast.py:732-804"],
      "sglang": ["s:distributed/device_communicators/shm_broadcast.py:32-85",
                 "s:distributed/device_communicators/shm_broadcast.py:339-432"]},
     ["transport#1", "transport#2"],
     divergence="vLLM 的对象存储用单调 ID、内存区间和可回收回调管理变长数据；SGLang 的消息环用固定槽及每个读者的完成标志。")
unit("topo.groups", "S9_topology", "contract", "B0",
     "给定 world size 与 TP/PP/DP/EP 配置，各进程组成员集合互不重叠且符合预期排布；每个进程的组内秩一致。",
     "纯函数枚举：无需 GPU，对所有小规模配置计算组成员。",
     ["parallel"],
     {"vllm": ["v:distributed/parallel_state.py:2085-2177", "v:distributed/parallel_state.py:2295-2318"],
      "sglang": ["s:distributed/parallel_state.py:1853-2127"]},
     ["dist#1", "dist#2"],
     divergence="SGLang 另有 attention-TP、MoE-TP、MoE-EP、MoE-DP 独立组；成员集合需分别比较。")
unit("topo.shard_load", "S9_topology", "value", "B1",
     "按秩加载的列/行并行、融合 QKV 与量化权重分片拼接（反量化后）等于检查点张量。",
     "模拟各秩加载同一检查点，无需通信。",
     ["parallel", "dtype_quant"],
     {"vllm": ["v:model_executor/layers/linear.py:439-619", "v:model_executor/layers/linear.py:745-966", "v:model_executor/layers/linear.py:74-122"],
      "sglang": ["s:layers/linear.py:291-466", "s:layers/linear.py:489-712", "s:layers/linear.py:85-128"]},
     ["dist#5", "dist#6", "dist#7", "dist#8"],
     divergence="SGLang 另行处理 BitsAndBytes 4bit 偏移；vLLM 处理 Marlin/块缩放偏移。")
unit("topo.collective", "S9_topology", "value", "B5",
     "自定义 all-reduce、PyNCCL、对称内存等通信路径与标准集合通信结果一致，图捕获内外均成立。",
     "随机张量对照 torch.distributed 参考。",
     ["parallel", "graph"],
     {"vllm": ["v:distributed/parallel_state.py:730-803", "v:distributed/device_communicators/pynccl.py:29-181"],
      "sglang": ["s:distributed/communication_op.py:18-84", "s:distributed/device_communicators/quick_all_reduce.py:52-266"]},
     ["dist#3", "dist#4"])
unit("topo.parallel_equiv", "S9_topology", "equivalence", "B5",
     "TP/PP/DP/EP 配置下的请求输出与单卡配置在容差内一致，请求不丢失不重复。",
     "同一请求集在不同并行配置下比较。",
     ["parallel", "dtype_quant"],
     {"vllm": ["v:distributed/parallel_state.py:2085-2177", "v:config/parallel.py:913-969"],
      "sglang": ["s:managers/data_parallel_controller.py:125-181", "s:managers/scheduler_pp_mixin.py:806-869"]},
     ["dist#1", "dist#9", "dist#10", "dist#11"],
     depends=["topo.groups", "topo.shard_load", "topo.collective"], perturbation="parallelism",
     gap="分片与通信各自正确时，流水级间微批顺序、DP 路由与注意力 TP 归一化仍可能出错（gpu_model_runner/model_runner 的 DP/TP 填充计数）。")
unit("topo.ep_dispatch", "S9_topology", "value", "B5",
     "专家并行的分发与合并保持 token 数量与顺序；输出等于复制专家参考；各秩专家映射一致。",
     "固定路由的参考比较 + 映射一致性检查。",
     ["parallel"],
     {"vllm": ["v:model_executor/layers/fused_moe/modular_kernel.py:288-480"], "sglang": ["s:layers/moe/ep_moe/layer.py:181-231", "s:eplb/lplb_solver.py:32-61"]},
     ["dist#12", "dist#13"], depends=["op.moe"], status="extension")
unit("topo.pd_transfer", "S9_topology", "equivalence", "B5",
     "PD 分离中解码端收到的 KV 等于本地计算的 KV；传输失败被检测且不被消费。",
     "同请求本地/分离执行比较 + 故障注入。",
     ["parallel", "page_size"],
     {"vllm": ["v:distributed/kv_transfer/kv_transfer_state.py:19-93"], "sglang": ["s:disaggregation/base/conn.py:97-213"]},
     ["dist#14"], depends=["kv.mapping", "eq.prefix_hit"], perturbation="pd_disaggregation", status="extension",
     gap="两端各自的布局正确时，握手把哪个请求的哪些页发给谁、超时是否被当作成功，只在端到端传输中可见。")
unit("topo.weight_update", "S9_topology", "protocol", "B5",
     "在线权重更新完成后所有秩参数等于给定张量；没有请求观察到新旧混合版本。",
     "更新前后输出对照 + 更新期间并发请求检查。",
     ["parallel"],
     {"vllm": ["v:v1/worker/gpu_model_runner.py:1-40"], "sglang": ["s:weight_sync/utils.py:14-101", "s:weight_sync/tensor_bucket.py:19-90"]},
     ["dist#15"], status="extension")

PERTURBATIONS = {
    "graph_vs_eager": "CUDA 图重放/填充 与 即时执行",
    "compile_vs_eager": "编译执行 与 即时执行",
    "incremental_vs_whole": "增量处理 与 整体处理",
    "stream_vs_nonstream": "流式 与 非流式组装",
    "chunking": "分块 prefill 与 整体 prefill",
    "prefix_cache": "前缀缓存命中 与 冷启动",
    "preemption": "抢占/回撤后恢复 与 不中断",
    "overlap": "异步/重叠调度 与 同步",
    "batch_composition": "单请求、批量、乱序、填充",
    "adapter_mix": "混合适配器批 与 单独执行",
    "encoder_cache": "编码缓存命中 与 不命中",
    "speculation": "推测解码 与 普通解码",
    "offload": "卸载恢复 与 常驻",
    "parallelism": "多卡并行配置 与 单卡",
    "pd_disaggregation": "PD 分离 与 本地执行",
}

model = {
    "version": "source-model-v1",
    "snapshots": SNAPSHOTS,
    "axes": {"stage": STAGES, "kind": KINDS, "boundary": BOUNDARIES},
    "conditions": CONDITIONS,
    "constraints": CONSTRAINTS,
    "perturbations": PERTURBATIONS,
    "units": U,
}

if __name__ == "__main__":
    (HERE / "space_model.json").write_text(json.dumps(model, ensure_ascii=False, indent=1))
    print(len(U), "units")
