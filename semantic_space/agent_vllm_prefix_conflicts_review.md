# vLLM 产品调用领域冲突的源码核查

以固定 vLLM 源码 `76a0ccc209343971c6f4b56f6380a76202428385` 为依据，把 [`bulk-call-oracle-links.tsv`](bulk-call-oracle-links.tsv) 中 `bulk_status=source_linked` 的 vLLM 入口与 [`test-entry-classification.tsv`](test-entry-classification.tsv) 对齐。按当前 `PRODUCT_PREFIX_AREAS` 推断候选调用的领域，筛得 43 条与原 `primary_area` 不一致的记录；已逐项回读调用、所列断言及测试体上下文，明细见 [`agent_vllm_prefix_conflicts_review.tsv`](agent_vllm_prefix_conflicts_review.tsv)。可用 [`agent_vllm_prefix_conflicts_build.py`](agent_vllm_prefix_conflicts_build.py) 复现筛选和逐项表。

这 43 条不能统称为“原归类错误”。18 条原候选领域更贴近被检查性质，4 条产品前缀领域更合适，12 条两者都不适合，6 条需要同时保留两种领域或边界条件，2 条候选产品调用只用作断言的参考/解释工具，另 1 条构造器只是后续检查的准备材料。43 条中 33 条原候选为 `ingress`，41 条前缀推为 `model`；集中原因是多模态模型把输入处理与输出解析方法也放在 `vllm.model_executor.models` 下，模块位置不能直接代表验证性质。

具体例子表明应以断言对象为中心。`tests/models/multimodal/processing/test_llava_onevision2.py:275` 调用模型包里的视频后端，但断言选取帧数及首尾位置，属于输入视频帧选择。`tests/models/multimodal/processing/test_moss_transcribe_diarize.py:55` 等 10 条直接检查转写 segment 的文本、时间和说话人，原路径标成输入、产品前缀标成模型，主要性质实际是输出解析。`tests/models/multimodal/processing/test_qwen2_5_omni_embed.py:418` 将视频、音频 embedding 放进相应 token 位置，跨越输入模态绑定与模型计算表示，适合保留边界关系。`tests/v1/sample/test_diffusion_gemma_reads.py:382` 的 logits mask 函数虽然在模型包内，断言针对无约束采样行不改变 logits 对象，原 sampling 候选仍有依据。

还有三类链接不应拿来给测试定领域。`tests/config/test_config_generation.py:37` 的 `deep_compare` 只是比较两份 `EngineArgs.create_engine_config` 结果的断言工具；`tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py:1427` 的 `get_offload_block_hash` 只是把 scheduler 产生的 offered keys 解码成待比较 hash；`tests/lora/test_moe_lora_ep_load.py:46` 的 `ExpertMapManager` 构造器提供 placement 状态，真正的非连续专家放置拒绝由后续 `FusedMoEWithLoRA._ep_check` 检查。这些局部调用与断言在语法上相连，但不承担整项测试的核心性质。

对下一轮批量处理，建议将产品前缀用作**候选证据**，同时保留原候选领域和调用角色。先用调用结果到断言的流向区分被测结果、准备对象、参考计算和输出解码；再按断言字段与异常条件判定输入接受、模型计算、输出解析、缓存/传输或配置性质。涉及多模态 embedding 合并、路径校验、LoRA 包装下参数查找等边界时允许多领域关联。对经 helper 才能判断的入口暂标未决，不以 `source_linked` 直接升级为确定的 Verification Space 单元。本次核查是固定源码的局部语义审查，没有运行这些测试。
