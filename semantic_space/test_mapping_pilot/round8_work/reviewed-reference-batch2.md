# vLLM kernel 数值参考对照源码审查：第二批

在固定的 vLLM 2026-11-05 源码快照中，从尚未审查的 `tests/kernels/` 候选里，按文件核对待测调用、参考值构造和数值断言。本批确认 42 个文件的 164 个源码级测试入口具有局部输出对照；表中每行保留测试定义、断言与参考构造的源码行，以及参数、后端门禁和误差规则。按目录计，Attention 30 项、MoE 32 项、量化 28 项、core 19 项，其余 kernel 文件 55 项。这里记录的是源码中的检查关系，没有运行测试，也没有推论模型或服务结果。

几个需要区分的对照关系：`tests/kernels/core/test_layernorm.py:69-78` 将 RMSNorm 待测路径与 `forward_native` 比较，并分别检查有残差时的两个返回值；`:250-260` 的 Gemma 变体则以 FP32 公式计算参考值后转回输入精度。`tests/kernels/attention/test_flashinfer.py:370-384` 使用同一分页 KV 输入构造 `ref_paged_attn` 参考结果，按给定容差比较 Decode 输出；Prefill 与 FP8 KV 的断言、容差各自记录。`tests/kernels/moe/test_block_fp8.py:268-315` 的两个 MoE 路径使用不同的舍入边界，ROCm 大尺寸时为模块化路径另造 FP32 SiLU 参考值，因此没有把两条路径写成同一条无条件等价关系。`tests/kernels/test_fused_minimax_m3_qknorm_rope_kv_insert.py:341-351` 检查归一化、RoPE 和查询量化结果；同一测试中的 cache 插入断言没有计作该数值关系。

明确排除了仅验证缓存搬运、权重布局、通信/路由索引或元数据形状的候选。例如 `test_deep_gemm_mega_moe.py` 这批候选主要核对权重和 TMA 输入布局；`test_amx_mla.py::test_concat_and_cache_mla_round_trip` 核对缓存写回；`test_deepseek_v4_cpu_kernels.py::test_save_partial_states_cpu_matches_eager` 核对状态缓存。它们可能对应其他验证单元，但不属于本批的算子数值参考关系。逐项排除名单和本轮未进入审查的候选见证据 JSON。

单元模型可继续使用 `compute.operator_reference_relation`，但应把参考来源类型（数学公式、框架原生实现、另一后端或同算子的不同输入布局）、输出字段、数值容差、dtype、形状、设备/后端及状态条件作为显式属性；FP8 值与 scale、Attention 输出与 LSE、多返回值 Norm 等测试不能只用一个笼统的“数值一致”标签。待审的其余候选仍需按这些条件核查，不能由本批结果外推其性质。

交付文件为 `reviewed-reference-batch2.tsv` 和 `reviewed-reference-batch2-evidence.json`。证据 JSON 为每个源文件记录快照、路径、SHA256、确认的测试定义行、比较断言行和参考值构造片段；TSV 的 `oracle_source` 可直接定位检查位置。
