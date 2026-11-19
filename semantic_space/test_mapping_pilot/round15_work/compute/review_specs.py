import ast,csv,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
P=OUT.parents[1];ROOT=P.parents[3];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'
D='compute.kernel_dispatch_binding';N='compute.operator_reference_relation';E='compute.execution_relation'
S={};DEFERRED=[]
def add(path,impl,items):
 for name,unit,desc in items:S[path,name]=(unit,desc,impl)
B='compute.kernel_call_binding';K='compute.kernel_plan_consistency';R='compute.kernel_registration_binding'
W='compute.warmup_execution_policy'
PROPOSED=[]
p='tests/model_executor/test_triton_dispatcher.py';impl=['vllm/triton_utils/dispatcher.py:56','vllm/triton_utils/dispatcher.py:153','vllm/triton_utils/dispatcher.py:168','vllm/triton_utils/dispatcher.py:234','vllm/utils/cpu_triton_utils.py:12','vllm/model_executor/warmup/jit_warmup_triton_helper.py:327']
items=[
('override_patches_defining_module_and_warmup_owner',R,'真实registry处理伪模块/伪Triton，替换定义模块与warmup owner引用，调用CPU替身的参数和值精确匹配。'),
('override_forwards_positionally_for_mismatched_names',B,'原kernel与CPU替身参数名不一致时，按原参数顺序传1、2、0并保留可选值None。'),
('override_covers_from_import_copies',R,'两个伪模块持有同一原kernel时，注册替换同步更新from-import副本为KernelOverride。'),
('override_owner_launch_goes_through_impl',B,'伪kernel的真实warmup owner执行runtime launch，CPU替身收到1、2、0及CONST8。'),
('override_forwards_grid_to_kernel_like_impl',B,'提供__getitem__协议的替身直接收到原grid(8,)，并收到对应输入参数；无实际计算。'),
('override_plain_impl_never_sees_grid',B,'普通CPU callable不接收grid，只收到原kernel参数1/2/0及默认None。'),
('override_delegates_warmup_to_impl',B,'替身实现具有warmup时，grid(1,)与MAX_NUM_TOKENS8原样转发。'),
('override_warmup_noop_without_impl_warmup',B,'替身实现无warmup时返回None且不执行普通CPU实现。'),
('register_triton_kernel_swaps_raw_for_device_function',R,'符合Triton调用协议的伪替身直接按对象身份写入定义与import模块，不加wrapper。'),
('register_non_triton_impl_is_wrapped',R,'普通callable注册为KernelOverride，调用输出及记录参数正确；不执行真实Triton。'),
('register_kernel_auto_imports_defining_module',R,'移除真实rejection_sampler模块后注册会重新import并安装CPU替身，恢复原模块后重绑；检查调用参数，不检查采样算法。'),
('real_triton_kernel_override_end_to_end',R,'给真实JIT函数对象注册普通CPU替身后，只断言原arg_names三字段保留；没有执行Triton加法。'),
('cpu_slot_mapping_override_forwards_kernel_arguments',B,'真实CPU slot mapping适配器调用模拟torch op，只向op传四个指针标记和block16；不计算slot mapping。')]
add(p,impl,[('test_'+n,u,d) for n,u,d in items])
p='tests/model_executor/test_jit_warmup_triton_launcher.py';impl=['vllm/model_executor/warmup/jit_warmup_triton_helper.py:43','vllm/model_executor/warmup/jit_warmup_triton_helper.py:86','vllm/model_executor/warmup/jit_warmup_triton_helper.py:171','vllm/model_executor/warmup/jit_warmup_triton_helper.py:237','vllm/model_executor/warmup/jit_warmup_triton_helper.py:327','vllm/model_executor/warmup/jit_warmup_triton_helper.py:392','vllm/model_executor/warmup/jit_warmup_triton_helper.py:526','vllm/model_executor/warmup/jit_warmup_triton_helper.py:626','vllm/v1/worker/block_table.py:392']
items=[
('triton_launcher_supports_compile_and_runtime_adapters',B,'伪Triton kernel在compile时收到grid1和预热输入，runtime adapter收到grid2、两个位置输入和CONST7；无真实编译。'),
('failed_parallel_warmup_does_not_leak_into_runtime',W,'真实Triton async模式中注入编译失败，异常传播后active_mode为None，后续伪kernel runtime能执行一次。'),
('autotuning_warmup_stays_serial',W,'模拟autotune模式及4threads，将async入口设为fail哨兵；两个compile keys走runtime式串行调优调用。'),
('registered_warmup_overlaps_kernels_and_runtime_stays_serial',W,'真实Triton store内核两个常量在非caller线程编译且barrier证明并发；退出后runtime复用实例，首次常量3在caller线程同步编译，核对写入值。'),
('triton_launcher_supports_cpu_function_wrappers',B,'wrapped普通CPU函数通过真实launcher收到runtime、2、CONST7，无Triton编译。'),
('triton_kernel_decorator_returns_launcher',B,'伪key推导和伪内核下核对warmup/runtime的指针别名、stride5、默认second2及CONST7；拒绝未知关键字；预热keys覆盖1/2。'),
('triton_kernel_decorator_without_triton',B,'模拟无Triton，真实decorator仍将runtime输入与CONST7转交伪kernel。'),
('triton_kernel_decorator_exhausts_large_ranges_before_deduplication',K,'伪key推导按CONST7/8去重，必须先逐一调用dispatch覆盖1..8192，再得到两个key，不能提前截断。'),
('triton_kernel_decorator_propagates_dispatch_assertions',K,'伪kernel、伪key推导时，输入second2触发的dispatch断言必须传播到get_warmup_keys。'),
('triton_kernel_dispatch_uses_device_fake_tensors',K,'伪kernel/key推导时，依赖Tensor接口的dispatch用目标平台FakeTensor重试，保留设备、shape和连续性并生成一个key。'),
('triton_kernel_decorates_native_launchers',B,'伪native kernel保留原[]runtime参数调用；两个second choices产生预热keys，compile使用grid1及指定CONST7。'),
('triton_warmup_inputs_expands_explicit_pointer_dtypes',B,'读取真实空JIT函数的参数表，显式pointer dtype生成FP32占位，保留value2；非指针及缺失指针声明分别拒绝。'),
('warmup_pointer_specializes_like_the_live_tensor',K,'真实HIP backend specialization谓词对CPU live张量与metadata stand-in一致，覆盖默认小storage与自定义4GiB ptr_range；不实际编译。'),
('warmup_pointer_range_survives_the_fake_tensor_fallback',K,'伪kernel/key推导中FakeTensor重试保留4GiB指针范围，推导和warmup接收的metadata均不退回小storage默认。'),
('triton_key_derivation_applies_wrappers_and_runtime_options',K,'真实heuristics包装空JIT函数但binder/device为替身；BLOCK按value2得16，并传debug及instrumentation等runtime options。'),
('triton_key_derivation_covers_autotune_configs_and_jit_identity',K,'真实autotune包装空JIT函数、binder/device替身；BLOCK16/32配置得两个key，函数cache_key及对象身份被纳入，不同JIT函数key不相交。'),
('compute_slot_mapping_uses_named_launcher_inputs',B,'真实ComputeSlotMappingKernel仅把launch替换记录器，compile按world2/rank1/block16/stride128生成grid2及明确常量，不运行地址计算。')]
add(p,impl,[('test_'+n,u,d) for n,u,d in items])
PROPOSED.append({'id':W,'name':'计算预热并发策略与执行上下文隔离','kind':'obligation','phase':'model_execution','state_scope':'deployment','conditions':['kernel_selection_profile','execution_layout'],'property':'计算预热或调优按声明的并发、串行及分组策略执行；预热退出或失败后其编译上下文不残留到正式执行，已预热实例按计划复用，正式执行首次遇到的新实例按运行时策略编译。','observable':['compile_thread','active_compile_context','warmup_calls','runtime_calls','compile_count','failure'],'oracle':'记录编译/调优调用与线程身份，使用同步屏障或异常注入检查实际并发及退出；核对预热后缓存命中与运行时首次编译，并声明编译器和计算内核的替身范围。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':'vllm/model_executor/warmup/jit_warmup_triton_helper.py','line':237,'kind':'source'},{'path':'vllm/model_executor/warmup/jit_warmup_triton_helper.py','line':270,'kind':'source'},{'path':'tests/model_executor/test_jit_warmup_triton_launcher.py','line':113,'kind':'existing_test'}]}})
p='tests/model_executor/kernels/test_b12x_linear.py';impl=['vllm/model_executor/kernels/linear/scaled_mm/b12x.py:30','vllm/model_executor/kernels/linear/scaled_mm/b12x.py:101','vllm/model_executor/kernels/linear/scaled_mm/b12x.py:242','vllm/model_executor/kernels/linear/scaled_mm/b12x.py:302','vllm/model_executor/kernels/linear/mxfp8/b12x.py:49','vllm/model_executor/kernels/linear/mxfp8/b12x.py:70','vllm/model_executor/kernels/linear/mxfp4/b12x.py:22','vllm/model_executor/kernels/linear/nvfp4/b12x.py:20','vllm/model_executor/kernels/linear/nvfp4/b12x.py:144','vllm/utils/b12x.py:56','vllm/utils/b12x.py:132','vllm/model_executor/layers/quantization/modelopt_block_quant.py:319','vllm/model_executor/kernels/linear/__init__.py:651','vllm/model_executor/kernels/linear/__init__.py:1200','vllm/model_executor/layers/quantization/utils/fp8_utils.py:1157','vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:90']
items=[
('backend_registration_priority_and_selection',D,'五种量化族检查真实优先级列表，模拟CUDA与支持谓词为True后显式b12x选择对应类；无数值内核。'),
('module_lookup_is_dynamo_safe',E,'缓存中替换为伪b12x模块，其run仅x+1；torch.compile eager/fullgraph包装访问器后输出等于x+1，不证明b12x GEMM。'),
('tensor_fp8_can_implement_supported_config',D,'静态per-tensor FP8、BF16输入输出、64x128配置通过资格谓词。'),
('block_fp8_checks_runtime_support',D,'模拟SM120但外部runtime支持探针False，拒绝block-FP8并返回对应原因。'),
('block_fp8_requires_matching_supported_dtypes',D,'block-FP8拒绝FP32和BF16/FP16不匹配，接受匹配FP16。'),
('block_fp8_requires_aligned_features',D,'block-FP8输入K192或输出N192不按128对齐时，分别返回维度原因。'),
('tensor_fp8_process_weights_packs_modelopt_layout','model.parameter_representation_relation','外部pack为记录替身，真实预处理传入权重转置连续值及0.5*0.25输出scale；清空来源weight/scale仍保留loader及input_scale，登记packed对象/warmup provider。'),
('tensor_fp8_apply_quantizes_and_uses_packed_weight',B,'quant与GEMM均替身，核对量化张量指针、packed/bias身份、BF16和M6，输出恢复2x3x48；未验证FP8量化或矩阵乘法。'),
('mxfp8_can_implement_supported_config',D,'MXFP8 256x512资格接口返回允许；该实现当前无进一步shape约束。'),
('mxfp8_support_check_reports_missing_import',D,'模拟SM120且导入返回None，拒绝并给安装原因。'),
('mxfp8_support_respects_runtime_probe',D,'模拟SM120但外部支持探针False，MXFP8拒绝。'),
('mxfp8_process_weights_packs_modelopt_layout','model.parameter_storage_layout','pack为替身且来源未初始化，仅核对FP8权重48x128及uint8 scale裁剪48x4，保留loader且清空来源参数；不证明scale数值。'),
('mxfp8_reload_reuses_packed_tensor_addresses','model.parameter_binding','外部pack为clone替身；第二次加载weight全1/scale全3，packed对象与两个地址不变、内容更新，原来源参数清空。'),
('block_fp8_process_weights_keeps_native_block_layout','model.parameter_storage_layout','模拟FP8平台，真实预处理保留128x128 FP8权重和1x1 FP32 scale及loader；未初始化输入只断言布局。'),
('block_fp8_upcasts_e8m0_weight_scales','model.parameter_representation_relation','模拟FP8平台，E8M0或uint8指数125均准确上转FP32数值0.25。'),
('mxfp8_apply_uses_packed_weight',B,'外部mm为常数替身，输入展平6x128，packed/bias身份及expected_m6正确，输出恢复2x3x48同dtype。'),
('block_fp8_apply_uses_b12x_recipe_api',B,'外部mm返回常数13，wrapper按a/as/weight/ws顺序及BF16 out_dtype传参，输出6x256且保留常数；不是GEMM数值验证。'),
('mxfp4_requires_dynamic_activations',D,'仅dynamic MXFP4激活key通过，None拒绝并说明。'),
('fp4_processes_scale_and_preserves_loader','model.parameter_binding','MXFP4/NVFP4 scale-swizzle均为替身返回更大storage，参数绑定其地址、保留旧loader并登记warmup provider；不证明swizzle数学映射。'),
('mxfp4_apply_calls_native_blockscaled_gemm',B,'量化和mm均替身，wrapper将packed/scale/weight精确传给native API并加bias1到常数3，恢复2x3x48；不证明量化GEMM。'),
('nvfp4_can_implement_supported_config',D,'NVFP4资格接口对None返回允许；该接口未验证配置。'),
('backend_selects_w4a16',D,'模拟CUDA和候选表及W4A16支持，显式b12x加use_a16只选择W4A16实现。'),
('nvfp4_apply_calls_native_blockscaled_gemm',B,'量化/mm替身；非连续stride2输入展平后仍同指针非连续，global scale身份及swizzled标志正确；mm接收alpha，常数3+bias1恢复形状。'),
('nvfp4_w4a16_uses_weight_only_scale_and_a16_dispatch',B,'pack/mm/swizzle替身，activation量化为fail哨兵；weight-only global scale绑定，预热M1/8及runtime输入连续BF16、required_mode a16，bias/形状正确，FP32拒绝。'),
('nvfp4_w4a16_preserves_bf16_activations_under_graph_replay',N,'真实W4A16内核且禁activation量化；由测试显式FP4 LUT、block scale、global0.125构造权重，以FP64 matmul转BF16加bias为参考；编译图改动stride2输入后replay两轮。'),
('block_quant_dense_compiled_graph',N,'实际ModelOpt block-quant方法和b12x实现，对外部b12x.testing解码参考产生的BF16权重做FP32 matmul；三codec、profile分支和M1/4/17图replay检查误差及无新增分配。外部参考解码实现不在固定快照，未将其独立性另作保证。'),
('q8_embedding_graph',N,'实际Q8 embedding方法对外部b12x.testing解码表的索引值逐位比较，覆盖int32/int64和动态7/9/70行，改IDs后replay无额外分配。外部参考解码实现不在固定快照，未将其独立性另作保证。')]
add(p,impl,[('test_b12x_'+n,u,d) for n,u,d in items])
p='tests/kernels/quantization/test_zen_da8w4.py';impl=['vllm/model_executor/kernels/linear/mixed_precision/zentorch.py:55','vllm/model_executor/kernels/linear/mixed_precision/zentorch.py:124','vllm/model_executor/kernels/linear/mixed_precision/zentorch.py:281','vllm/model_executor/kernels/linear/mixed_precision/cpu.py:26','vllm/model_executor/layers/fused_moe/oracle/int_wna16.py:125','vllm/model_executor/layers/fused_moe/oracle/int_wna16.py:165','vllm/model_executor/layers/fused_moe/oracle/int_wna16.py:1007','vllm/model_executor/layers/fused_moe/experts/zentorch_moe.py:61','vllm/model_executor/layers/fused_moe/experts/cpu_moe.py:68','vllm/model_executor/kernels/linear/zentorch_utils.py:35','tests/kernels/quant_utils.py:23','tests/kernels/moe/test_cpu_fused_moe.py:418']
items=[
('da8w4_eligible_for_symmetric_bf16_layer',D,'模拟Zen平台，对真实CT packed对称INT4/BF16层资格处理成功；无Zentorch安装时repack为本地参考替身。'),
('da8w4_not_eligible',D,'模拟Zen，FP32激活或非对称INT4/zero-point使DA8W4处理返回False；未断言完整fallback输出。'),
('da8w4_not_eligible_when_env_disabled',D,'模拟Zen但环境开关关闭，DA8W4处理返回False。'),
('da8w4_process_weights_layout','model.parameter_storage_layout','真实adapter产生INT8 packed[N,K/2]及BF16 scale[G,N]，两者连续且与checkpoint参数storage别名；可选Zentorch repack替身，未检查数值。'),
('da8w4_apply_weights_matches_dequantized_reference','model.parameter_representation_relation','CT打包权重经生产重排，再通过DA8W4调用后对照原量化权重解码的matmul+bias；无Zentorch安装时op与期望共用_dynamic_quant_matmul，仅验证重排/解码接线，不证明真实Zentorch数值内核。安装时才调用真实op。'),
('da8w4_falls_back_to_w4a16_when_op_missing',D,'op可用性替身否定dynamic_qlinear，_maybe_process_da8w4_weights返回False；测试未运行W4A16 fallback。'),
('can_implement_requires_zen_cpu',D,'平台is_zen_cpu替身False导致资格拒绝及非空原因。'),
('can_implement_accepts_shapes_cpuwna16_rejects',D,'模拟CPU/Zen及ops可用，N112不满足generic oneDNN32对齐被拒绝，而Zen资格允许。'),
('can_implement_group_size_must_divide_k',D,'K512时group -1/128/512允许，6或0拒绝并报告必须整除。'),
('zen_cpu_first_in_cpu_backend_priority',D,'模拟CPU时优先级首为ZEN_CPU且保留generic CPU候选。'),
('zen_cpu_process_weights_layout','model.parameter_storage_layout','MoE CT权重转换后保持E,N,K/8 int32 packed，scale转E,G,N BF16并无zero-points；可选repack参考替身，仅shape/type检查。'),
('zen_cpu_repack_is_value_exact','model.parameter_representation_relation','MoE转换的W13逐专家由测试signed-nibble解码，对照外部CT unpack结果；无Zentorch安装时repack为独立本地参考，实际验证adapter打包关系。'),
('zen_cpu_int8_decode_keeps_full_width_values','model.parameter_representation_relation','W8转换产生E,2I,H全宽INT8，每个专家精确等于量化前保存的signed8量化值；不执行MoE。'),
('zen_cpu_num_bits_gating',D,'真实CT量化配置W8允许/W2拒绝，只调用兼容性谓词。'),
('zen_cpu_rejects_unknown_checkpoint_format',D,'duck-typed Args即使含group128/bits4，不能确定checkpoint格式则拒绝并含类名。'),
('zen_cpu_group_size_gating',D,'真实CT group32/128允许，channel(-1)或group2拒绝动态激活路径。'),
('zen_cpu_rejects_zero_points_but_takes_bias',D,'Zen兼容性谓词允许bias但拒绝任意zero-points组合。'),
('zen_cpu_rejects_moe_wna16_layout',D,'真实MoeWNA16Config不属于支持checkpoint布局，兼容性返回非空原因。'),
('zen_cpu_disabled_by_env',D,'VLLM_CPU_INT4_W4A8=False使Zen MoE兼容性拒绝。'),
('zen_cpu_backend_maps_to_experts_class',D,'ZEN_CPU后端准确映射唯一ZentorchExpertsWNA16类。'),
('zen_experts_support_predicates',D,'资格谓词按groupINT4/INT8、activation枚举、interleaved标记及无EP要求返回；未执行专家。'),
('zen_experts_score_every_routing_method_they_accept',D,'非Custom路由的支持集合与scoring映射键完全一致，MiniMax2明确sigmoid+renormalize；未计算路由分数。'),
('zen_experts_take_routing_config_off_the_layer','deployment.effective_configuration','用StubMoELayer与SimpleNamespace替身，真实postload将custom routing函数身份、renormalize及sigmoid配置复制到experts；函数不执行。'),
('swigluoai_perm_only_for_zen_swigluoai',D,'仅声明interleaved需求且activation SWIGLUOAI返回排列，SILU或无标志类返回None。'),
('swigluoai_perm_interleaves_weights_scales_and_bias','model.parameter_representation_relation','以生产排列重排W13及scale后，转换权重独立解码的偶/奇行分别等于原gate/up；bias同排列逐值检查。测试未对转换后的scale数值另断言。')]
add(p,impl,[('test_'+n,u,d) for n,u,d in items])
DEFERRED.append({'snapshot':'vllm','path':p,'test_name':'test_zen_cpu_conversion_logs_the_width','reason':'只断言加载日志位宽字符串，暂不归入数值转换或分派性质'})
p='tests/model_executor/test_weight_utils.py';impl=['vllm/model_executor/model_loader/weight_utils.py:1618','vllm/model_executor/model_loader/weight_utils.py:1634','vllm/model_executor/model_loader/weight_utils.py:1682','vllm/model_executor/model_loader/weight_utils.py:1772','vllm/model_executor/layers/quantization/base_config.py:232','vllm/model_executor/layers/quantization/base_config.py:267','vllm/model_executor/models/utils.py:58','vllm/model_executor/layers/layernorm.py:70','vllm/model_executor/layers/layernorm.py:162','vllm/model_executor/layers/layernorm.py:248','vllm/model_executor/layers/layernorm.py:339']
M='model.parameter_binding';Q='model.dummy_parameter_initialization'
items=[
('test_attention_sink_loader_shapes',M,'模拟TP1/2/4及各rank，sink加载本rank checkpoint连续heads，并把额外runtime heads填负无穷；逐值核对四种head尺寸。'),
('test_attention_sink_loader_rejects_invalid_shapes',M,'模拟TP2，checkpoint或runtime非1D、heads不整除、runtime太短分别拒绝，参数全1保持不变。'),
('test_attention_sink_loader_keeps_param_dtype',M,'CPU FP32 checkpoint写入BF16或FP32 sink，目标dtype保持且值等于arange。'),
('test_attention_sink_loader_accepts_cpu_checkpoint',M,'模拟TP2/rank1，实际CPU checkpoint写GPU参数，前四heads为4..7，其余两heads负无穷，保留目标设备。'),
('test_make_attention_sink_registers_loader_and_padding',M,'factory创建指定head数/dtype或默认dtype的无梯度参数，全负无穷且绑定load_attention_sink loader身份。'),
('TestMaybeRemapKvScaleName::test_non_scale_name_unchanged',M,'非scale的qkv_proj.weight名称原样返回；只验证名字映射，未加载张量。'),
('TestMaybeRemapKvScaleName::test_nvfp4_weight_scale_not_remapped',M,'NVFP4 weight_scale不是KV scale，完整名称原样返回；未加载张量。'),
('TestMaybeRemapKvScaleName::test_nvfp4_input_scale_not_remapped',M,'NVFP4 input_scale不是KV scale，完整名称原样返回；未加载张量。'),
('TestMaybeRemapKvScaleName::test_missing_target_returns_none',M,'KV scale重命名后的目标不在参数字典时返回None，不绑定不存在参数。'),
('test_checkpoint_weight_mapper_discards_serialized_g_idx',M,'默认checkpoint mapper丢弃.g_idx名称，保留.qweight名称；不读取权重数据。'),
('TestKvCacheScaleMapper::test_remap',M,'真实默认mapper对fused QKV/ModelOpt/Nemotron/bare/旧KV名称逐项映射到attn scale，保留已规范化名称与非KV权重/scale。'),
('TestKvCacheScaleMapper::test_idempotent',M,'四种checkpoint scale名首次映射非空，第二次映射保持不变；只验证命名幂等。'),
('TestKvCacheScaleMapper::test_composes_with_qkv_mapper',M,'合并model substring mapper与cache regex mapper后，普通q_proj融合为qkv_proj.q，KV scale先规范化到attn，避免误融合。'),
('test_dummy_fp8_weight_is_not_degenerate',Q,'E4M3/E5M2 placeholder权重初始化后全finite、绝对值<=1，4096元素超过99%非零。'),
('test_dummy_e8m0_scale_is_fixed_exponent',Q,'E8M0类型或名称包含scale的uint8按指数byte120初始化，逐字节相同。'),
('test_dummy_packed_weight_gets_random_bytes',Q,'uint8/int8/packedFP4两次初始化结果逐字节相同且4096字节包含超过250不同值。'),
('test_dummy_wide_integer_param_is_untouched',Q,'非ROCm平台INT32 g_idx已含arange时placeholder初始化保留其值，避免破坏索引。'),
('test_dummy_norm_weights_are_neutral',Q,'真实四种norm创建的dummy_weight_value元数据驱动初始化：RMS/LayerNorm/Gated权重1，Gemma权重0与LayerNorm bias0；无需执行norm。')]
add(p,impl,items)
for name,reason in [('test_download_weights_from_hf','下载及离线cache可用性独立于模型计算，本批未核查下载实现'),('test_drop_checkpoint_cache_releases_local_weight_files','只检查mock posix_fadvise调用数与错误吞并，本批不归为参数正确性')]:DEFERRED.append({'snapshot':'vllm','path':p,'test_name':name,'reason':reason})
PROPOSED.append({'id':Q,'name':'占位参数初始化遵守格式与中性值','kind':'obligation','phase':'model_execution','state_scope':'deployment','conditions':['parameter_lifecycle','numeric_format','parameter_mapping'],'property':'用于无真实检查点启动或性能探针的占位参数按类型、名称与显式中性值初始化：浮点取值满足声明的有限范围，编码缩放量和打包字节保持可用且确定，标记的归一化参数取中性值，声明保留的宽整数索引不改写。','observable':['initialized_parameter','encoded_bytes','finite_fraction','nonzero_fraction','deterministic_values','preserved_indices'],'oracle':'按初始化策略核对取值域、缩放指数字节、中性常数及重复结果，并比较保留项初始化前后的值；占位值不构成真实模型精度证据。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':'vllm/model_executor/model_loader/weight_utils.py','line':1682,'kind':'source'},{'path':'vllm/model_executor/layers/layernorm.py','line':70,'kind':'source'},{'path':'tests/model_executor/test_weight_utils.py','line':414,'kind':'existing_test'}]}})
p='tests/model_executor/model_loader/test_sharded_state_loader.py';impl=['vllm/model_executor/model_loader/sharded_state_loader.py:60','vllm/model_executor/model_loader/sharded_state_loader.py:70','vllm/model_executor/model_loader/sharded_state_loader.py:139','vllm/model_executor/model_loader/sharded_state_loader.py:193','vllm/model_executor/model_loader/sharded_state_loader.py:276','vllm/model_executor/model_loader/__init__.py:64','vllm/v1/worker/gpu_worker.py:1672','vllm/v1/executor/abstract.py:354','vllm/v1/engine/core.py:1202']
A='model.checkpoint_shard_artifact_binding'
items=[
('filter_subtensors',A,'state字典含相同tensor别名和连续/stride切片时，序列化集合仅保留覆盖它们的a/b/c且对象身份不变；不读取未初始化数值。'),
('load_weights_resolves_hf_repo_id',M,'模拟PP/TP和HF下载目录，真实safetensors临时文件加载到Linear权重；model_weights优先，下载替身收到revision/cache/ignore原参数。'),
('get_default_pattern',A,'模拟PP world1/2分别选择TP-only或PP+TP模板。'),
('render_default_pattern_default_tp',A,'模拟PP1默认生成model-rank-0-part-0文件名。'),
('render_default_pattern_default_pp',A,'模拟PP2默认生成含pp0/tp0/part0的文件名。'),
('render_default_pattern_custom_tp_only',A,'PP1显式legacy TP模板按rank0/part0渲染。'),
('render_default_pattern_tp_only_with_pp_fails',A,'PP2模板缺PP rank会跨stage碰撞，渲染前拒绝。'),
('render_default_pattern_pp_pattern_without_pp_fails',A,'PP1却提供pp_rank模板时拒绝requires pipeline parallelism。'),
('render_default_pattern_custom_pp',A,'PP2显式PP+TP模板生成当前pp0/tp0/part0名。'),
('render_default_pattern_incomplete_custom_patterns_fail',A,'PP2/TP2任一活动rank维未出现在模板中分别拒绝并指出维度。'),
('render_default_pattern_no_rank_field_with_tp_fails',A,'PP1/TP2模板只有part没有rank时拒绝TP碰撞。'),
('render_default_pattern_custom_pp_tp1_allows_pp_only',A,'PP2/TP1允许只含PP rank的模板，当前pp1生成对应文件名。'),
('render_default_pattern_escaped_field_not_rank_field',A,'PP2/TP2中双大括号逃逸字段不算真实rank占位，拒绝缺少PP/TP维。'),
('render_default_pattern_tp_aliases_accepted',A,'rank和tp_rank两别名分别在PP+TP与TP-only模板中绑定模拟当前TP rank。'),
('render_default_pattern_unsupported_field_fails',A,'模板含未知epoch占位时以ValueError明确拒绝。'),
('render_default_pattern_non_zero_ranks',A,'模拟不同PP/TP rank值，模板使用实际给定rank而非零；测试不校验rank是否小于模拟world size。'),
('render_default_pattern_part_wildcard',A,'part传星号用于glob时原样保留，TP rank4正确写入名称。'),
('save_model_pp_aware_filenames',A,'save_file为记录替身，模拟PP2/pp1/tp2保存仅向PP+TP文件名写一次；不核对文件内容。'),
('save_model_part_split',A,'save_file为记录替身，两个16byte参数max_size20分为part0/1两次保存；只检查名称序列。'),
('save_model_rejects_bad_pattern_before_overwrite',A,'真实目录预存shard，PP模板rank为转义字面值时拒绝，原bytes及目录列表均保持。'),
('sharded_state_loader','model.execution_variant_relation','真实Llama3.2-1B原checkpoint与保存后sharded_state在独立进程、同TP1/2、max_num_seqs1、greedy256/ignore_eos下，4提示completion对象字典精确相同；enable_lora仅启用能力未挂适配器。'),
('sharded_state_loader_pp','model.execution_variant_relation','真实Llama3.2-1B在TP1/PP2保存重载，子进程均成功且生成PP-aware文件名；同4提示greedy256 completion字典精确相等。'),
('sharded_state_loader_pp_tp','model.execution_variant_relation','真实Llama3.2-1B在TP2/PP2保存重载，子进程均成功且生成PP-aware文件名；同4提示greedy256 completion字典精确相等。')]
add(p,impl,[('test_'+n,u,d) for n,u,d in items])
PROPOSED.append({'id':A,'name':'分片检查点文件与执行rank对应','kind':'obligation','phase':'model_execution','state_scope':'deployment','conditions':['parallel_topology','parameter_mapping','parameter_lifecycle'],'property':'分片检查点的文件集合按声明的并行rank和part编号命名及分卷，模板须区分所有活动rank维；别名和子张量按共享存储关系去重，不安全或未知模板在写入前拒绝并保留既有文件。','observable':['shard_filename','part_sequence','retained_tensor_keys','tensor_identity','existing_file_bytes','rejection'],'oracle':'给定rank拓扑和模板逐项比较名称及分卷序列，用共享/切片张量核对序列化集合，并在无效模板条件下检查写入前后文件状态；记录保存函数是否替身，名称检查不替代权重内容校验。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':'vllm/model_executor/model_loader/sharded_state_loader.py','line':70,'kind':'source'},{'path':'vllm/model_executor/model_loader/sharded_state_loader.py','line':276,'kind':'source'},{'path':'tests/model_executor/model_loader/test_sharded_state_loader.py','line':375,'kind':'existing_test'}]}})
