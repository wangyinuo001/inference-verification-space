import ast,csv,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=OUT.parents[1];ROOT=P.parents[3];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'
S={};DEFERRED=[];PROPOSED=[]
D='compute.kernel_dispatch_binding';B='compute.kernel_call_binding';N='compute.operator_reference_relation';E='compute.execution_relation';O='compute.output_constraints';L='model.parameter_storage_layout';M='model.parameter_binding';R='model.parameter_representation_relation';C='deployment.effective_configuration';I='deployment.configuration_admissibility';K='compute.kernel_plan_consistency'
def add(path,impl,items):
 for name,unit,desc in items:S[path,name]=(unit,desc,impl)
p='tests/quantization/test_turboquant.py';cfg='vllm/model_executor/layers/quantization/turboquant/config.py'
items=[
('preset_parses',C,'四个显式TurboQuant preset在head128解析为配置对象；仅检查类型。'),
('invalid_preset_raises',I,'不存在的turboquant_invalid在head128明确ValueError。'),
('key_mode',C,'head128下四preset的key_fp8/key_quant_bits/key_mse_bits与手写PRESET_EXPECTED逐字段一致。'),
('value_mode',C,'head128下四preset的value_quant_bits与手写表一致。'),
('bits_and_centroids',C,'head128下四preset的mse_bits、n_centroids、centroid_bits与显式表一致。'),
('norm_correction',C,'四preset的norm_correction与显式布尔表一致。'),
('packed_sizes','cache.capacity_plan','head128下key/value/slot及对齐slot字节数与手写表一致。'),
('slot_equals_key_plus_value','cache.capacity_plan','四preset单槽总字节数等于key及value字节数之和。'),
('padded_slot_is_even','cache.capacity_plan','四preset对齐slot不小于实际slot且为偶数。'),
('key_value_packed_sizes_positive','cache.capacity_plan','head128下key/value packed bytes都严格正。'),
('n_centroids_is_2_to_mse_bits',C,'四preset质心数量等于2的mse_bits次方。'),
('centroid_bits_always_positive',C,'四preset用于生成质心的位宽严格正。'),
('mse_key_or_fp8_exclusive',C,'FP8 key要求key_mse_bits0/key_quant_bits8，MSE key要求正mse位宽且key位宽3或4。'),
('all_presets_all_head_dims','cache.capacity_plan','head64/96/128/256逐preset保留head_dim且slot是key+value、向上偶数对齐。'),
('boundary_skip_layers_basic',C,'伪dense32层配置默认边界保护返回字符串层0/1/30/31。'),
('boundary_skip_layers_zero',C,'dense32层显式n0不跳过任何层。'),
('boundary_skip_layers_small_model',C,'dense4层默认边界保护覆盖四层且去重。'),
('boundary_skip_layers_cap_at_half',C,'dense8层请求保护n10时结果长度为8；未检查具体集合。')]
add(p,[cfg+':46',cfg+':178',cfg+':216'],[('TestTurboQuantConfig::test_'+n,u,d) for n,u,d in items])
add(p,[cfg+':236'],[('TestHybridAttentionIndices::test_'+n,C,d) for n,d in [
('layer_types_full_attention','伪hybrid layer_types包含linear/full_attention时返回全局索引2/4/5。'),('layers_block_type_jamba','伪layers_block_type mamba/attention交替时返回全局索引1/3。'),('attn_type_list_minimax','伪HF attn_type_list数值0/1时返回1/3/4。'),('no_hybrid_hints_returns_empty','无hybrid识别字段的伪config返回空列表。')]])
add(p,['vllm/model_executor/layers/attention/attention.py:637','vllm/v1/attention/backends/turboquant_attn.py:141'],[('TestTurboQuantKVCacheSpec::test_kv_cache_spec_sets_kv_quant_mode','cache.capacity_plan','伪Attention decoder/head128/KV4/block32对每preset生成未打包FullAttentionSpec；随后真实backend customize_spec设置state_content_bytes为配置aligned slot；只查规格不分配缓存。')])
add(p,['vllm/v1/attention/backends/turboquant_attn.py:222','vllm/v1/attention/backends/turboquant_attn.py:227'],[('TestTurboQuantWorkspaceReservation::test_'+n,K,d) for n,d in [
('metadata_builder_reserves_decode_and_continuation_prefill_workspace','用记录型FakeWorkspaceManager及伪config构造真实builder，检查decode三缓冲尺寸16x8x4x129、16x8x128、16x8和FP32/FP16类型；chunked prefill另请求两1x4x8192x128 FP16缓冲。不实际分配。'),('metadata_builder_skips_continuation_prefill_when_disabled','同伪allocator/config但chunked prefill=False，仅申请decode三缓冲，不申请continuation prefill。')]])
cen='vllm/model_executor/layers/quantization/turboquant/centroids.py'
add(p,[cen+':31',cen+':84'],[('TestCentroids::test_'+n,u,d) for n,u,d in [
('centroids_shape',O,'真实get_centroids(128,bits2/3/4)输出长度4/8/16。'),('centroids_sorted',O,'真实get_centroids(128,bits2/3/4)各相邻元素严格升序。'),('centroids_different_dims_not_identical',E,'相同bits3但dim64与128求解的质心张量不逐元素相同。'),('centroids_symmetric_around_zero',O,'bits2/3/4、dim128求解质心均值绝对值及首尾和绝对值小于0.01；近似对称约束。'),('centroids_within_4sigma',O,'dim128、bits2/3/4全部质心绝对值小于4*sqrt(1/128)。')]])
add(p,[cen+':22',cen+':31'],[('TestLloydMax::test_'+n,u,d) for n,u,d in [
('solve_shapes',O,'dim128/bits2/3/4求解输出质心长度2^bits及边界长度2^bits-1。'),('centroids_sorted',O,'真实solve_lloyd_max输出质心严格升序。'),('boundaries_sorted',O,'真实solve_lloyd_max输出边界严格升序。'),('boundaries_between_centroids',O,'每个返回边界严格位于对应相邻质心之间。'),('boundaries_are_midpoints',N,'返回边界与返回的相邻质心算术中点相差小于1e-6；内部解析关系而非外部质量基准。'),('solve_deterministic',E,'相同dim128/bits3重复求解，质心和边界逐元素完全相同。'),('solve_dtype_float32',O,'dim128/bits3求解质心与边界均FP32。'),('centroids_match_scipy_reference',N,'真实梯形积分Lloyd-Max(dim128,bits3/4)质心与测试独立Scipy quad数值积分200次、tol1e-10高斯求解参考比较，max_abs_error<1e-3；需要Scipy。')]])
DEFERRED += [{'path':p,'tests':['TestCentroids::test_centroids_cached'],'reason':'对象缓存身份性质不冒充数值或KV缓存性质'}, {'path':p,'tests':['TestRotationMatrix::*','TestHadamardRotation::*'],'reason':'仅检查测试内QR/Hadamard辅助实现，未调用生产实现'}, {'path':p,'tests':['TestStoreDecodeRoundTrip::test_single_token_roundtrip'],'reason':'尚未完成store/decode生产kernel审查，保留待审'}]
p='tests/quantization/test_compressed_tensors.py';ct='vllm/model_executor/layers/quantization/compressed_tensors/';config=ct+'compressed_tensors.py';schemes=ct+'schemes/'
add(p,[config+':305'],[
('test_compressed_tensors_rejects_runtime_activation_order',I,'显式4bit group128 pack-quantized配置的group/dynamic/True激活排序被拒绝，错误匹配group/dynamic activation ordering。'),
('test_compressed_tensors_accepts_preordered_weights',C,'static/weight/False/None输入规范化后在Linear权重actorder字段保留枚举值或None。')])
add(p,[config+':159',config+':783',config+':998',ct+'utils.py:73'],[
('test_get_quant_method_returns_linear_method_for_parallel_lm_head',D,'Mock(spec=ParallelLMHead)和re:.*lm_head目标，INT8 channel weight-only返回CompressedTensorsLinearMethod。'),
('test_get_quant_method_returns_none_for_ignored_parallel_lm_head',D,'同目标但ignore匹配lm_head时Mock头返回None。'),
('test_get_quant_method_returns_none_for_unmatched_parallel_lm_head',D,'Mock ParallelLMHead不匹配Linear目标时返回None，不错误强制量化。'),
('test_find_matched_target_returns_none_on_no_match',D,'Mock Linear实例及qkv_proj路径均不匹配no_match_target时返回None。'),
('test_get_scheme_dict_returns_none_on_no_match',D,'INT8 channel配置目标matched_layer与unmatched_layer不同，schema查询返回None。'),
('test_scheme_selection',D,'显式13组weight_bits/strategy/input/output/format逐项对照expected_scheme：W8A8走Int8，input+output静态INT8走WNA8O8，2至8bit weight-only pack走WNA16。只检查类选择，不检查GEMM结果。')])
add(p,[config+':783',schemes+'compressed_tensors_wNa16.py:63','tests/quantization/utils.py:74'],[
('test_compressed_tensors_wNa16',D,'真实tinyllama W4A16 channel checkpoint检查qkv方法/策略、group=-1、pack_factor8、symmetric；另仅要求四token贪心返回非空，未比较生成质量。')])
add(p,[config+':783',schemes+'compressed_tensors_w4a4_nvfp4.py:29','vllm/model_executor/kernels/linear/__init__.py:1200','tests/quantization/utils.py:74'],[
('test_compressed_tensors_nvfp4',D,'两个真实TinyLlama NVFP4/A16 checkpoint检查scheme、use_a16、group16；ROCm NVFP4A16预期强制Marlin不支持错误。')])
add(p,[config+':783',schemes+'compressed_tensors_w4a4_mxfp4.py:39'],[
('test_compressed_tensors_mxfp4',D,'CUDA80+真实TinyLlama MXFP4检查四投影CompressedTensorsLinearMethod/W4A4Mxfp4和group32；生成仅非空，不断言质量。')])
add(p,[config+':783',ct+'compressed_tensors_moe/compressed_tensors_moe.py:28'],[
('test_compressed_tensors_moe_ignore_with_model',D,'真实tiny Qwen3MoE checkpoint检查第0层量化MoE与第3层UnquantizedFusedMoEMethod，均由MoERunner包装；四token生成仅非空。'),
('test_compressed_tensors_mxfp8_moe_setup',D,'CUDA75+ Qwen3-30B-MXFP8使用dummy加载并限制4层，检查dense MXFP8 scheme与专家MXFP8 MoE方法；非空生成不证明真实checkpoint质量。')])
add(p,[ct+'compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py:332','vllm/model_executor/layers/fused_moe/oracle/fp8.py:725'],[
('test_compressed_tensors_w8a8_fp8_moe_forwards_swiglu_params',B,'object.__new__构造量化方法，Mock layer给alpha1.702/betaNone/limit7，真实get_fused_moe_quant_config把三字段传入TRITON配置；不执行MoE。')])
add(p,['vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py:89','vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py:307',schemes+'compressed_tensors_wNa16.py:40'],[
('test_humming_supports_compressed_tensors_wna16_quant_key',D,'2至8bit ScalarType、FP16静态group128对称weight key及None activation，真实Humming支持谓词均为真；不调用后端。')])
add(p,[config+':459',config+':783',schemes+'compressed_tensors_w4a8_mxfp4.py:23',schemes+'compressed_tensors_w4a4_mxfp4.py:39'],[
('test_mxfp4_linear_with_fp8_inputs_requests_fp8_activations',D,'CUDA下MXFP4 group32权重+动态FP8 group128输入选择W4A8Mxfp4并检查kernel.config.activation_quant_key为kFp8Dynamic128Sym；不做数值前向。')])
DEFERRED += [{'path':p,'reason':'其余未完成底层存储/数值/runner链路的条目不纳入本批；只非空输出或字符串格式不冒充数值性质'}]
p='tests/quantization/test_quark.py';q='vllm/model_executor/layers/quantization/quark/';qc=q+'quark.py';qm=q+'quark_moe.py';qu=q+'utils.py';fp='vllm/model_executor/layers/quantization/utils/fp8_utils.py'
add(p,[qc+':105'],[
('test_quark_config_has_no_model_specific_fused_mappings',C,'空QuarkConfig不预置gate_up_proj和fused_wqa_wkv模型专属融合名。'),
('test_quark_config_preserves_existing_packed_modules_mapping',C,'CustomQuarkConfig预置custom_proj:a/b映射在实例初始化后保留。')])
add(p,[qc+':237',qc+':293',qc+':490',qc+':565',qc+':727',qc+':798',qc+':884',qm+':173','vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py:50',qu+':31','tests/conftest.py:254'],[
('test_quant_method_dispatch_ignored',D,'无量化配置或MXFP4配置中显式exclude linear/experts/q_proj/down_proj，实际简化层实例和类型target查询均选unquantized；专家配置8个expert/top2/hidden256。'),
('test_quant_method_dispatch_target',D,'32个QTENSOR_CONFIGS显式条目逐项查询真实Quark target：对FP8、INT8、OCP MX、NVFP4、W4A8核对weight/activation key及方法类；指定的静态block输入、group错配、缺block_size和二维MXFP8 experts按表中异常类型/消息拒绝。无后端计算。'),
('test_quant_method_dispatch_mixed_mxfp4_mxfp8_experts',D,'全局MXFP4及仅model.layers.1.block_sparse_moe.experts*覆盖MXFP8，层0/1/10/11方法集合分别OCP/MXFP8/OCP/OCP；shared_experts.down_proj不误命中。'),
('test_quant_method_dispatch_unsupported',D,'单entry不支持INT8 per_group及multi-entry INT8均在target查询和简化专家实例构造时报Unsupported FusedMoe scheme。'),
('test_glm5next_genuine_fusion_resolves_gate_up_to_block_fp8',D,'给gate_up_proj真实gate_proj/up_proj融合映射，匹配两者FP8 block覆盖而非全局MXFP4，返回QuarkW8A8Fp8PerBlock。')])
add(p,[qc+':237',qc+':727',qc+':970',q+'schemes/quark_ocp_mx.py:55',q+'schemes/quark_w8a8_fp8.py:223','vllm/model_executor/layers/quantization/utils/ocp_mx_utils.py:50','tests/conftest.py:254'],[
('test_quant_method_dispatch_mxfp8_2d_block',D,'简化Linear实例上2D [32,32] E8M0配置选择OCP_MX、scale_block_rows32，experts不误选Linear；[128,128]变体选择FP8PerBlock。'),
('test_quant_method_dispatch_mxfp8_canonical',D,'简化Linear实例canonical per_group32/e8m0 MXFP8选择OCP_MX与kMxfp8Static、scale_block_rows1。')])
add(p,[fp+':53',q+'schemes/quark_w8a8_fp8.py:223',q+'schemes/quark_w8a8_fp8.py:247'],[
('test_quark_w8a8_fp8_per_block_registers_weight_scale',L,'FP8 block缩放查询优先使用Quark scheme的weight_scale，否则回落weight_scale_inv；create_weights分配/校验/kernel均替身时只注册weight_scale名称，不注册weight_scale_inv，不证明分配数值。')])
add(p,[qm+':2024',qm+':2039'],[
('test_quark_w8a8_mxfp8_moe_weight_shapes',L,'模拟MXFP8后端选择后，9experts/hidden256/intermediate64分配FP8权重(9,128,256)/(9,256,64)和uint8尺度(9,128,8)/(9,256,2)，BLOCK标签/无input_scale；intermediate48拒绝。')])
add(p,[qm+':638',qm+':700',fp+':1347'],[
('test_quark_w8a8_fp8_moe_per_block_requires_dynamic_group_input',D,'模拟后端选择和config，block128权重配静态tensor输入在MoE构造被ValueError拒绝。'),
('test_quark_w8a8_fp8_moe_per_block_weight_shapes',L,'模拟后端/config/TP1，4experts、hidden512、intermediate256创建真实权重与128x128分组尺度，核对shape/BLOCK标签且无scale_inv命名。'),
('test_quark_w8a8_fp8_moe_per_block_rejects_misaligned_partition','compute.input_admissibility','同模拟环境下intermediate192无法整除128，真实create_weights明确ValueError。'),
('test_quark_fp8_w8a8_per_block_moe',L,'真实tiny-Qwen checkpoint/TP1检查FP8 block方法和尺度shape w13(4,4,1)/w2(4,1,2)、无scale_inv；生成四token仅非空，不推导数值精度。')])
add(p,[qu+':31','vllm/model_executor/layers/quantization/utils/config_utils.py:435'],[
('test_quark_should_ignore_layer_checks_children',D,'check_children=True时experts子项0.down_proj被忽略使对应experts父层忽略。'),
('test_quark_should_ignore_layer_rejects_partial_fused_matches',D,'qkv融合只忽略q_proj时拒绝部分量化配置。')])
add(p,['vllm/model_executor/layers/quantization/utils/quant_utils.py:630'],[('test_'+n,D,d) for n,d in [
('fused_name_listed_directly_is_skipped','qkv_proj/gate_up_proj融合名直接列入忽略表时返回True。'),
('unfused_shards_listed_is_skipped','q/k/v三个展开名均在忽略表时qkv融合整体被忽略。'),
('partial_shards_raises','qkv展开仅q_proj在忽略表时ValueError。'),
('not_skipped_when_nothing_listed','忽略gate_up_proj不影响不同的qkv_proj。'),
('non_fused_layer_unaffected','非融合o_proj只在同层同名命中时忽略，不命中层1同名。'),
('substr_match_on_fused_name','substring模式self_attn.qkv_proj匹配完整融合路径。'),
('suffix_match_at_module_boundary','五组suffix案例要求模块边界：b_proj不匹配q_b_proj/kv_b_proj，带层5后缀不匹配层6。')]])
add(p,['vllm/models/glm5next/common/model.py:1288'],[
('test_glm5next_gate_up_proj_mapping_is_genuine_fusion',C,'直接检查生产GLM5Next类gate_up_proj融合映射为gate_proj/up_proj。')])
add(p,['vllm/models/glm5next/common/model.py:1511','vllm/models/glm5next/common/model.py:1548'],[
('test_glm5next_attn_loader_accepts_quark_weight_scale',M,'记录型参数替身先接收kv_a权重再接收Quark weight_scale；第二项到齐后BF16反量化送入fused_qkv_a shard1且加入loaded，数值参考调用相同_dequant_fp8_block所以只证明缓冲/路由而非独立反量化正确性。'),
('test_glm5next_attn_loader_accepts_deepseek_weight_scale_inv',M,'记录型o_proj参数替身接受weight_scale_inv后送直接shardNone；值对照同一生产_dequant_fp8_block，只证明命名/加载路由。')])
add(p,[qc+':343',qc+':629',qc+':970',qu+':69',q+'schemes/quark_w4a16_int4.py:50'],[('TestQuarkInt4Format::test_'+n,u,d) for n,u,d in [
('quark_order_pack_method_uses_native_int4_scheme',D,'order打包/对称group128配置创建原生W4A16Int4且pack_reorder=False。'),
('quark_int4_scheme_supports_asymmetric_weights',D,'非对称group128配置仍选原生W4A16Int4且is_symmetric=False。'),
('quark_int4_scheme_requires_weight_config_fields',I,'逐次删除group_size或symmetric，真实init_scheme按缺失字段报ValueError。')]])
add(p,[qm+':330'],[('TestQuarkInt4Format::test_quark_int4_moe_uses_native_moe_method',D,'模拟select_wna16后端，非对称group128+reorder构造原生INT4 MoE检查group128/pack_reorder/use_wna16_backend；不执行MoE。')])
add(p,['vllm/model_executor/layers/fused_moe/experts/triton_moe.py:783'],[('TestQuarkInt4Format::test_triton_wna16_experts_supports_asymmetric_int4',D,'四个对称/非对称INT4 group32及默认key搭配None activation均被真实TritonWNA16支持谓词接受。')])
add(p,[qm+':330',qm+':549'],[('TestQuarkInt4Format::test_quark_moe_loader_shards_w2_zero_point_across_tp_ranks',M,'模拟后端和layer，随机uint8(8,16) packed零点分别经真实loader以TP1/TP2 rank0/1加载；两rank等于TP1结果按列重排后的两份，检查分片关系而非独立零点解码正确性。')])
add(p,[qc+':1026','vllm/model_executor/models/utils.py:58','vllm/model_executor/layers/quantization/base_config.py:232'],[('TestQuarkInt4Format::test_quark_shared_expert_gate_keeps_quantized_tensors',M,'真实cache scale mapper不丢弃shared_expert_gate的weight/weight_scale/weight_zero_point三名称。')])
add(p,[qc+':144','vllm/model_executor/models/utils.py:58',qu+':31'],[('TestQuarkInt4Format::test_'+n,C,d) for n,d in [
('quark_apply_mapper_updates_exclude_and_layer_quant_config','真实前缀mapper改写layer_quant_config键及exclude的lm_head名称，旧键删除。'),
('quark_apply_mapper_ignores_non_string_list_entries','algo_config包含dict的列表保留原嵌套内容，不当作字符串名称表重写。'),
('quark_exclude_matches_via_mapper_prefix','lm_head经前缀mapper改为language_model.lm_head后命中ignore，不忽略mlp.gate。')]])
add(p,[qu+':93'],[('test_quark_int4_canonicalizes_pack_for_kernel_layout',R,'order/reorder×对称/非对称四组合，固定包含正负nibble的2x8值及零点经真实canonicalize；测试手写signed/unsigned unpack dequant在group2/unit scales上证明前后反量化值完全相同。')])
DEFERRED += [{'path':p,'reason':'其余instantiation后端构造、实际模型有限性、外部Quark kernel与lm_eval质量入口未完整补齐对应实现，未计为本批确认'}]
p='tests/quantization/test_modelopt.py';mo='vllm/model_executor/layers/quantization/modelopt.py'
add(p,[mo+':235',mo+':737',mo+':2687',mo+':2828',mo+':2895'],[('test_'+n,D,d) for n,d in [
('modelopt_nvfp4_quantizes_parallel_lm_head','Mock ParallelLMHead在NVFP4配置下返回ModelOptLinearMethod，spec为Nvfp4Static/Dynamic；MoE activation override None或per_token不改变dense key。'),
('modelopt_nvfp4_leaves_excluded_parallel_lm_head_unquantized','显式exclude lm_head使Mock头选择UnquantizedLinearMethod。'),
('modelopt_nvfp4_reads_exclude_modules_from_config_json','从flat quant_algo NVFP4配置读取exclude_modules后Mock lm_head选择UnquantizedLinearMethod。'),
('modelopt_nvfp4_config_dispatches_w4a4_method','真实NVFP4 config对Mock Linear返回Nvfp4Static/Dynamic spec的方法。'),
('modelopt_nvfp4_config_dispatches_w4a16_method','真实W4A16_NVFP4 config对Mock Linear返回Nvfp4Static/None activation的方法。')]])
add(p,[mo+':1291',mo+':1362'],[
('test_modelopt_mxfp8_routes_nvfp4_experts_to_compressed_tensors',D,'M3.1混合配置对Mock RoutedExperts转交mock compressed-tensors NVFP4方法，精确传moe_config/prefix及QAT per-token key或默认dynamic key；纯mxfp8无moe_quant_algo。'),
('test_modelopt_mxfp8_rejects_unknown_expert_format',I,'M3.1配置覆盖moe_quant_algo=MXFP4或format=float-quantized均被ValueError拒绝。')])
add(p,[mo+':2712',mo+':2764'],[
('test_modelopt_mxfp8_pre_processed_weights_follow_kernel',D,'手工构造MXFP8方法kernel为None或mock能力False时不支持preprocessed；True时跳过postload调用，恢复False时RuntimeError。')])
add(p,[mo+':2292',mo+':2764'],[
('test_modelopt_fp8_updates_weight_dims_after_transpose',R,'真实FP8 per-tensor后处理将3x2权重转成2x3并设processed_weight_dims(0,1)，mock kernel后处理恰调用一次；只验证shape/layout和转交，不证明未初始化权重值。')])
add(p,[mo+':170',mo+':2828'],[
('test_modelopt_linear_algos_table_matches_resolve',D,'枚举六个LINEAR_ALGOS逐一resolve得到非空weight key；QUANT_ALGOS附加MIXED_PRECISION，owner分组与显式FP8/NVFP4/MXFP8集合一致。')])
add(p,[mo+':1677',mo+':1781',mo+':1862',mo+':1958',mo+':2828',mo+':2895'],[('test_'+n,u,d) for n,u,d in [
('modelopt_mixed_precision_dispatches_every_linear_algo',D,'六个LINEAR_ALGOS逐项写入单down_proj混合配置，对Mock Linear检查ModelOptLinearMethod类型；未比较数值。'),
('modelopt_mixed_precision_dispatches_block_fp8_moe',D,'FP8_PB_WO/FP8_BLOCK_SCALES专家recipe用mock Fp8MoEMethod记录，精确转入原layer及block[128,128]/dynamic config。'),
('modelopt_mixed_precision_quantizes_parallel_lm_head',D,'lm_head NVFP4 recipe使Mock头返回Nvfp4Static/Dynamic方法。'),
('modelopt_mixed_precision_resolves_declared_packed_projection',D,'显式qkv融合映射把q/k/v三个同MXFP8 recipe解析为融合MXFP8。'),
('modelopt_mixed_precision_does_not_quantize_unlisted_fused_sibling',D,'in_proj_qkv/z显式同FP8合并为qkvz，但未列出的b/a融合ba返回None。'),
('modelopt_mixed_precision_infers_fused_gate_up_projection',D,'仅列出gate/up两个NVFP4 recipe时推断gate_up融合为Nvfp4Static/Dynamic。'),
('modelopt_mixed_precision_does_not_infer_missing_sibling_linear',D,'只列gate或o_proj时不量化对应未列down或qkv Mock Linear，返回UnquantizedLinearMethod。'),
('modelopt_mixed_precision_dispatches_linear_layer',D,'NVFP4、W4A16_NVFP4、FP8、MXFP8四recipe分别检查Mock Linear方法及精确weight/activation key配对。'),
('modelopt_mixed_precision_builds_w4a16_sibling_config',C,'混合配置同时包含NVFP4与W4A16_NVFP4时建立两个独立quant_method字段的子配置。'),
('modelopt_block_quant_rejects_incompatible_block_metadata',I,'IQ2_XS/IQ2_XXS/Q8_0各recipe逐次破坏group_size为0、payload75、packing unknown，按codec block contract ValueError拒绝。')]])
add(p,[mo+':2895'],[('test_modelopt_linear_method_builder_registry_override','compute.kernel_registration_binding','临时注册NVFP4 builder返回sentinel，对Mock Linear实际config分派返回同一sentinel，证明builder override生效。')])
add(p,[mo+':879'],[
('test_modelopt_nvfp4_moe_dispatches_to_marlin_when_w4a16',D,'NVFP4与W4A16_NVFP4对mock backend selector分别传dynamic activation和None，核对use_a16；selector已mock，名称虽称Marlin但没有证明最终选中Marlin。'),
('test_modelopt_nvfp4_moe_validates_activation_override',D,'模拟backend返回下per-token NVFP4只有W4A4+FlashInferTRTLLM允许；W4A16、MXFP8 override、Marlin后端分别匹配错误，成功时检查per_token标志和selector key。')])
add(p,['vllm/model_executor/model_loader/weight_utils.py:200',mo+':737',mo+':1781'],[
('test_modelopt_nvfp4_moe_activation_override',C,'Mock模型HF config经真实get_quant_config加载modelopt_fp4/mixed，per_token开关正确写入MoE override且不向原HF字典泄漏_online_quantization_args。')])
add(p,[mo+':2375',mo+':2618',mo+':2622',mo+':2633',mo+':2733',mo+':2764',mo+':2819'],[
('test_modelopt_fp8_pb_wo_hides_output_padding',B,'模拟FP8 kernel/TP1下logical2624x128分配并后处理补零到2688行；kernel计划用2688，mock返回4x2688后wrapper裁剪前2624列并加bias，输出contiguous且kernel bias=None；不验证GEMM数值。'),
('test_modelopt_fp8_pb_wo_rejects_non_128_input','compute.input_admissibility','真实block128 scheme.create_weights收到input100时拒绝输入未128对齐。')])
add(p,['vllm/model_executor/models/utils.py:1192','vllm/model_executor/models/qwen3_dflash.py:424',mo+':270',mo+':2030'],[
('test_draft_quant_config_uses_runtime_module_names',C,'prefix空/model×start_layer0/1时draft mixed量化键完成encoder别名、midlayer重命名、层偏移与prefix补齐；原native dict不变，通配ignore保留且不误偏移nested lilicorr或已global88。'),
('test_draft_quant_config_preserves_quark_settings',C,'Quark配置映射draft prefix model/start88后与原深拷贝字典完全相同，避免重写其语义配置键。'),
('test_draft_weight_aliases_match_module_boundaries',M,'DFlash mapper在裸/model.前缀下仅匹配encoder.fc/output_norm_enc/midlayer完整模块名及其weight；fcx、_extra等保持原名。')])
add(p,[mo+':2030','vllm/model_executor/models/gemma4.py:1427','vllm/model_executor/models/gemma4_mm.py:1068','vllm/model_executor/models/utils.py:58'],[
('test_modelopt_mixed_precision_composes_gemma4_mappers',C,'先多模态再text Gemma4 mapper组合后，mixed expert及gate_up键保留language_model.model前缀，不被二次误写，resolve仍NVFP4。')])
add(p,['vllm/model_executor/layers/vocab_parallel_embedding.py:486'],[
('test_vocab_parallel_embedding_weight_loader_accepts_scalar_scale',M,'Mock holder无output_dim路径，标量2.0加载到单元素Parameter时reshape为(1,)且值完全相等。')])
add(p,[mo+':2431',mo+':2447','vllm/model_executor/layers/linear.py:351','vllm/model_executor/layers/linear.py:407'],[
('test_modelopt_mxfp8_preserves_per_row_checkpoint_scales',M,'MXFP8 kernel factory替身下真实ReplicatedLinear64x64的weight_scale loader保持arange128 reshape64x2 uint8 checkpoint值。')])
DEFERRED += [{'path':p,'reason':'本批只确认显式config/dispatch/metadata/wrapper及名称映射；未完成完整runner和专家block loader链路的其余入口保留待审。'}]
p='tests/quantization/test_online.py';on='vllm/model_executor/layers/quantization/online/';wu='vllm/model_executor/model_loader/weight_utils.py'
add(p,[on+'base.py:315',on+'base.py:458',on+'requantization.py:42','tests/conftest.py:254'],[
('test_online_linear_methods_resolve_activation_quant_key',D,'简化_TestLinear上五种weight key选择对应真实online方法；省略activation时实例和resolver返回方法默认activation key；仅构造/配置不前向。'),
('test_online_activation_override',D,'linear和MoE各五种weight key乘activation=None/FP8DynamicTensor，只有MXFP4允许显式覆盖，其余真实resolver以activation override错误拒绝。')])
add(p,[on+'mxfp4.py:88',on+'mxfp4.py:157',on+'requantization.py:42','tests/conftest.py:254'],[
('test_online_mxfp4_activation_key_is_honoured_or_rejected',D,'None及全部命名activation key分别构造真实MXFP4 linear/MoE：静态key明确拒绝；运行环境缺后端允许指定三类错误；成功路径检查方法key及kernel.config或MoE backend计算key一致，默认MXFP4 MoE允许BF16后端。未前向。')])
add(p,[on+'base.py:458',on+'mxfp4.py:88','vllm/model_executor/layers/quantization/input_hadamard.py:62','tests/conftest.py:254'],[
('test_online_hadamard_target_wraps_resolved_method',D,'targets=*linear*、MXFP4、activation=None、Hadamard32真实构造返回InputHadamard包装器，内部方法MXFP4且activationNone/size32；不测旋转值。')])
add(p,[wu+':200','vllm/config/quantization.py:323','vllm/config/quantization.py:368'],[
('test_legacy_fp8_online_quantization_uses_per_tensor_shorthand',C,'临时最小Llama配置+legacy fp8别名下，返回online配置args与同生产resolve(fp8_per_tensor, overrides)一致并检查弃用消息；覆盖None/ignore/block配置，证明别名接线而非独立merge算法。'),
('test_online_activation_only_spec_raises',I,'临时最小Llama/online配置仅给MoE或targets activation=mxfp8、不含weight，get_quant_config明确ValueError，禁止静默无操作。'),
('test_online_overlay_requires_normalized_quantization_config',I,'compressed-tensors空config_groups checkpoint加原始dict online overlay，get_quant_config在未归一化类型检查处AssertionError。'),
('test_online_shorthand_selects_checkpoint_or_online_config',D,'五种临时最小Llama场景：fp8block/mxfp4/mxfp8无checkpoint元数据选online；mxfp8内联或hf_quant_config文件元数据选ModelOptMxFp8Config，均无额外online overlay。')])
add(p,[wu+':200','vllm/model_executor/layers/quantization/auto_awq.py:239','vllm/model_executor/layers/quantization/auto_awq.py:243'],[
('test_online_overlay_loads_checkpoint_config_file',C,'临时AWQ quantize_config.json的wbit4/group128/zero_pointTrue加载为AutoAWQ配置，online MXFP8 overlay args保持传入对象；不绕过checkpoint文件解析。')])
add(p,['vllm/model_executor/models/qianfan_ocr.py:98','vllm/model_executor/models/internvl.py:631',on+'base.py:315','vllm/config/quantization.py:323'],[
('test_qianfan_online_fp8_keeps_vision_layers_unquantized',D,'手工__new__ Qianfan模型+两层vision伪HF配置，三种FP8 shorthand补充vision/lm_head/mlp1忽略项；语言qkv仍选择FP8方法，其他共享args实例及args.ignore不被改写。')])
add(p,[on+'nvfp4.py:95',on+'nvfp4.py:153','vllm/model_executor/layers/quantization/utils/flashinfer_utils.py:298'],[
('test_online_nvfp4_reuses_kernel_when_weights_are_reprocessed',B,'平台能力/backend/格式转换/replace_parameter/kernel全为替身，三次_setup_kernel接收1/2/0.5偏移的8张量；实际g1/g2尺度copy保持指针且每轮postprocess观察新值，kernel/config只建一次；gate=True/False分别转发hidden alignment256/512。不证明量化kernel数值。')])
add(p,[on+'moe_base.py:123',on+'moe_base.py:132',on+'moe_base.py:173',on+'moe_base.py:202'],[
('test_online_requantized_moe_padding',R,'手工online方法及mock源返回CPU arange两专家、hidden16/intermediate32，gated单/双w13 shard在目标16/24及32/48尺寸下与手写按shard复制并补零参考严格相等；无padding保留身份；缩维拒绝，有bias且需padding拒绝。')])
add(p,[on+'base.py:315',on+'base.py:381',on+'base.py:458','tests/quantization/utils.py:33','tests/conftest.py:254'],[
('test_online_quantization_targets',D,'实际ColumnParallelLinear 1x1/disable_tp上三组target匹配：o_proj block、qkv tensor、q/k/v展开tensor；核对真实方法类及source/key/pattern元数据；未匹配路径unquantized且不新增元数据。'),
('test_online_quantization_records_global_config',D,'实际ColumnParallelLinear 1x1/disable_tp上global linear=fp8_per_block选择BlockFP8方法，记录该prefix的linear/spec/None元数据。'),
('test_online_quantization_targets_ignore_collision',I,'同一o_proj同时被targets及ignore命中，真实resolver在dispatch时报ValueError。'),
('test_online_quantization_targets_reject_unsupported_layer',D,'lm_head目标匹配实际VocabParallelEmbedding 1x1但非Linear/MoE，get_quant_method明确拒绝不支持的层类型。')])
add(p,[on+'fp8.py:116'],[
('test_is_tp_sharded_false_when_scale_is_already_global',K,'伪TP4布局：replicated全尺寸64x32不需要跨rank尺度归约；column局部output8且per-channel不归约output时False，默认归约output时True。只检查归约计划谓词。')])
add(p,['vllm/model_executor/layers/fused_moe/all2all_utils.py:56'],[
('test_nvfp4_one_sided_rejects_unsupported_input_dtype','compute.input_admissibility','hidden6144+NVFP4配置、未打包输入FP32/FP8E4M3/FP8E5M2/int8四种dtype在layout计算入口被ValueError拒绝，要求FP16/BF16。未执行通信。')])
DEFERRED += [{'path':p,'reason':'其他重载全链路、后端通信容量、实际模型数值及TP量化入口未完成生产实现审查，保留待审；只确认本批显式列出的dispatch/config/padding/call边界'}]
