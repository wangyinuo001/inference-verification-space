import ast,csv,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
P=OUT.parents[1];ROOT=P.parents[3];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'
D='compute.kernel_dispatch_binding';N='compute.operator_reference_relation';E='compute.execution_relation'
S={};DEFERRED=[]
def add(path,impl,items):
 for name,unit,desc in items:S[path,name]=(unit,desc,impl)
p='tests/kernels/mhc/test_mhc_kernels.py'
impl=['vllm/model_executor/kernels/mhc/tilelang.py:61','vllm/model_executor/kernels/mhc/tilelang.py:256','vllm/model_executor/kernels/mhc/torch.py:6','vllm/model_executor/layers/mhc.py:384','vllm/models/deepseek_v41/nvidia/ops/mhc.py:60','vllm/model_executor/kernels/mhc/triton.py:118']
items=[
('deepseek_v41_all_reduce_fusion_requires_kernel_support',D,'替身TP通信器仅在TP4、非EP、hidden5120、hc4及multicast指针有效时允许融合。'),
('hc_collapse_preserves_weighted_residual_sum',N,'Triton collapse对照FP32逐stream加权和后转BF16；连续输出、非连续输入、尾部和空行，以及one-hot和抵消行精确检查。'),
('v41_dspark_head_collapses_with_last_ffn_mix',E,'DSpark层和post为替身，检查上一层mix传递身份，并以最后FFN mix对固定streams加权得到head输入；不证明整个模型精度。'),
('hc_collapse_before_gather_is_bitwise_equal',E,'单设备模拟SP分片与padding，先合并再collapse和各片collapse后合并结果逐位相同。'),
('deepseek_v41_mhc_pre_delayed',N,'TileLang delayed pre对照PyTorch投影/Sinkhorn/旧mix折叠；fused_norm分支改对照未融合TileLang加CUDA RMSNorm，按DeepGEMM与否使用不同误差预算，另核对graph replay。'),
('deepseek_v41_mhc_overlap_preserves_outputs_on_replay',E,'侧流overlap对照串行TileLang，join前消费layer input仍正确；修改输入后五次graph replay结果精确相同。'),
('deepseek_v41_mhc_fused_post_pre_delayed',N,'融合post/pre与独立einsum加PyTorch delayed参考比较系数；残差对照独立post，collapse对照以融合残差为输入的非融合路径；检查aux均值、形状及replay。'),
('deepseek_v41_decoder_mixes_match_torch',E,'attention/FFN替身缩放，Engram可选替身；decoder融合/overlap混合逻辑对照替换mHC为PyTorch的同decoder运行；检查carried mix与graph适用边界。'),
('deepseek_v41_capture_previous_aux',E,'decoder的attention/FFN及可选Engram为替身，前层aux等于独立post结果的stream均值且不含后续Engram注入；未请求capture返回None。'),
('deep_gemm_mega_mhc_correctness',N,'DeepGEMM mega mHC对照独立TileLang post加delayed pre含RMSNorm；残差/输入与系数分别设容差；变更x后graph replay逐位等于eager。'),
('mhc_pre_tilelang',N,'TileLang pre对照测试内FP32投影、sigmoid、Sinkhorn和stream加权参考；可选RMSNorm在参考末尾执行。'),
('hc_prenorm_gemm_tilelang',N,'TileLang prenorm输出矩阵乘法和平方和分别对照FP32 PyTorch实现，使用独立容差。'),
('mhc_post_tilelang',N,'TileLang post对照PyTorch转置混合矩阵bmm加sublayer项后转BF16。'),
('mhc_fused_post_pre',N,'融合post/pre各输出对照先PyTorch post再pre，检查残差、post mix、res mix和layer input。'),
('mhc_fused_post_pre_reuse_residual_is_exact',E,'reuse_residual与普通路径所有输出精确一致，普通路径不改输入，且仅非融合路径复用残差存储指针。'),
('mhc_pre_rocm_fallback_applies_norm',N,'模拟AITER无fused norm且禁TileLang，ROCm回退pre输出对照PyTorch mHC和显式RMSNorm。'),
('mhc_fused_rocm_fallback_applies_norm',N,'禁用AITER融合与TileLang，ROCm回退post/pre输出对照PyTorch序列及显式RMSNorm。'),
('mhc_pre_delayed_rocm_aiter',N,'ROCm AITER delayed pre对照PyTorch参考，carried或identity入口均检查原残差对象保留及四个输出。'),
('mhc_pre_delayed_rocm_aiter_fused_rms_norm',N,'AITER fused delayed seam按post/no_post/identity_pre对照PyTorch post、delayed pre再RMSNorm；无post时保留残差对象。'),
('mhc_post_collapse_rms_norm_warmup_covers_runtime',D,'清空实际Triton编译缓存，warmup恰生成一个变体且不同runtime token数不增缓存；非数值参考检查。'),
('deepseek_v41_rocm_mhc_overlap_matches_fused_seam',E,'ROCm overlap对照串行MHCPreDelayedOp，残差/系数精确一致，join前后输入允许一个BF16 ULP，修改输入后graph replay重复检查。'),
('deepseek_v41_rocm_decoder_overlap_matches_serial',E,'attention/FFN和可选Engram为替身；ROCm decoder FULL图且token不超限才调用overlap split，结果与无overlap串行路径比较。'),
('mhc_pre_delayed_rocm_aiter_declines_unsupported',D,'spy包裹真实AITER调用：broadcast不得进入AITER，norm不得进入unfused AITER，支持时进入fused；无norm正控必须进入AITER。broadcast结果附PyTorch参考检查。'),
('mhc_pre_rocm_aiter_fuses_norm',N,'AITER pre含RMSNorm对照PyTorch mHC pre加F.rms_norm。'),
('mhc_fused_rocm_aiter_fuses_norm',N,'AITER fused post/pre含norm，对照PyTorch post/pre加F.rms_norm，逐项检查四返回张量。'),
('hc_head_triton',N,'Triton head写入预填NaN的缓冲区，返回None且无NaN，对照FP32归一化投影sigmoid后stream加权参考。'),
('hc_head_tilelang',N,'TileLang head对同一FP32归一化投影sigmoid参考，附输出形状、BF16和无NaN检查。'),
('deepseek_v4_layer_reuse_residual_keeps_caller_input',E,'attention/FFN替身缩放时，decoder reuse_residual与普通执行所有输出精确相等且不修改caller输入，覆盖carried分支。'),
('deepseek_v4_mhc_broadcast_finalize_sums_hc_streams','model.parameter_representation_relation','模拟首PP rank，finalize权重按hc stream维求和形成broadcast参数。'),
('deepseek_v4_mhc_broadcast_refit_refreshes_in_place','model.parameter_representation_relation','权重增加1后重新finalize，broadcast缓冲区对象身份不变且数值等于新权重按stream求和。'),
('mhc_fused_post_pre_delayed_rocm_aiter',N,'AITER融合delayed seam对照PyTorch post再delayed pre，按一BF16 ULP容差比较残差及系数/输入。'),
('mhc_fused_post_pre_delayed_falls_back_for_large_batches',D,'AITER选择启发式对1 token不建议unfused，对2^20 token建议unfused；只检查谓词。'),
('hc_expand_preserves_bits_in_eager_and_graph',E,'hc_expand与unsqueeze/expand/contiguous参考逐字节相同，含NaN、负零、stride、空输入；改输入后graph replay重复比较。')]
add(p,impl,[('test_'+n,u,d) for n,u,d in items])
for n in ['hc_collapse_custom_op_supports_compile','mhc_fused_post_pre_delayed_custom_op_supports_compile','mhc_pre_delayed_custom_op_supports_compile']:
 DEFERRED.append({'snapshot':'vllm','path':p,'test_name':'test_'+n,'reason':'仅torch.library.opcheck注册/编译契约；不混入数值性质'})
p='tests/model_executor/layers/test_fused_shared_expert.py'
impl=['vllm/model_executor/layers/quantization/utils/config_utils.py:129','vllm/model_executor/layers/fused_moe/utils.py:64','vllm/models/deepseek_v4/amd/model.py:199','vllm/model_executor/models/utils.py:525','vllm/model_executor/models/qwen3_next.py:107','vllm/_aiter_ops.py:2916','vllm/models/glm5next/common/model.py:1422','vllm/model_executor/layers/quantization/quark/quark.py:798']
names={
'determine_expert_counts_fuse_shared_experts_override':'显式融合开关决定返回的共享专家数为2或0。',
'resolve_layer_fused_shared_expert_skips_compatibility_when_disabled':'模拟融合关闭，安装fail替身确保不调用兼容性检查。',
'resolve_layer_fused_shared_expert_normalizes_unavailable_aiter':'模拟AITER不可用返回None，最终融合资格严格为False。',
'resolve_layer_fused_shared_expert_passes_module_prefixes':'spy记录兼容性调用，断言expert/shared前缀和moe_backend完整传递。',
'resolve_layer_fused_shared_expert_logs_skip_reason':'模拟不兼容或EP禁用后融合关闭，日志包含具体原因。',
'deepseek_v4_shared_expert_fse_uses_mtp_quantization_config_prefix':'替身DeepSeek量化类与配置，MTP层前缀转成mtp.0.ffn后匹配FP4配置并允许融合。',
'deepseek_v41_quant_config_rejects_fp8_mtp_shared_expert_fse':'真实V4.1量化配置配模拟当前模型配置，MTP shared为FP8、routed为FP4时不融合并报告mtp路径。',
'deepseek_v41_fp8_experts_keep_fp8_fse_check':'V4.1 expert_dtype=fp8时沿用block FP8融合资格。',
'deepseek_v4_heterogeneous_fhmoe_token_policy':'spy支持函数按token阈值决定异构融合；零token不查询支持函数。',
'deepseek_v4_heterogeneous_fhmoe_aiter_capability':'伪AITER模块及签名验证能力探针，要求正整数token、完整shared参数和严格True返回；异常或truthy整数均拒绝。',
'deepseek_v4_heterogeneous_fhmoe_aiter_capability_catches_import_error':'伪模块导入失败被能力探针转换为False。',
'deepseek_v4_heterogeneous_fhmoe_aiter_capability_catches_signature_error':'替身inspect.signature抛TypeError或ValueError后能力探针返回False。',
'deepseek_v4_heterogeneous_fhmoe_compatibility_gates':'模拟gfx950/AITER及标准DSV4模型形状，DP/PCP、topk_method或FHMoE支持不符禁用异构融合；探针总以1 token查询。',
'glm5_next_fuses_shared_experts_only_in_tuned_setups':'模拟gfx950及并行配置，TP4/8且无DP/PCP/EP才允许融合，其他组合检查warning原因。',
'quark_shared_expert_fse_compatibility':'Quark shared投影在exclude中则拒绝，否则允许。',
'online_targets_off_shared_expert_defer_to_checkpoint':'online target仅覆盖无关linear_attn层时，shared融合仍遵守checkpoint排除列表。',
'online_only_one_shared_projection_matching_keys':'仅down_proj在线MXFP4但其他shared及routed同key时允许融合。',
'online_fp8_shared_expert_rejects_quark_mxfp4_fse':'shared在线FP8与routed MXFP4 key不匹配时拒绝并说明。',
'online_mxfp4_shared_expert_fse_compares_resolved_keys':'online activation省略按默认MXFP4，显式null按BF16；仅与routed最终weight/activation key相同才融合。',
'modelopt_mxfp8_shared_expert_fse_compatibility':'ModelOpt MXFP8要求shared/routed同时排除或同时量化，不对称则拒绝。',
'modelopt_mxfp8_shared_expert_fse_keeps_the_stored_format':'ModelOpt shared存储MXFP8，routed NVFP4时不融合，routed MXFP8时允许。',
'quark_shared_expert_fse_exclude_is_scoped_to_the_layer':'第78层shared排除不影响第3层融合，第78层返回排除原因。',
'quark_shared_expert_fse_ignores_sibling_gate_exclusions':'相邻shared_expert_gate与gate排除不算shared投影排除；真正down_proj排除必须关闭融合。',
'quark_shared_expert_fse_requires_matching_layer_quant_configs':'shared所有投影FP8而routed全局FP4，拒绝融合并报告配置差异。',
'quark_shared_expert_fse_rejects_partial_packed_projection_override':'routed FP8但packed shared仅gate片覆盖FP8、up沿用全局FP4时拒绝融合。',
'non_quark_shared_expert_fse_is_incompatible':'ExpertsInt8不支持融合兼容性推断，返回False及类名原因。',
'block_fp8_shared_expert_fse_is_compatible':'有128x128权重block的FP8配置允许融合。',
'per_tensor_fp8_shared_expert_fse_is_incompatible':'缺少weight_block_size的per-tensor FP8拒绝融合。',
'store_dtype_fp8_shared_expert_fse_is_incompatible':'FP8 routed存储改为MXFP4/NVFP4时拒绝shared融合。',
'fp8_shared_expert_fse_rejects_asymmetric_ignored_layers':'只忽略routed或某shared投影的非对称忽略策略拒绝融合。',
'fp8_shared_expert_fse_allows_symmetric_ignored_layers':'routed及shared全部忽略量化时允许融合。',
'fp8_shared_expert_fse_expands_packed_projections':'packed gate/up均被忽略而routed未忽略，展开后检测两侧不对称并拒绝。',
'quark_ocp_mx_shared_expert_uses_supported_tp_shards':'宽640组32在TP1/2/4下对齐，不需复制shared。',
'quark_ocp_mx_shared_expert_is_replicated_for_tep8':'宽640/TP8分片80不对齐组32，EP启用时选择复制。',
'quark_ocp_mx_shared_expert_is_replicated_when_not_tp_divisible':'宽672/TP8分片84不对齐组32，EP启用时选择复制。',
}
add(p,impl,[('test_'+n,D,d+' 仅资格、配置或分派检查，未执行MoE数值计算。') for n,d in names.items()])
add(p,impl,[('test_'+n,u,d) for n,u,d in [
('deepseek_v4_heterogeneous_fhmoe_keeps_native_intermediate_width','model.parameter_representation_relation','FP8共享权重保留native中间宽384；E8M0块scale按128行/4列重复扩展，w2 scale补齐列为0x7f；只检查scale字节与形状，不读取未初始化权重数值。'),
('deepseek_v4_heterogeneous_fhmoe_rejects_invalid_routes','compute.input_admissibility','异构路由拒绝非2D、两张量形状不等和topk6却非7列的输入。'),
('aiter_fused_moe_validates_shared_expert_arguments','compute.input_admissibility','无shared参数且ID=-1合法未启用；缺失权重/scale或负shared ID拒绝；完整四张量与非负ID返回True。'),
('deepseek_v4_shared_expert_mapper_fallback','model.parameter_binding','checkpoint共享权重/scale名按layer融合集合改写到追加expert槽，非融合层保留独立down_proj名。'),
('fused_shared_expert_layers_matches_container_index_space','model.parameter_binding','替身MoE树的ModuleList相对索引和ModuleDict绝对索引产生正确融合层集合，权重重定向遵守该集合。'),
('mixed_fused_shared_experts_rejected_with_ep_or_nixl_ep','deployment.configuration_admissibility','替身两层融合策略混合时，EP或nixl_ep配置抛NotImplementedError及对应原因。'),
('glm5_next_fused_shared_expert_loads_into_the_shared_slot','model.parameter_binding','共享槽数为0/1且多于1拒绝；shared参数名重定向至routed末尾槽288，原routed名不变；未实际加载张量。'),
('quark_shared_expert_fse_rejects_partially_excluded_packed_projection','deployment.configuration_admissibility','packed gate/up仅排除gate时拒绝不一致量化方案。'),
('quark_ocp_mx_group_size_for_layer','deployment.effective_configuration','当前shared down_proj组大小32，忽略相邻gate不影响，忽略本投影返回None。'),
('quark_ocp_mx_shared_expert_rejects_plain_tp8','deployment.configuration_admissibility','宽640/TP8分片80不对齐组32且无EP时拒绝。'),
('quark_layer_config_from_name_checks_packed_projections','deployment.configuration_admissibility','packed w1/w3配置FP4/FP8不一致，按名获取配置抛ValueError。'),
('quark_packed_layer_config_must_match_global_config','deployment.configuration_admissibility','packed w1覆盖FP8而w3沿用FP4全局配置，配置匹配拒绝。'),
('fp8_shared_expert_fse_propagates_partial_shard_exclusion','deployment.configuration_admissibility','FP8 packed gate/up仅排除gate，传播分片部分忽略ValueError。')]])
for n,reason in [('models_fse_init','涉及13架构与MTP装配加载，已读测试但尚未逐架构核对生产加载路径；保留未确认'),('deepseek_v4_heterogeneous_fhmoe_accepts_appended_shared_route','成功返回型检查无显式assert；暂不纳入本批断言证据生成')]:DEFERRED.append({'snapshot':'vllm','path':p,'test_name':'test_'+n,'reason':reason})
T='compute.tuning_configuration_binding';R='compute.kernel_registration_binding'
p='tests/kernels/helion/test_config_manager.py';impl=['vllm/kernels/helion/config_manager.py:47','vllm/kernels/helion/config_manager.py:172']
descs={
'TestConfigSet::test_config_set_creation':'空配置集保留kernel名，平台列表为空。',
'TestConfigSet::test_config_set_from_dict':'平台及CaseKey字典解析为Helion配置，block/warp/stage/pid字段保留。',
'TestConfigSet::test_config_set_get_config_keyerror':'不存在平台或配置key分别抛带对应上下文的KeyError。',
'TestConfigSet::test_config_set_get_platforms':'配置平台列表按字典序返回。',
'TestConfigSet::test_config_set_get_config_keys':'平台配置keys按字符串顺序返回，缺失平台返回空列表。',
'TestConfigSet::test_config_set_to_dict':'配置集序列化回字典后CaseKey及所有原配置字段相等。',
'TestConfigManager::test_config_manager_creation_default_base_dir':'默认配置根目录名为configs。',
'TestConfigManager::test_config_manager_creation_custom_base_dir':'显式配置根目录规范化为resolved路径。',
'TestConfigManager::test_get_config_file_path':'kernel名和可选平台映射到kernel目录或平台JSON路径。',
'TestConfigManager::test_ensure_base_dir_exists':'首次ensure创建嵌套配置目录并返回同一路径。',
'TestConfigManager::test_load_config_set_file_not_exists':'不存在kernel目录加载为空配置集且保留名称。',
'TestConfigManager::test_load_config_set_valid_file':'按平台JSON加载配置，保留key及block/warp等字段。',
'TestConfigManager::test_load_config_set_invalid_json':'无效JSON加载回退为空平台配置集。',
'TestConfigManager::test_save_config_set':'保存为按平台JSON数组，回读key与完整配置原值相等。',
'TestConfigManager::test_save_config_set_creates_directory':'保存配置时创建不存在的嵌套目录及平台文件。',
'TestConfigManager::test_get_platform_configs':'按kernel/platform返回对应CaseKey配置含default，其他平台数据隔离，缺失平台为空。',
'TestConfigManager::test_singleton_returns_same_instance':'同一根目录反复创建返回同一管理器对象。',
'TestConfigManager::test_singleton_with_default_base_dir':'默认根目录反复创建返回同一对象及路径。',
'TestConfigManager::test_singleton_error_on_different_base_dir':'已有管理器时指定不同根目录抛ValueError。',
'TestConfigManager::test_reset_instance_allows_new_base_dir':'reset后允许新根目录并获得不同管理器对象。',
'TestConfigManager::test_get_instance_returns_existing':'get_instance返回已初始化管理器身份。',
'TestConfigManager::test_get_instance_raises_if_not_initialized':'初始化前get_instance抛RuntimeError。'}
add(p,impl,[(n,T,d+' 使用临时JSON目录/真实配置对象，未执行计算内核。') for n,d in descs.items()])
p='tests/kernels/helion/test_register.py';impl=['vllm/kernels/helion/register.py:85','vllm/kernels/helion/register.py:146','vllm/kernels/helion/register.py:253','vllm/kernels/helion/register.py:443','tests/kernels/helion/helpers.py:28']
# Map explicitly inspected methods by class-qualified AST names below.
short={
'rejects_autotuner_fn':('deployment.configuration_admissibility','配置指定冲突autotuner_fn时拒绝。'),
'warns_on_static_shapes_true':('deployment.effective_configuration','static_shapes=True时warning说明将覆盖为False。'),
'init_raises_without_picker':(D,'模拟配置管理器与GPU，缺少picker时构造配置内核拒绝。'),
'config_selector_validates_picker_result':('compute.kernel_plan_consistency','模拟Helion装饰与平台，picker返回未登记key时selector抛ValueError。'),
'config_selector_handles_none_from_picker':('compute.kernel_plan_consistency','picker返回None时selector返回default key对应配置对象。'),
'create_decorated_kernel_passes_helion_settings':('compute.kernel_call_binding','spy helion.kernel收到print_output_code=True且static_shapes强制False。'),
'key_and_config_selector_use_same_logic':('compute.kernel_plan_consistency','模拟配置器，输入batch50按picker选择batch64 key，key函数字符串与autotune配置对象一致。'),
'init_disables_on_missing_configs':(D,'模拟平台无配置时wrapper禁用并记录No configs available原因。'),
'disabled_wrapper_raises_on_call':(D,'模拟无配置wrapper调用抛disabled RuntimeError。'),
'disabled_wrapper_get_configured_op_raises':(D,'禁用wrapper获取配置实现抛disabled RuntimeError。'),
'disabled_wrapper_supports_get_inputs':('compute.kernel_call_binding','禁用wrapper仍调用替身input_generator并原样返回其对象。'),
'disabled_wrapper_supports_run_autotune':('compute.kernel_call_binding','禁用wrapper仍可创建替身autotune实现并返回该实现的配置对象；未执行真实调优。'),
'init_caches_configured_kernel':(R,'模拟编译器和配置管理器，wrapper急切构建并缓存配置实现，两次获取对象相同。'),
'init_eagerly_initializes_hop_path':(E,'HOP路径构建时探测GPU一次，执行时禁止再次探测；真实加法Helion输出对x+y检查。'),
'init_eagerly_initializes':(R,'CustomOp路径构建时探测GPU一次并建立配置实现和torch.ops条目。'),
'get_or_register_custom_op_returns_cached_op':(R,'替身torch.ops命名空间已有同名op，注册接口返回原对象。'),
'get_or_register_custom_op_registers_new_op':(R,'替身注册接口仅调用一次，注册decorated函数及mutates_args保留原对象，返回新op。'),
'get_registered_kernels_returns_copy':(R,'查询registry返回等内容的独立字典副本。'),
'get_kernel_by_name_returns_kernel':(R,'按名查询返回已注册wrapper对象。'),
'get_kernel_by_name_returns_none_for_missing':(R,'缺失名称查询返回None。'),
'register_kernel_auto_generates_fake_impl':(R,'mock infer_fake_impl按原kernel和None设置调用一次，生成fake绑定在wrapper。'),
'register_kernel_creates_wrapper':(R,'临时配置registry注册后wrapper保留显式名及原kernel函数身份。'),
'register_kernel_auto_detects_name':(R,'未传名称时取原kernel函数名。'),
'register_kernel_registers_in_global_registry':(R,'注册后全局registry同名键指向新wrapper。'),
'register_kernel_passes_helion_settings':(R,'注册settings对象原样绑定wrapper。'),
'register_kernel_supports_decorator_syntax':(R,'显式注册名和fake实现被保留在wrapper。'),
'register_kernel_raises_on_duplicate_registration':(R,'重复注册相同名称抛ValueError，不覆盖原条目。'),
'register_kernel_rejects_autotuner_fn_in_settings':('deployment.configuration_admissibility','注册阶段拒绝替身settings中的自定义autotuner。'),
'register_kernel_no_warning_with_static_shapes_false':('deployment.effective_configuration','static_shapes=False注册不产生覆盖warning。'),
'disabled_kernel_appears_in_registry':(R,'模拟平台缺少配置，禁用wrapper仍以原名登记全局registry。'),
'compiled_graph_contains_helion_hop':(E,'真实加法Helion经torch.compile自定义backend，输出与eager相等；附FX中HOP节点及一次backend调用检查。'),
'inductor_backend_compiles_helion_hop':(E,'真实加法Helion带乘法/偏置/relu外层，Inductor输出对eager相等；附生成源码仅一个triton.jit内核检查。'),
}
for cls in ast.parse((SRC/p).read_text()).body:
 if isinstance(cls,ast.ClassDef):
  for fn in cls.body:
   if isinstance(fn,ast.FunctionDef) and fn.name.startswith('test_'):
    key=fn.name.removeprefix('test_')
    if key in short:
     u,d=short[key];add(p,impl,[(cls.name+'::'+fn.name,u,d)])
    else:DEFERRED.append({'snapshot':'vllm','path':p,'test_name':cls.name+'::'+fn.name,'reason':'仅正常返回无显式断言，暂不纳入断言证据生成'})
PROPOSED=[
{'id':T,'name':'调优配置的键、存储与作用域绑定','kind':'obligation','phase':'model_execution','state_scope':'deployment','conditions':['kernel_selection_profile','operator_profile'],'property':'调优配置按算子、平台和形状键检索，读写及表示转换保留键与配置字段；配置源路径按声明规则确定，共享管理器的初始化、复用及重置保持配置源作用域一致，缺失或无效配置返回声明的空结果或错误。','observable':['configuration_key','configuration_fields','configuration_source_path','manager_identity','lookup_error'],'oracle':'比较配置读写前后的键和值、平台隔离、路径和管理器身份，逐项记录缺失/错误/重置条件。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':'vllm/kernels/helion/config_manager.py','line':47,'kind':'source'},{'path':'vllm/kernels/helion/config_manager.py','line':172,'kind':'source'},{'path':'tests/kernels/helion/test_config_manager.py','line':154,'kind':'existing_test'}]}},
{'id':R,'name':'计算实现注册名称与包装对象绑定','kind':'obligation','phase':'model_execution','state_scope':'deployment','conditions':['kernel_selection_profile'],'property':'计算实现的注册名称映射到声明的原函数、包装对象、模拟实现、可变参数约定与设置；查询和重复注册遵守约定的身份、缺失及拒绝行为，初始化后复用对应配置实现，禁用状态与登记状态按接口分别保留。','observable':['registered_name','kernel_wrapper_identity','raw_function','fake_implementation','mutation_metadata','registry_snapshot','disabled_state'],'oracle':'对注册调用和查询返回的对象、元数据及名称逐项比较；明确模拟注册器和真实注册器的范围，不把登记成功当作数值正确。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':'vllm/kernels/helion/register.py','line':351,'kind':'source'},{'path':'vllm/kernels/helion/register.py','line':443,'kind':'source'},{'path':'tests/kernels/helion/test_register.py','line':824,'kind':'existing_test'}]}}
]
(OUT/'proposed-units.json').write_text(json.dumps(PROPOSED,ensure_ascii=False,indent=2))

p='tests/model_executor/test_jit_warmup.py';impl=['vllm/model_executor/warmup/jit_warmup.py:105','vllm/model_executor/warmup/jit_warmup.py:561','vllm/model_executor/warmup/jit_warmup.py:651','vllm/model_executor/warmup/jit_warmup.py:1008','vllm/model_executor/warmup/jit_warmup_triton_helper.py:135']
names={
'triton_scalar_specialization_rep':'整数值按Triton类型范围、exact1和16整除类映射到明确代表值。',
'triton_scalar_specialization_rep_rejects_out_of_range':'超出signed64至unsigned64联合范围的整数抛OverflowError。',
'trace_dispatch_expands_ranges_dedupes_and_ignores_unused_inputs':'ToyKernel经生产AST tracer展开token1..5，去除重复block key并忽略未引用debug范围。',
'warmup_range_uses_custom_advancement':'range使用next_power_of_2+1递增，得到block1/2/4/8/16。',
'warmup_range_validates_custom_advancement':'同时指定step和advance或advance不递增时拒绝。',
'warmup_cases_support_lambda_advancement':'warmup lambda递增倍增，展开1/2/4/8具体输入。',
'compile_key_uses_defaults_locals_attributes_and_expressions':'生产AST evaluator解析ToyKernel默认参数、局部变量、属性、条件和运算，得到声明key。',
'trace_dispatch_combines_zipped_rows_with_independent_values':'zip输入保持token/mode行关联，lanes独立轴做笛卡尔积，key顺序和数值明确匹配。',
'trace_dispatch_filters_with_traced_predicate':'生产trace根据block*lanes<=4谓词过滤键。',
'zip_inputs_validates_input_rows':'拒绝空行组、非mapping、空mapping、非字符串键及不一致行字段。',
'trace_dispatch_rejects_bad_positional_groups_and_duplicates':'拒绝非zip位置组以及重复定义的dispatch字段。',
'helper_calls_support_keywords_and_reject_star_kwargs':'helper具名关键字可解析为向上取整8，helper调用**kwargs被拒绝。',
'dispatch_helper_calls_resolve_python_builtins':'max/min内置函数在compile key表达式中按输入解析。',
'dispatch_body_must_be_local_assignments_then_compile_key_return':'dispatch函数体分支或非自身kwargs解包在构建阶段拒绝。',
'dispatch_can_forward_compile_key_fields':'dispatch自身kwargs可转发compile key字段并与AST字段合并，重复及额外未知字段拒绝。',
'forwarded_compile_key_fields_exclude_predicate_only_inputs':'只用于筛选的max_tokens不作为CompileKey字段转发。',
'dispatch_supports_tuple_and_mapping_subscriptions':'元组与mapping下标表达式对应key值3和7。',
'dispatch_reports_unsupported_expression_with_context':'不支持set表达式时错误包含表达式及支持类型说明。',
'warmup_compiles_all_returned_keys_in_order':'记录型compile替身按预热key顺序收到1/2/4；无真实编译执行。',
'runtime_cache_miss_compiles_and_caches_executor':'记录型compile为key1缓存对象，运行key1不重编，key2 miss仅新增一次。',
'registry_records_only_inside_model_setup_context':'上下文外注册被忽略，内部相同请求去重且未开始compile。',
'registry_capture':'capture装饰构造期收集一个请求，构造结束后注册不改变owner registry。',
'registry_expands_requests_and_deduplicates_owner_keys':'同owner两个范围预热合并为1/2/4/8唯一keys，compile替身各接收一次。',
'registry_passes_vllm_config_to_default_requests':'默认注册去重，并按方法签名传registry配置使compile key为bias7。',
'registry_default_request_supports_config_free_kernel':'不接收配置的get_warmup_keys仍正常预热key7，compile为记录替身。'}
add(p,impl,[('test_'+n,'compute.kernel_plan_consistency',d+' 生产追踪/编排代码处理测试定义的Toy kernel，不验证GPU编译器或数值内核。') for n,d in names.items()])
DEFERRED.append({'snapshot':'vllm','path':p,'test_name':'test_get_ast_full_name_handles_names_attributes_and_other_nodes','reason':'仅AST名称解析工具，不单独确认为计算计划关系'})
p='tests/model_executor/test_routed_experts_capture.py';impl=['vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:45','vllm/model_executor/layers/fused_moe/routed_experts_capturer.py:187','vllm/model_executor/layers/fused_moe/modular_kernel.py:1670','vllm/distributed/aux_output_connector/worker.py:75']
C='compute.expert_route_capture_binding'
names={
'base_router_capture_pre_eplb_mapping':'DummyRouter固定逻辑ID且映射函数加10，capture回调收到映射前ID而返回路由ID已加10；不是真实EPLB计算。',
'public_binding_binds_target_model_router':'模拟MoERunner对象，公共binding将layer7绑定到router回调并传递原ID张量。',
'public_binding_supports_direct_capture_source':'实现直接capture source协议的模块绑定layer3与原ID张量。',
'public_binding_gives_monolithic_layers_a_sink':'替身monolithic专家绑定layer-owned sink，回调带正确layer5。',
'public_binding_rejects_monolithic_without_replay_support':'替身monolithic专家声明不支持capture则binding拒绝。',
'routed_experts_sink_holds_a_dispatched_batch':'CPU真实sink分配int16，容量按max_tokens乘EP或DP group展开。',
'monolithic_capture_survives_kernel_rebuilds':'生产monolithic包装路径调用Toy专家，重建offset0/7后sink捕获当前ID，不支持capture的新专家拒绝；不验证MoE数学结果。',
'routed_experts_capturer_single_dp_no_metadata':'绕过构造分配CPU缓冲，单DP无元数据时按层写完整topk行，尾部未写保持-1。',
'single_dp_ep_capture_restores_tp_shards':'模拟TP gather返回完整含padding ID，图模式与缓冲复用下恢复有效行；单token不gather，其余两次。',
'single_dp_tp_capture_does_not_regather_full_batch':'已完整或补齐的本地batch不调用TP gather，只拷贝有效token行。',
'single_dp_tp_capture_rejects_unexpected_batch_without_writing':'输入行数与full/shard布局都不匹配时拒绝且整个buffer未改写。',
'capture_uses_pre_sp_size_after_model_shards_padding_mask':'SP缩短mask后仍按原batch descriptor恢复3行，模拟gather一次并保留尾部。',
'routed_experts_capturer_exposes_output_profile':'模拟CPU平台构造，专家数256输出uint8、257输出uint16，token维为3层x2专家且buffer容量8。',
'routed_experts_capturer_narrows_snapshot':'内部int32存储准确ID，snapshot转换uint8/uint16并拥有不同指针且值相同。',
'routed_experts_capturer_dp_naive_concatenated_all_ranks':'DP rank1从跨rank连接行中提取自己的2:5行。',
'routed_experts_capturer_dp_modular_local_tokens':'已经是DP rank1局部行时直接完整保存。',
'routed_experts_capturer_dp_ep_gathered_shards':'DP-major/TP-minor已gather数据按local_sizes跳过padding，提取rank1有效两行。',
'routed_experts_capturer_sp_modular_gathers_tp_shards':'非连续TP局部分片先连续化再以dim0提交模拟gather，只保存有效3行。',
'routed_experts_capturer_rejects_incomplete_dp_shard_groups':'local_sizes分片组数不能被DP rank数整除时拒绝，不写buffer。',
'routed_experts_capturer_rejects_layer_before_restoring_rows':'越界layer -1/4在gather前拒绝且buffer未改写。',
'routed_experts_capturer_dp_preserves_empty_and_unpadded_spans':'DP空rank/all-idle/无padding同尺寸等边界按声明span提取，不额外gather且尾部不写。',
'routed_experts_capturer_dp_unexpected_batch_raises':'DP输入行数既非全局也非本地及分片长度时拒绝，原buffer保留。',
'aux_output_worker_connector_binds_capture_on_non_output_rank':'模拟非首TP rank，建立capturer与模型绑定但无output store/buffer；prepare输出None且不snapshot。'}
add(p,impl,[('test_'+n,C,d) for n,d in names.items()])
add(p,impl,[('test_get_aux_output_connector_passes_config','compute.kernel_call_binding','替身AuxOutputWorkerConnector构造器收到原model及config且返回同一connector对象。'),('test_aux_output_worker_connector_default_capacity','deployment.effective_configuration','替身storage/缓冲构造器，按KV块数10、cache/hash块32/16、路由shape计算max_bytes2560、对象128bytes及pending16；构造阶段不提前分配。'),('test_v2_model_runner_accepts_routed_experts','deployment.configuration_admissibility','模拟配置及空entry_points时V2 runner不把routed experts capture列为不支持特性。')])
PROPOSED.append({'id':C,'name':'专家路由捕获与层及token归属对应','kind':'obligation','phase':'model_execution','state_scope':'request','conditions':['operator_profile','parallel_topology','execution_layout'],'property':'专家路由捕获回调绑定当前模型层和声明阶段的逻辑专家ID，按DP/TP分片及有效token元数据恢复本rank行并去除填充；稳定快照保留ID值及输出类型，层或批布局不合法时拒绝且不写入，无效后端与重建后不支持捕获的实现不产出陈旧路由。','observable':['captured_expert_ids','layer_id','local_token_rows','capture_buffer','snapshot_dtype','capture_callback','rejection'],'oracle':'用明确路由ID和分片元数据核对回调、写入区域与快照，分别记录真实路由或替身路由、通信是否模拟及kernel重建条件。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':'vllm/model_executor/layers/fused_moe/routed_experts_capturer.py','line':91,'kind':'source'},{'path':'vllm/model_executor/layers/fused_moe/routed_experts_capturer.py','line':187,'kind':'source'},{'path':'vllm/model_executor/layers/fused_moe/modular_kernel.py','line':1670,'kind':'source'},{'path':'tests/model_executor/test_routed_experts_capture.py','line':295,'kind':'existing_test'}]}})
(OUT/'proposed-units.json').write_text(json.dumps(PROPOSED,ensure_ascii=False,indent=2))

p='tests/model_executor/test_utils.py'
impl=['vllm/model_executor/utils.py:79','vllm/model_executor/utils.py:126','vllm/model_executor/utils.py:165','vllm/model_executor/utils.py:270','vllm/model_executor/utils.py:288','vllm/model_executor/utils.py:303','vllm/model_executor/utils.py:311','vllm/model_executor/utils.py:338','vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py:80','vllm/model_executor/parameter.py:525','vllm/model_executor/layers/quantization/utils/humming/linear.py:37','vllm/model_executor/layers/fused_moe/modular_kernel.py:538','vllm/model_executor/layers/fused_moe/experts/trtllm_lora_moe.py:579','vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py:37']
names={
'replace_parameter_preserves_custom_attribute':('model.parameter_binding','以普通Tensor或Parameter替换时，保留原参数身份、is_shuffled标记、相同字节数的存储地址，数值/device/stride与转置新数据一致。'),
'replace_parameter_keeps_object_by_default':('model.parameter_storage_layout','TP group为单rank替身；改变形状、步幅或dtype后保留ModelWeightParameter对象及loader/dim元数据，容量不缩小，仅新数据能放入旧存储时保留地址，并核对值。'),
'replace_parameter_online_quantization_frees_source':('model.parameter_storage_layout','模拟uses_meta_device量化策略，FP32权重替换BF16后存储字节数精确减半；不验证实际量化。'),
'humming_replace_keeps_loaded_params':('model.parameter_binding','Humming替换将新weight绑定旧packed参数，将scale view绑定旧scale参数；保留旧大容量、正确新值及仅运行时名称。'),
'replace_parameter_preserves_weight_loader':('model.parameter_binding','替换后普通loader函数不额外绑定self；实际调用记录精确收到当前参数和加载张量身份。'),
'replace_parameter_weight_loader_comes_from_old_parameter':('model.parameter_binding','replacement含stale loader及私有字段时，沿用旧loader且不带入私有loader字段，同时保留新is_shuffled标记。'),
'replace_parameter_does_not_rebind_plain_function_attribute':('model.parameter_binding','replacement普通函数属性保持同一函数对象，参数3调用仍返回6，无隐式self参数。'),
'replace_parameter_preserves_bound_method_attribute':('model.parameter_binding','replacement已绑定方法保留原dispatcher接收者，调用7记录在原dispatcher。'),
'replace_parameter_registers_new_param_for_requantization_source':('model.parameter_binding','old参数标记requantization_source时另建参数，旧全零数据保留，新参数全一。'),
'move_parameter_aliases_one_param':('model.parameter_binding','checkpoint/runtime两名称指向同一参数；替换后双方值更新且loader/dim保留，kernel参数列表去重并随set_kernel_parameter切换运行时名称。'),
'move_parameter_rejects_another_param':('model.parameter_binding','目标名称已被另一个参数占用时move_parameter抛ValueError；测试未另断言拒绝后的数据状态。'),
'retired_param_keeps_data_but_is_not_live':('model.parameter_binding','retire保留scale数据但live查询及kernel列表隐藏它，set_kernel_parameter后恢复live身份。'),
'replace_parameter_with_own_parameter_keeps_layout':('model.parameter_storage_layout','传回已登记PackedvLLMParameter时保持对象/类/loader/输入输出维和packing标签，数据保持新全一值。'),
'kept_param_ops_skip_python_torch_function':('model.parameter_representation_relation','替换保留ModelWeightParameter类型，其torch-function判定与普通Parameter相同，eager加1返回普通Tensor且数值全2。'),
'transposed_kept_param_keeps_checkpoint_dims':('model.parameter_representation_relation','MPLinearKernel变换先浅复制dim标签，实际权重变为2x3来源的转置，而保留参数的checkpoint input/output维仍为1/0。'),
'copy_weight_prefault_copies_exact_bytes':('model.parameter_representation_relation','模拟集成GPU资格、实际CPU copy：scalar/empty/BF16/FP8/narrow/单元素stride输入复制后与连续clone逐字节相同。'),
'copy_weight_prefault_maps_the_mmap_view':('resource.shared_region_population','模拟集成GPU资格，真实safetensors mmap输入和meta输出，通过/proc/self/smaps断言copy前驻留量不足、预触达后覆盖src全部字节；未测真实设备传输。'),
'refill_derived':('model.parameter_binding','已有同形状派生张量原地刷新，改变形状或原来缺失时新绑定；返回身份与属性相同且值正确，始终不注册parameter。'),
'mxfp4_lora_experts_keep_mxfp4_constants':('model.parameter_binding','绕过专家构造器仅测试keep_constants：继承MXFP4常量名及bias index，连续两个专家对象引用层中同一组张量；未运行MoE。')}
add(p,impl,[('test_'+n,u,d) for n,(u,d) in names.items()])
# Additional production paths read while checking the actual decoder/capture boundaries.
extra={
 'tests/kernels/mhc/test_mhc_kernels.py':['vllm/models/deepseek_v41/amd/mhc.py:52','vllm/models/deepseek_v41/amd/model.py:314','vllm/models/deepseek_v41/nvidia/model.py:500','vllm/models/deepseek_v41/nvidia/dspark.py:215','vllm/models/deepseek_v4/nvidia/model.py:716','vllm/models/deepseek_v4/nvidia/model.py:1220','vllm/models/deepseek_v41/nvidia/ops/mega_mhc.py:47','vllm/model_executor/kernels/mhc/triton.py:167','vllm/_aiter_ops.py:5080'],
 'tests/model_executor/layers/test_fused_shared_expert.py':['vllm/model_executor/layers/fused_moe/layer.py:77'],
 'tests/model_executor/test_routed_experts_capture.py':['vllm/model_executor/layers/fused_moe/router/base_router.py:282','vllm/config/vllm.py:3359']}
for key,(u,d,im) in list(S.items()):
 if key[0] in extra:S[key]=(u,d,im+extra[key[0]])

def defs(tree):
 out={}
 for x in tree.body:
  if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)):out[x.name]=x
  if isinstance(x,ast.ClassDef):
   for y in x.body:
    if isinstance(y,ast.FunctionDef):out[x.name+'::'+y.name]=y
 return out
idx=list(csv.DictReader((P/'full_delivery/all-entries.tsv').open(),delimiter='\t'))
skip=set()
for rd in ['round12_work','round13_work','round14_work']:
 for f in (P/rd).rglob('crosswalk.tsv'):
  if f.parent==OUT:continue
  skip.update((r['snapshot'],r['path'],r['test_name'],r['line']) for r in csv.DictReader(f.open(),delimiter='\t'))
rows=[];ev=[];reviews=[]
for path in sorted({p for p,n in S}):
 txt=(SRC/path).read_text();tree=ast.parse(txt);dd=defs(tree)
 reviews.append({'snapshot':'vllm','path':path,'source_sha256':hashlib.sha256((SRC/path).read_bytes()).hexdigest(),'review_scope':'仅本批显式列出的测试入口及其调用helper，非整文件无条件确认'})
 for r in idx:
  key=(r['snapshot'],r['path'],r['test_name'],r['line'])
  if r['snapshot']!='vllm' or r['path']!=path or r['review_status']=='property_reviewed' or key in skip or (path,r['test_name']) not in S:continue
  fn=dd[r['test_name']];unit,desc,impl=S[path,r['test_name']]
  helpers=[];seen=set()
  def scan(f):
   for a in ast.walk(f):
    if isinstance(a,ast.Name) and a.id in dd and a.id not in seen and not a.id.startswith('test_'):
     seen.add(a.id);helpers.append(dd[a.id]);scan(dd[a.id])
  scan(fn)
  # Include actual fixture setup and supporting fake/recording classes.
  def is_fixture(f):
   return any('fixture' in ast.unparse(d) for d in f.decorator_list)
  fixtures=[f for f in dd.values() if is_fixture(f)]
  needed={a.arg for a in fn.args.args+fn.args.kwonlyargs}
  for f in fixtures:
   if any('autouse=True' in ast.unparse(d) for d in f.decorator_list):needed.add(f.name)
  added=True
  while added:
   added=False
   for f in fixtures:
    if f.name in needed and f.name not in seen:
     seen.add(f.name);helpers.append(f);scan(f)
     needed.update(a.arg for a in f.args.args+f.args.kwonlyargs);added=True
  owner=next((c for c in tree.body if isinstance(c,ast.ClassDef) and r['test_name'].startswith(c.name+'::')),None)
  if owner is not None:
   for setup in owner.body:
    if isinstance(setup,ast.FunctionDef) and setup.name in ['setup_method','teardown_method','setup_class','teardown_class']:
     helpers.append(setup);scan(setup)
  checks=[]
  for f in [fn]+helpers:
   for a in ast.walk(f):
    if isinstance(a,ast.Assert) or (isinstance(a,ast.Call) and (ast.unparse(a.func).split('.')[-1].startswith('assert_') or ast.unparse(a.func).endswith('.raises'))):checks.append({'source':f'{path}:{a.lineno}','expression':ast.unparse(a)})
  assert checks
  cond={'test_range':f'{path}:{fn.lineno}-{fn.end_lineno}','module_conditions':[ast.unparse(x) for x in tree.body if isinstance(x,(ast.Assign,ast.AnnAssign))],'decorators':[ast.unparse(x) for x in fn.decorator_list],'class_decorators':[] if owner is None else [ast.unparse(x) for x in owner.decorator_list],'module_guards':[ast.unparse(x) for x in tree.body if isinstance(x,ast.If)],'supporting_classes':[{'source':f'{path}:{c.lineno}','body':ast.get_source_segment(txt,c)} for c in tree.body if isinstance(c,ast.ClassDef) and not c.name.startswith('Test')],'input_and_call_source':ast.get_source_segment(txt,fn),'helpers':[{'source':f'{path}:{h.lineno}','body':ast.get_source_segment(txt,h)} for h in helpers],'implementation_anchors':impl,'external_helper_anchors':[a for a in impl if a.startswith('tests/')],'external_helper_sources':[{'path':ap,'body':(SRC/ap).read_text()} for ap in sorted({a.rsplit(':',1)[0] for a in impl if a.startswith('tests/')})],'validation':'仅静态源码核对，未导入或执行上游，未使用GPU'}
  row={k:r[k] for k in ['snapshot','path','test_name','line']};row.update(unit_id=unit,property=desc+' 实际断言：'+'；'.join(x['expression'] for x in checks),oracle_source=';'.join(dict.fromkeys(x['source'] for x in checks)),conditions=json.dumps(cond,ensure_ascii=False),rule_id='round14_compute_source_'+Path(path).stem)
  rows.append(row);ev.append({**{k:row[k] for k in ['snapshot','path','test_name','line','unit_id']},'review':desc,'assertions':checks,'conditions':cond})
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
impl_paths=sorted({x.rsplit(':',1)[0] for _,_,im in S.values() for x in im})
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':reviews,'entry_reviews':ev,'supporting_source_reviews':[{'snapshot':'vllm','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()} for p in impl_paths],'confirmed_rows':len(rows),'deferred':DEFERRED},ensure_ascii=False,indent=2))
(OUT/'notes.md').write_text(f'确认{len(rows)}条。逐项阅读输入、调用和断言。每条保留完整源码参数及helper，数值参考、同核变形、分派、mock边界分开描述。未执行上游或GPU。新增3项框架无关性质（当前证据仅vLLM），详见proposed-units.json；其余使用现有性质。\n')
print('confirmed',len(rows))
