import ast,csv,json,hashlib,re
from pathlib import Path
P=Path(__file__).resolve().parents[4]; ROOT=P.parents[1];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).parent
index=list(csv.DictReader((P/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
existing_keys=set()
if (OUT/'crosswalk.tsv').exists():
 existing_keys={(r['snapshot'],r['path'],r['test_name'],r['line']) for r in csv.DictReader((OUT/'crosswalk.tsv').open(),delimiter='\t')}
SPEC={}
def spec(path,product,default_unit,description,overrides=None,exclude=None,helpers=None):
 SPEC[path]={'product':product,'default':default_unit,'description':description,'overrides':overrides or {},'exclude':exclude or {},'helpers':helpers or []}
spec('tests/kernels/test_bf16x3_router_gemm_rocm.py','vllm/model_executor/layers/fused_moe/router/bf16x3_router_gemm_rocm.py','compute.operator_reference_relation','BF16x3路由GEMM与FP64矩阵乘法比较，检查本项实际断言的相对L2误差、top-k专家集合、输出类型或指定缓冲区。',{
 'test_split_reconstructs_exactly':('model.parameter_representation_relation','FP32权重拆成三个BF16项后求和精确恢复原值，同时检查形状、类型、连续性。'),
 'test_split_is_exact_across_normal_exponents':('model.parameter_representation_relation','正负尾数乘2的-100至99次方构成FP32输入，BF16三项求和逐元素等于输入。'),
 'test_split_does_not_mutate_input':('model.parameter_representation_relation','BF16三项拆分不改写来源权重。'),
 'test_split_rejects_non_finite':('compute.input_admissibility','拆分拒绝inf、-inf和NaN权重。'),
 'test_split_rejects_non_fp32':('compute.input_admissibility','拆分拒绝非FP32权重。'),
 'test_validates_layout_assumptions':('compute.input_admissibility','GEMM拒绝拆分项数、K维、类型或连续性不符合契约的输入与输出。'),
 'test_is_supported_rejects_ineligible_inputs':('compute.kernel_dispatch_binding','资格谓词按token数、输入/权重类型、K维及激活步幅判断支持。'),
 'test_is_supported_rejects_mismatched_devices':('compute.kernel_dispatch_binding','输入与权重不在同一设备时资格谓词拒绝。'),
 'test_is_supported_accepts_odd_expert_counts':('compute.kernel_dispatch_binding','资格谓词接受给定非二次幂专家数。'),
 'test_writes_into_supplied_out':('compute.execution_relation','显式输出缓冲区被原样返回，其结果逐元素等于默认输出分配路径。'),
})
spec('tests/kernels/ir/test_layernorm.py','vllm/ir/ops/layernorm.py','compute.operator_reference_relation','RMSNorm或残差融合RMSNorm的实现输出与native参考比较；输入克隆防止原地更新污染另一侧，误差采用IR算子get_tolerance。',{
 'test_rms_norm_registration':('compute.kernel_dispatch_binding','平台对应的RMSNorm实现支持标志与预期字典一致。'),
 'test_fused_add_rms_norm_registration':('compute.kernel_dispatch_binding','平台对应的残差融合RMSNorm实现支持标志与预期字典一致。'),
 'test_aiter_rejects_unsupported_dtypes':('compute.kernel_dispatch_binding','AITER支持谓词拒绝float32及float64输入。'),
 'TestRMSNorm::test_native_semantics':('compute.execution_relation','RMSNorm输出形状/类型/设备、缩放前后近似关系、去权重后的均方以及None权重和全1权重关系。'),
 'TestFusedAddRMSNorm::test_inplace_semantics':('compute.execution_relation','default与maybe_inplace分别满足输入保留或别名写回，并在相同输入下产生逐元素相同结果。'),
},exclude={'TestRMSNorm::test_torch_opcheck':'opcheck检查自定义算子注册/编译契约，现有数值关系单元不能完整替代该性质。','TestFusedAddRMSNorm::test_torch_opcheck':'同上，包含alias语义与编译契约。'},helpers=['tests/ir/ir_test_utils.py'])
spec('tests/model_executor/layers/test_rocm_unquantized_gemm.py','vllm/model_executor/layers/utils.py','compute.kernel_dispatch_binding','ROCm线性包装器在本项模拟平台和布局下选择或跳过指定内核，并对照F.linear检查返回结果；带real_kernel的三项实际调用设备内核，其他分派项使用明确的MagicMock替身。',{
 'test_rocm_unquantized_gemm_single_output_real_kernel':('compute.operator_reference_relation','ROCm单输出通道GEMM与F.linear在fp16/bf16、1至5行输入下比较。'),
 'test_rocm_unquantized_gemm_noncontiguous_activation_real_kernel':('compute.operator_reference_relation','非连续激活的真实wvSplitK调用与F.linear比较。'),
 'test_rocm_unquantized_gemm_padded_weight_real_kernel':('compute.operator_reference_relation','gfx11补齐权重行步幅的真实wvSplitK调用与F.linear比较。'),
 'test_cache_aliasing_pad_elems':('model.parameter_storage_layout','根据权重行元素数和元素字节数，确定避开缓存别名所需的padding元素数。'),
 'test_maybe_pad_weight_avoid_cache_aliasing':('model.parameter_representation_relation','gfx11权重padding保留形状和值，改变行步幅；重复padding保留缓冲区身份。'),
 'test_maybe_pad_weight_avoid_cache_aliasing_skips_non_gfx11':('model.parameter_storage_layout','非gfx11平台保留权重对象身份。'),
 'test_maybe_pad_weight_avoid_cache_aliasing_skips':('model.parameter_storage_layout','无需padding、不连续或非2D输入保留权重存储身份。'),
})
spec('tests/kernels/attention/test_pack_unpack_triton.py','vllm/v1/attention/ops/common.py','compute.operator_reference_relation','序列pack依据lengths将源张量切片对应到批内有效位置；unpack按同一lengths恢复已打包有效输入。',{
 'test_pack_seq_custom_padding_fp8':('compute.operator_reference_relation','比较两个长度10序列的有效张量值；result[:,10:]为空，padding断言不构成非空padding性质证据。'),
 'test_pack_seq_edge_cases_fp8':('compute.output_constraints','只核对单批、短序列和不同长度下pack输出形状。'),
 'test_pack_seq_shape_consistency':('compute.output_constraints','只核对pack批数、最大长度和特征维度。'),
},exclude={'test_pack_seq_default_negative_inf_padding_fp8':'lengths=[10,10]，结果最大长度10，result[:,10:]为空；torch.all在空张量上恒真，不能确认负无穷padding性质。'})
spec('tests/kernels/attention/test_triton_mla_causal_verify_flatten.py','vllm/v1/attention/backends/mla/triton_mla.py','compute.kernel_call_binding','MLA包装器将请求展开为逐query行并向spy内核提交block_table和seq_lens；因果行只见前缀与本行之前token，非因果行提交完整长度。本项检查提交元数据，不执行注意力数值计算。',{
 'test_multi_token_decode_flags_are_declared_together':('compute.kernel_dispatch_binding','元数据构建器同时声明UNIFORM_BATCH图支持和UNIFORM query长度支持。'),
 'test_non_uniform_block_is_rejected':('compute.input_admissibility','query行数不能均分到请求时拒绝。'),
 'test_causal_multi_token_decode_is_rejected_under_dcp':('compute.input_admissibility','DCP world size2拒绝因果多token decode；同配置非因果路径必须到达spy内核。'),
})
spec('tests/kernels/test_gate_linear_rocm_dispatch.py','vllm/model_executor/layers/fused_moe/router/gate_linear.py','compute.kernel_dispatch_binding','模拟ROCm架构、bias、权重/输出精度及并行规模后，GateLinear资格标志与指定实现支持范围一致；本项不执行实际GEMM。',{
 'test_rocm_bf16x3_dispatch_is_a_custom_op':('compute.output_constraints','meta输入调用自定义dispatch算子，检查输出shape=(8192,128)和float32类型，不检查数值。'),
 'test_rocm_bf16x3_no_split_built_when_tier_disabled':('model.parameter_storage_layout','BF16x3分派未启用时，process_weights_after_loading不创建拆分权重缓冲区。'),
},exclude={'test_rocm_bf16x3_is_reachable_for_low_m_shapes':'通过inspect.getsource检查两个字符串的位置先后，属于源码结构约束，不直接证明运行路径可达。'})
spec('tests/kernels/attention/test_use_trtllm_attention.py','vllm/utils/flashinfer.py','compute.kernel_dispatch_binding','TRTLLM资格与分派按模拟设备能力、工件可用性、head整除关系、DCP规模、prefill/decode、KV/query精度和显式开关返回声明布尔值。')


spec('tests/quantization/test_quantization_config_args.py','vllm/config/quantization.py','deployment.effective_configuration','量化配置解析检查用户字符串、显式字段、checkpoint类型与默认值组合后的权重/激活QuantKey、linear/moe槽位及旋转配置；不执行量化内核。',{
 'test_quant_spec_rejects_unknown_name':('deployment.configuration_admissibility','拒绝未定义量化格式名称。'),
 'test_args_string_shorthand_missing_slot_raises':('deployment.configuration_admissibility','量化简写未定义linear槽位时拒绝，而非静默忽略。'),
 'test_targets_reject_non_string_keys':('deployment.configuration_admissibility','targets键必须为字符串。'),
 'test_targets_reject_overlapping_patterns':('deployment.configuration_admissibility','同一层匹配多个量化target时拒绝。'),
 'test_targets_reject_partially_matched_fused_layer':('deployment.configuration_admissibility','融合QKV仅部分分片匹配量化target时拒绝。'),
 'test_targets_reject_fused_shards_with_different_schemes':('deployment.configuration_admissibility','融合分片匹配不同量化方案时拒绝。'),
 'test_targets_reject_moe_only_shorthand_for_linear_layer':('deployment.configuration_admissibility','仅定义MoE槽位的量化方案不能指定给LinearBase。'),
 'test_hadamard_size_must_be_power_of_two':('deployment.configuration_admissibility','Hadamard分块大小非正或非2次幂时拒绝。'),
 'test_hadamard_size_not_implemented_for_unvalidated_methods':('deployment.configuration_admissibility','FP8 linear及MXFP4 MoE的未支持Hadamard组合明确拒绝。'),
 'test_mxfp4_hadamard32_target_resolves_to_rotated_mxfp4_linear':('compute.kernel_dispatch_binding','指定层名匹配旋转MXFP4 target后，解析为Mxfp4OnlineLinearMethod并保留旋转尺寸32。'),
},exclude={'test_quant_spec_string_representation_uses_quantization_name':'仅检查配置字符串格式，暂不归为系统执行配置性质。','test_quant_spec_equality_ignores_explicit_field_metadata':'只检查数据对象相等性忽略显式字段元数据，不保证部署解析过程。','test_rocm_platform_supports_mxfp4_hadamard32':'隐式无异常检查需独立记录，本批生成器不把缺少断言当作数值或分派检查。'},helpers=['tests/quantization/utils.py'])
spec('tests/quantization/test_config_utils.py','vllm/model_executor/layers/quantization/utils/config_utils.py','deployment.effective_configuration','量化目标与ignore配置按具体层名、正则和显式fnmatch解析；融合层优先直接匹配，否则分别匹配各分片。检查实际匹配集合、解析一致性或指定拒绝。',{
 'test_online_ignore_supports_fnmatch_patterns':('deployment.configuration_admissibility','同一层同时属于量化target和ignore时拒绝。'),
 'test_should_ignore_layer_rejects_partially_matched_fused_layer':('deployment.configuration_admissibility','融合QKV各分片的ignore结果不一致时拒绝。'),
},exclude={'test_get_layer_name_after_index':'字符串后缀提取工具，不单独宣称已完成层到量化配置绑定。'},helpers=['tests/quantization/utils.py'])
spec('tests/model_executor/model_loader/test_stacked_shard_tracking.py','vllm/model_executor/layers/linear.py','model.parameter_binding','分别加载checkpoint分片后检查缺失分片清单；每参数独立记录，reload清空旧记录，零行分片不要求加载，打包分片对应正确目标。',{
 'test_unknown_shard_id_is_rejected':('compute.input_admissibility','显式拒绝不存在的QKV分片标识。'),
 'test_loader_lists_missing_shards_as_weights_not_loaded':('model.parameter_binding','模型loader将尚未加载的K/V分片列入未初始化权重错误。'),
})
spec('tests/kernels/quantization/test_mxfp4_kernel_selection.py','vllm/model_executor/kernels/linear/mxfp4/emulation.py','compute.kernel_dispatch_binding','MXFP4实现按激活量化格式、模拟硬件和注册支持状态选择或拒绝；本项实际检查支持标志、原因或解析出的实现。',{
 'test_emulation_kernel_derives_quant_dequant_func_from_config':('compute.kernel_dispatch_binding','weight-only选择恒等激活处理；MXFP4动态激活选择指定QDQ函数对象。'),
 'test_emulation_kernel_dequantizes_at_load_and_keeps_activation_qdq':('compute.kernel_call_binding','以固定反量化替身和乘0.5激活替身检查加载时只反量化一次、调用时激活处理一次，返回结果与F.linear相同且保留权重dtype/device/梯度标志。替身不能验证真实量化数值。'),
 'test_emulation_kernel_opt_out_dequantizes_per_invocation':('compute.kernel_call_binding','关闭加载时反量化后，两次执行各调用一次反量化和激活QDQ；结果与固定反量化权重的F.linear相同且原权重保持uint8。'),
 'test_init_mxfp4_linear_kernel_raises_when_no_kernel_matches':('deployment.configuration_admissibility','注册实现全部不支持当前模拟平台时报告无可用内核。'),
},exclude={'test_can_implement_is_abstract':'只检查类具有方法属性，并未检查支持范围或实际分派。'})


spec('tests/kernels/quantization/test_mxfp4_w4a8_rdna3.py','vllm/model_executor/kernels/linear/mxfp4/rdna3_hip.py','compute.kernel_dispatch_binding','gfx11 MXFP4线性实现按激活格式、行数和设备支持状态选择W4A8或高精度反量化路径；各项实际检查以本项断言为准。',{
 'test_w4a8_gemv_matches_reference':('compute.operator_reference_relation','uint8 E2M1权重与E8M0缩放独立解码为FP64，每32项激活独立量化int8后进行FP64矩阵乘法；内核逐元素误差不超过2倍输出eps乘绝对积累值加1e-3，另检查shape/dtype。'),
 'test_w4a8_gemv_close_to_bf16_dequant':('compute.operator_reference_relation','W4A8输出与不量化激活的FP64反量化矩阵乘法比较，整体相对L2误差小于1e-2。'),
 'test_w4a8_gemv_rejects_out_of_range_m':('compute.input_admissibility','行数0及超过MAX_W4A8_BATCH_SIZE时原生算子拒绝。'),
 'test_w4a8_gemv_cuda_graph':('compute.execution_relation','同一输入下CUDA Graph replay与eager输出逐元素完全相同。'),
 'test_linear_kernel_dispatch':('compute.kernel_dispatch_binding','通过wraps追踪实际GEMV调用，检查行数边界的分派；同时对照FP64反量化矩阵乘法加bias，相对L2误差小于2e-2。'),
 'test_resolve_linear_activation_key':('deployment.effective_configuration','用户非空激活精度覆盖checkpoint精度，否则沿用checkpoint。'),
 'test_resolve_linear_activation_key_explicit_null':('deployment.effective_configuration','显式activation:null覆盖checkpoint激活精度。'),
 'test_quantization_config_parses_int8_activation':('deployment.effective_configuration','int8_per_group_32解析为动态32项int8激活key且不设置weight。'),
},exclude={'test_w4a8_gemv_opcheck':'自定义算子注册/编译/alias契约，暂不混入数值一致性。'})
spec('tests/kernels/core/test_muse_spark_norms.py','vllm/models/muse_spark/common/ops/fused_rmsnorm.py','compute.operator_reference_relation','MuseSpark RMSNorm、残差门控和后续归一化在相同输入下比较实际融合实现与独立FP64参考或native/Triton执行；断言明确返回输出和残差stream的误差。',{
 'test_nothing_is_rounded_before_the_store':('compute.operator_reference_relation','FP32输出对FP64权重及归约参考，最大绝对误差除参考最大幅值小于1e-5，检查最终输出dtype。'),
 'test_folding_the_cast_in_beats_casting_around_it':('compute.operator_reference_relation','将FP32输入直接送融合归一化，其对FP64参考的相对最大误差不超过先转BF16路径。'),
 'test_out_dtype_defaults_to_the_input_dtype':('compute.output_constraints','融合和native默认输出精度均等于输入精度，分别检查FP32与BF16。'),
 'test_postnorm_gated_residual_empty_input':('compute.output_constraints','空token残差归一化返回(0,HIDDEN) FP32张量。'),
 'test_postnorm_gated_residual_cuda_graph_replays_changing_inputs':('compute.execution_relation','两组不同seed的新输入复制入图的静态缓冲区，replay输出分别对照同一当前输入的Triton路径。'),
 'test_unsupported_flashinfer_post_norm_uses_triton':('compute.kernel_dispatch_binding','禁止FlashInfer post-weight支持后，安装失败spy确保不进入该实现；Triton回退输出和stream分别对照native。'),
})
spec('tests/model_executor/model_loader/test_registry.py','vllm/model_executor/model_loader/default_loader.py','deployment.configuration_admissibility','loader注册类型、线程数、加载策略组合及显式文件格式按声明契约接受或拒绝。',{
 'test_register_model_loader':('deployment.effective_configuration','用户注册load_format选择相应自定义loader类实例。'),
 'test_default_loader_hf_still_falls_back_to_pt':('deployment.effective_configuration','HF加载模式允许回退到本地pt权重文件并标明非safetensors；只检查文件选择不反序列化。'),
 'test_weights_track_reports_missing_quantized_weights':('model.parameter_binding','量化模型缺失权重scale和bias报错，在线构造权重、KV参数和空占位等声明例外不报缺失。'),
 'test_weights_track_keeps_unquantized_models_unchanged':('model.parameter_binding','非量化加载沿用带postprocess方法层的豁免，普通norm权重缺失仍报告。'),
 'test_moe_runner_reports_registered_expert_names':('model.parameter_binding','带替身专家的runner和AutoWeightsLoader组合返回包含routed_experts层级的已加载参数全名。'),
})
spec('tests/kernels/attention/test_rocm_aiter_mla_head_padding.py','vllm/v1/attention/backends/mla/rocm_aiter_mla.py','compute.execution_relation','MLA query head padding/unpadding按重复或tile规则保留真实head值，检查形状连续性、零拷贝或往返结果；不将这些包装器关系视为注意力内核数值比较。',{
 'test_is_valid_num_heads':('compute.kernel_dispatch_binding','head计数支持1至128及更大的16倍数，拒绝0与129。'),
 'test_fp8_prefill_rejects_head_counts_padding_doubles':('compute.kernel_dispatch_binding','FP8 prefill按padding膨胀比限制head计数，decode资格分别检查。'),
 'test_nondivisor_and_multitoken_never_use_gluon':('compute.kernel_dispatch_binding','非16因子head、至少16head及多query不走Gluon单token路径。'),
 'test_divisor_gluon_selection_follows_arch':('compute.kernel_dispatch_binding','1/4/8head单token路径仅在gfx950选择Gluon。'),
 'test_asm_padding_env_default_is_auto':('deployment.effective_configuration','清除环境配置后padding模式默认为auto。'),
 'test_asm_padding_env_force_asm_disables_gluon':('compute.kernel_dispatch_binding','显式asm模式禁用所测head数的Gluon分派。'),
 'test_asm_padding_env_force_gluon_follows_arch':('compute.kernel_dispatch_binding','显式Gluon模式仍受gfx950可用性约束。'),
 'test_asm_padding_env_auto_matches_arch_gate':('compute.kernel_dispatch_binding','auto模式同时遵守架构与head整除条件。'),
 'test_h12_aiter_mla_decode_matches_reference':('compute.operator_reference_relation','12head BF16 MLA decode经padding后去除多余head，与FP32 scaled_dot_product_attention独立参考对照，atol/rtol均1e-2且输出有限。'),
})

def walk_defs(tree):
 out={}
 for n in tree.body:
  if isinstance(n,ast.FunctionDef):out[n.name]=(n,None)
  if isinstance(n,ast.ClassDef):
   for f in n.body:
    if isinstance(f,ast.FunctionDef):out[n.name+'::'+f.name]=(f,n)
 return out
rows=[];deferred=[];reviews=[]
for path,s in SPEC.items():
 text=(SRC/path).read_text();tree=ast.parse(text);defs=walk_defs(tree)
 for r in index:
  if r['snapshot']!='vllm' or r['path']!=path or (r['review_status']=='property_reviewed' and (r['snapshot'],r['path'],r['test_name'],r['line']) not in existing_keys):continue
  name=r['test_name'];pair=defs.get(name)
  if not pair:continue
  fn,cls=pair
  if name in s['exclude'] or fn.name in s['exclude']:
   deferred.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'reason':s['exclude'].get(name,s['exclude'].get(fn.name))});continue
  unit,desc=s['overrides'].get(name,s['overrides'].get(fn.name,(s['default'],s['description'])))
  checks=[]
  for a in ast.walk(fn):
   if isinstance(a,ast.Assert):checks.append((a.lineno,ast.unparse(a)))
   elif isinstance(a,ast.Call):
    n=ast.unparse(a.func).split('.')[-1]
    if n=='raises' or n.startswith('assert_'):checks.append((a.lineno,ast.unparse(a)))
  if not checks:deferred.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'reason':'无明确断言，需核对helper后再映射'});continue
  helper_names={a.func.id for a in ast.walk(fn) if isinstance(a,ast.Call) and isinstance(a.func,ast.Name)}
  helper_refs=[f'{path}:{defs[n][0].lineno}' for n in helper_names if n in defs and not n.startswith('test_')]
  helper_ranges=[f'{path}:{defs[n][0].lineno}-{defs[n][0].end_lineno}' for n in helper_names if n in defs and not n.startswith('test_')]
  checks.sort()
  source_refs=[f'{path}:{l}' for l,_ in checks]+helper_refs
  for h in s['helpers']:
   ht=ast.parse((SRC/h).read_text())
   for a in ht.body:
    if isinstance(a,ast.FunctionDef) and a.name in helper_names:source_refs.append(f'{h}:{a.lineno}')
  rows.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'unit_id':unit,'property':'验证类别：'+desc+' 本项实际检查：'+'；'.join(f'L{l}: {x}' for l,x in checks),'oracle_source':'; '.join(source_refs),'conditions':json.dumps({'source_test':f'{path}:{fn.lineno}-{fn.end_lineno}','helpers':helper_ranges,'decorators':[ast.unparse(d) for d in ((cls.decorator_list if cls else [])+fn.decorator_list)],'implementation':s['product'],'module_constants':f'{path}:1','evidence':'源码核对，未执行；仅确认当前断言范围'},ensure_ascii=False),'rule_id':'round11_compute_source_'+Path(path).stem})
 product=SRC/s['product']
 reviews.append({'snapshot':'vllm','path':path,'source_sha256':hashlib.sha256((SRC/path).read_bytes()).hexdigest(),'implementation_path':s['product'],'implementation_sha256':hashlib.sha256(product.read_bytes()).hexdigest() if product.is_file() else None,'review_method':'逐段阅读调用、构造输入、模拟替身、参考及断言，程序仅提取证据和排除已归类入口'})
fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
supplementary_paths=[
 'tests/ir/ir_test_utils.py','tests/quantization/utils.py',
 'vllm/model_executor/layers/quantization/online/base.py',
 'vllm/model_executor/layers/quantization/compressed_tensors/utils.py',
 'vllm/model_executor/layers/quantization/quark/utils.py',
 'vllm/model_executor/kernels/linear/__init__.py',
 'vllm/model_executor/kernels/linear/mxfp4/base.py',
 'vllm/model_executor/kernels/linear/mxfp4/flashinfer.py',
 'vllm/model_executor/kernels/linear/mxfp4/xpu.py',
 'vllm/model_executor/kernels/linear/mxfp4/aiter.py',
 'vllm/model_executor/kernels/linear/mxfp4/marlin.py',
 'vllm/models/muse_spark/common/layers.py',
 'vllm/models/muse_spark/nvidia/model.py',
 'vllm/model_executor/model_loader/__init__.py',
 'vllm/model_executor/layers/fused_moe/runner/moe_runner.py',
]
supplementary=[{'snapshot':'vllm','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest(),'review_scope':'仅核对本轮测试调用涉及的函数和分支，不表示整个文件逐项审查'} for p in supplementary_paths]
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':reviews,'supporting_source_reviews':supplementary,'deferred':deferred,'confirmed_rows':len(rows)},ensure_ascii=False,indent=2))
(OUT/'notes.md').write_text(f'本轮确认{len(rows)}项，暂不确认{len(deferred)}项。所有属性由源码核对产生，未执行上游测试。本批覆盖15个测试文件，含ROCm GEMM与MLA、IR RMSNorm、量化配置与分派、权重分片完整性及残差归一化。每项保留独立断言位置、输入构造范围和装饰器条件。meta输出、mock分派、实际数值对照分开标明。pack默认padding测试读取空切片，仅得到空集合上的恒真断言；不记为padding性质确认。\n')
print('confirmed',len(rows),'deferred',len(deferred))
