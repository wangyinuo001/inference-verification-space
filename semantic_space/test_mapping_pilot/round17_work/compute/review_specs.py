import ast,csv,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=OUT.parents[1];ROOT=P.parents[3];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'
S={};DEFERRED=[];PROPOSED=[]
D='compute.kernel_dispatch_binding';B='compute.kernel_call_binding';N='compute.operator_reference_relation';E='compute.execution_relation';O='compute.output_constraints';L='model.parameter_storage_layout';M='model.parameter_binding';R='model.parameter_representation_relation';C='deployment.effective_configuration';I='deployment.configuration_admissibility';K='compute.kernel_plan_consistency'
def add(path,impl,items):
 for name,unit,desc in items:S[path,name]=(unit,desc,impl)
F='compute.fusion_pattern_binding';W='compute.fusion_rewrite_structure'
b='vllm/model_executor/models/transformers/';fx=b+'fx_utils.py';fu=b+'fuser.py';fs=b+'fusers/'
PROPOSED += [
{'id':F,'name':'融合候选与计算数据流对应','kind':'obligation','phase':'model_execution','state_scope':'deployment','conditions':['kernel_selection_profile','execution_layout','numeric_format'],'property':'融合候选由原计算数据流、模块结构及每实例形状决定；识别出的投影、路由、共享分支与归一化参数对应原计算，未识别或不满足约束的结构按声明规则拒绝或回退。','observable':['matched_candidate','projection_names','router_metadata','normalization_parameters','eligibility','fallback'],'oracle':'构造具有明确连接关系、尺寸及模块状态的计算图，逐项对照匹配字段和接受/拒绝结果；仅结构断言不视为融合后数值等价证据。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':fs+'qkv.py','line':83,'kind':'source'},{'path':fs+'rms_norm.py','line':161,'kind':'source'},{'path':'tests/models/transformers/fusers/test_moe.py','line':418,'kind':'existing_test'}]}},
{'id':W,'name':'融合改写结构与适用约束','kind':'obligation','phase':'model_execution','state_scope':'deployment','conditions':['kernel_selection_profile','execution_layout'],'property':'融合改写将被选计算替换为声明的合并调用和切分表达式，保留未消费的控制流与名称语义；仅在已证明条件下折叠存在性判断，输入重绑定、别名修改或不安全引用使改写拒绝。','observable':['rewritten_ast','call_names','split_expression','preserved_control_flow','rejection'],'oracle':'比较改写前后AST、调用名称及常量表达式，并对重绑定、原地写入、共享视图或非存在性引用验证拒绝；数值等价由单独执行关系测试验证。','framework_scope':['vllm'],'evidence':{'vllm':[{'path':fx,'line':409,'kind':'source'},{'path':fs+'base.py','line':189,'kind':'source'},{'path':'tests/models/transformers/fusers/test_linear.py','line':1005,'kind':'existing_test'}]}}
]
p='tests/models/transformers/fusers/test_rms_norm.py';rn=fs+'rms_norm.py'
add(p,[fu+':54',fx+':249',rn+':161',rn+':223'],[
('test_detects_rms_norm_variants',F,'meta上十种显式RMS形态（rsqrt/pow/weightless/Gemma/torch fused）均匹配RMSNormFuser且zero_centered只对Gemma为真。'),
('test_non_rms_norms_are_not_matched',F,'meta上的减均值NotAnRMSNorm、nn.LayerNorm、SiLU三种结构不匹配RMS融合。'),
('test_gated_rms_norm_is_not_fused',F,'显式gate输入、fused RMS后额外SiLU乘和不可追踪gate尾三种结构全部拒绝，不把部分RMS子图当整个模块。')])
add(p,[fu+':54',rn+':223',rn+':278','vllm/model_executor/layers/layernorm.py:48','vllm/model_executor/layers/layernorm.py:155','tests/conftest.py:254'],[
('test_rms_norm_builds_vllm_class',F,'meta七种RMS模块融合后选择RMS/GemmaRMS与TPAware mixin，epsilon继承实例，weight有无及长度16/0对应原模块；无数值执行。'),
('test_fused_rms_norm_op_default_eps',F,'torch.nn.RMSNorm(16,eps=None)匹配fused路径，FP32配置构造结果epsilon等于finfo(FP32).eps；meta不执行。'),
('test_eps_is_derived_per_instance',F,'同一RMS类型两实例eps1e-5/1e-6共用结构候选，融合对象分别保留各自epsilon。'),
('test_eps_attr_is_found_by_value_not_name',F,'epsilon属性命名eps/tolerance×值1e-5/1e-6，匹配器发现实际属性并按各实例读值。'),
('test_literal_eps_is_not_mistaken_for_an_attribute',F,'forward字面量1e-4与无关lookalike属性同值时，不绑定属性eps_attr=None，融合epsilon保持字面量且记录提示。'),
('test_ambiguous_eps_attrs_are_disambiguated',F,'两个属性同为1e-6时通过标记重trace识别variance_epsilon，原对象属性完全恢复；另一实例decoy1e-3/实际1e-5仍读真实字段。')])
add(p,[rn+':118',rn+':278','vllm/model_executor/layers/layernorm.py:79','vllm/ir/ops/layernorm.py:10','tests/conftest.py:254'],[
('test_pow_spelled_norm_fuses_to_equivalent_math',E,'四种pow(-.5)拼写：seed0随机16权重复制到融合RMS，FP32随机4x16输入，融合输出与原测试模块pow均方归一化比较rtol1e-5/atol1e-6。'),
('test_weightless_norm_has_no_hidden_size',E,'无weight RMS声明宽72融合hidden_size0；手动tp_size2、无真实进程组，随机4x72输入输出与原无权重RMS一致，不触发gather。'),
('test_fused_norm_is_gather_capable',E,'RMS/Gemma两类TPAware包装器复制相同随机16权重，在完整宽4x16输入下输出与对应原生类相等；未检查跨rank gather执行。'),
('test_gathered_norm_rejects_uneven_sharding','compute.input_admissibility','TPAwareRMS权重宽8、手动TP2、输入2x3不满足3*2=8，通信前ValueError。')])
p='tests/models/transformers/fusers/test_linear.py'
add(p,[fu+':54',fs+'glu.py:122',fs+'glu.py:136',fx+':328'],[
('test_detects_and_rewrites_glu',E,'标准/交换乘法GLU×bias开关，真实fuser改写；测试nn.Linear拼接gate/up权重和SiluAndMulStub替换算子，4x16输入前后rtol/atol1e-5一致且保留模块身份；不验证分布式融合kernel。'),
('test_glu_fuses_alongside_unrelated_projections',E,'额外router投影不阻止gate/up融合；相同权重及4x16输入在nn.Linear/SiLU stub边界下改写前后两个输出一致。'),
('test_untraceable_tail_still_fuses',E,'GLU之后含数据依赖if的部分trace仍融合；stub合并linear/SiLU下原Python尾保留，4x16输出前后rtol/atol1e-5一致。'),
('test_glu_identifies_down_projection',F,'meta标准/反序GLU识别down_proj，无down的GLU返回down_name=None；未构造行并行替换。'),
('test_non_matching_modules_return_none',F,'仅串联两个Linear及GLU前含不可trace数据分支两种模块返回无GLU候选。'),
('test_unfusable_modules_are_not_fused',F,'串联MLP、Dropout伪激活、gate输出1/up输出32广播三种情况无候选或validate=False，不允许错误AndMul等半宽。')])
add(p,[fu+':54',fs+'qkv.py:83',fs+'qkv.py:100',fs+'base.py:189',fx+':249',fx+':409',fx+':654'],[
('test_detects_and_rewrites_qkv',E,'meta标准/逆序attention与KV4/2识别q和k/v集合（逆序MHA跳过）；数值执行统一KV4/layer_idx3、随机1x5x32，测试合并nn.Linear+q+2k+3v注意力替身下改写前后1e-5一致，保留kwargs/cache分支/实例属性。'),
('test_fuses_gemma4_qkv_obstacles',E,'四种存在性guard/分支分隔QKV结构融合后不再引用原q/k/v；实际运行用nn.Linear与q+2k+3v替身、KV4/layer_idx3/1x5x32，前后1e-5一致，live KV共享分支保留。'),
('test_qkv_identifies_output_projection',F,'meta六种attention结构含额外sink、全宽/每head norm及dropout，匹配输出投影均为o_proj；只检查数据流识别。'),
('test_qkv_refuses_unsound_rewrites',W,'六种输入重绑定、原地写、下标写、view别名写、functional inplace及isinstance非存在性引用，get_fuser经改写检查返回None；不以trace时死分支豁免。'),
('test_fuses_real_gemma4_attention',W,'依赖外部Transformers Gemma4可导入；meta hidden64/head4/KV2/head16/layers2的层0/1匹配q/k/v/o，改写代码不保留原q/k/v名称；仅结构检查、不执行数值。')])
add(p,[fx+':409',fx+':388',fx+':691'],[
('test_bypass_existence_guard_folds_identity_none_check',W,'构造三种self.v_proj is/not None（含None左侧）三元表达式AST，改写test为指定True/False常量。'),
('test_bypass_existence_guard_folds_bare_truthiness',W,'三元裸truthiness及not表达式AST，真实改写移除v_proj引用并产生True常量。'),
('test_bypass_existence_guard_folds_bare_if_statement',W,'if语句的self.v_proj条件AST折叠为True常量。'),
('test_bypass_existence_guard_refuses_non_guard_reference',W,'isinstance、weight属性、and/or可逃逸值以及==/!=None六种AST引用被ValueError拒绝，不能按对象存在性直接折叠。')])
add(p,[fs+'merged_column.py:54',fs+'merged_column.py:91',fs+'base.py:189',fx+':563',fx+':654'],[
('test_merged_column_fuser_folds_existence_guard',W,'四并列Linear其中proj_d有is-not-None guard，改写名称出现唯一merged_name并移除四原投影名；不执行输出。'),
('test_merged_column_fuser_rejects_input_mutation',W,'四种直接add_、view别名mul_、F.relu inplace、torch.relu_写入在投影之间发生，match先成功但update_forward以rebound or mutated拒绝。'),
('test_merged_name_is_unique_per_linear_names',M,'投影名元组(a_b,c)/(a,b_c)编码出不同合并名；相同元组重复实例同名，避免权重映射名称歧义。')])
add(p,[fu+':113',fs+'qkv.py:152',fs+'merged_column.py:113'],[
('test_heterogeneous_instances_fall_back_to_merged_column_fuser',F,'同PlainQKV类与children缓存两候选，output32/head8实例选QKV；同结构但head5不整除32实例回退MergedColumn候选。')])
add(p,[fu+':54',fs+'packed_qkv.py:67',fs+'packed_qkv.py:115'],[
('test_detects_and_rewrites_packed_qkv',E,'packed QKV KVhead1/2、q32/kv8或16识别后改写split读取output_sizes/tp_size；TP1手工尺寸替身和FakeMQA注意力、随机1x5x32输入前后1e-5一致；未测试TP分片值。'),
('test_per_head_split_is_not_packed_qkv',F,'每head拆分8/8/8而非整投影96输出的结构不匹配packed QKV，防止错误分片尺寸。')])
add(p,[fs+'base.py:155','vllm/model_executor/models/utils.py:58',fs+'glu.py:50',fs+'qkv.py:37'],[
('test_weight_mappings_are_scoped_to_fused_prefixes',M,'两个明确MLP prefix和一个attention prefix映射gate/up及q/k/v到合并名并附0/1/q/k/v shard；其他expert及其他层同名不改；packed_modules_mapping保留原投影名列表。')])
add(p,[fs+'glu.py:104',fs+'glu.py:115',b+'layers.py:19','vllm/model_executor/layers/activation.py:845'],[
('test_act_and_mul_derived_from_module',D,'SiLU两类选SiluAndMul，GELUTanh/nn.GELU分别选tanh/none近似GeluAndMul；Dropout/LayerNorm无名字映射且Dropout构造ValueError。未前向。')])
add(p,[fs+'base.py:72',fs+'qkv.py:152',fs+'qkv.py:169',fs+'packed_qkv.py:124',fs+'packed_qkv.py:137'],[
('test_head_counts_come_from_the_module_not_the_model',B,'QKV/packed两类meta模块head_dim8、heads8/KV4，伪global head16；QKVParallelLinear与row替换为记录型Identity替身，update_attrs传实际head_size8/heads8/KV4。'),
('test_validate_accepts_a_layer_the_model_wide_head_size_would_reject',F,'meta QKV q64/kv24与层head8可整除，伪global head16不整除kv但validate仍接受本层尺寸。')])
DEFERRED += [{'path':p,'tests':['test_fuser_is_cached_per_class_and_structure'],'reason':'纯候选缓存对象身份不冒充计算关系'}, {'path':p,'tests':['test_merged_column_fuser_supports_any_number_of_linears'],'reason':'共享自动权重加载与真实merged forward链路尚未全部完成，保留待审'}]
p='tests/models/transformers/fusers/test_moe.py'
add(p,[fs+'moe.py:208',fs+'moe.py:255',fs+'moe.py:291',fs+'moe.py:313',fx+':249',b+'utils.py:336','tests/models/transformers/fusers/test_linear.py:51'],[('test_'+name,F,desc+' 全部meta模块，仅检查融合计划；MoEExperts.forward为输入直通，不证明路由数值或专家计算。') for name,desc in [
('moe_fuser_detects_router','8专家/hidden16/top2，linear→softmax或sigmoid→topk归一化图识别gate及评分名，没有shared。'),
('moe_fuser_matches_router_scoring_the_selection','先topk logits再softmax/sigmoid的选择后评分图识别FP32 router，renormalize仅softmax=True。'),
('moe_fuser_matches_scaled_router','topk权重乘16的路由图仍匹配候选；未断言比例字段。'),
('moe_fuser_reads_renormalize_from_the_gate','显式sum除法/带1e-20分母偏移返回True；乘16完整图返回False；trace在data-dependent分支中断则None。'),
('moe_fuser_matches_grouped_router','先4组x2专家的group top2/mask，再routing top2图识别最后topk关联softmax评分。'),
('moe_fuser_matches_correction_router','sigmoid评分加e_score_correction_bias缓冲后top2图可匹配，scoring_func=sigmoid。'),
('moe_fuser_reads_router_dtype_from_the_gate','linear输入与权重显式FP32转换的router记录FP32，普通router为None，不误读topk后的回转dtype。'),
('moe_fuser_detects_shared_experts','GLU shared加单logit sigmoid gate图识别shared_expert及shared_expert_gate。'),
('moe_fuser_shared_without_gate','没有门控的GLU shared分支识别shared_expert且shared_gate=None。'),
('moe_fuser_detects_non_glu_shared_expert','非GLU的down(SiLU(up))共享分支也按数据流识别，并无shared_gate。'),
('moe_fuser_declines_unsupported','九种额外router参数、元组返回/变量元组、未覆盖weight/buffer子模块、专家多传scale、非shared附加项、平均输出、gate多传bias均拒绝候选。'),
('moe_fuser_ignores_nested_returns','嵌套helper返回tuple但block本身返回单tensor时仍匹配，不误读嵌套作用域。'),
('moe_fuser_router_requires_connected_dataflow','linear/softmax存在但topk直接读原输入、数据流不连接时拒绝。'),
('moe_fuser_unrecognized_router_moves_shared_expert','含额外bias的未知router保留Transformers路由，仍识别共享分支，scoring_func=None。'),
('moe_fuser_divided_shared_output','divides_shared_output=True时有shared分支的模块评分回退None；无shared基础block仍softmax。'),
('moe_fuser_scaled_output_needs_vllm_routing','未知带bias路由且专家输出乘2.5时拒绝，不能搬迁需要融合路由承担的scale。'),
('moe_fuser_matches_experts_called_by_keyword','专家以weights/index关键字传参仍按签名识别连接关系。'),
('moe_fuser_scaled_output_with_vllm_routing','已识别softmax路由的专家输出乘2.5+shared图接受，并识别shared_expert。'),
('moe_fuser_looks_through_casts_and_identity','FP32 combine再转回和identity latent前后包装两类图均识别shared_expert。')]])
# Shared production call path inspected for the numerical attention stand-ins.
for name in ['test_detects_and_rewrites_qkv','test_fuses_gemma4_qkv_obstacles','test_detects_and_rewrites_packed_qkv']:
 key=('tests/models/transformers/fusers/test_linear.py',name)
 unit,desc,impl=S[key]
 S[key]=(unit,desc,impl+['vllm/model_executor/models/transformers/__init__.py:68'])
for key,(unit,desc,impl) in list(S.items()):
 if key[0].endswith('test_rms_norm.py'):
  S[key]=(unit,desc,impl+['vllm/model_executor/hw_agnostic/_resolve.py:31'])
