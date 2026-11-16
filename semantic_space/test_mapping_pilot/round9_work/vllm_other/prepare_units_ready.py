"""Candidate ontology package; no main seed mutations."""
import json,hashlib
from extract_review import HERE,SOURCE
from review_cpu_gdn_batch6 import exact_entry

G='tests/kernels/mamba/cpu/test_cpu_gdn_ops.py';S='tests/kernels/test_cpu_fused_sampling.py'
def evidence(path,tests,product,product_lines):
 refs=[{'path':product,'line':l,'kind':'source'} for l in product_lines]
 for name in tests:
  e=exact_entry(path,name);refs.append({'path':path,'line':e['definition_line'],'kind':'existing_test'})
  refs.extend({'path':path,'line':c['line'],'kind':'existing_oracle'} for c in e['checks'])
 return {'vllm':refs}

units=[{
 'id':'compute.recurrent_state_transition','name':'递归计算状态续传与选定槽位更新','kind':'obligation','phase':'model_execution','state_scope':'request',
 'conditions':['recurrent_state_profile','operator_profile','numeric_format','execution_layout'],
 'property':'GDN或因果卷积按请求的初始状态、已接受token历史和分块边界延续递归计算；状态写回只改变选定槽位与有效列，未选中槽位和额外缓存列保持原值。',
 'observable':['state_before','state_after','selected_state_indices','accepted_token_counts','untouched_storage_regions','continued_output'],
 'oracle':'对GDN直接比较显式逐token递推得到的最终状态；对两次调用检查首段状态作为次段输入后的最终状态/输出；对卷积对照Torch状态更新；对DS适配器用哨兵写入scratch后按原slot映射核对完整存储，包括未选槽位和宽尾部。各检查保留独立粒度，不能从单次输出相等推出状态不受损。',
 'execution_boundary':'输入hidden/QKV、前序递归状态、slot映射和accept计数→本步输出与可供下一步使用的状态；不含KV块分配或引用计数。',
 'framework_scope':['vllm'],
 'evidence':evidence(G,['test_fused_sigmoid_gating_delta_rule_update_cpu','test_chunk_gated_delta_rule_cpu_two_call_split','test_causal_conv1d_update_cpu_accepts_wide_state','test_causal_conv1d_update_cpu_ds_adapter_wide_multi_token_copyback','test_conv_update_cpp_matches_torch'],'vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py',[33,44,52,93,494,566]),
 'framework_difference':'此包只审查固定vLLM CPU GDN/conv实现，未核对SGLang对应路径，暂仅vllm适用。'
}, {
 'id':'sampling.distribution_relation','name':'采样频率符合目标概率分布','kind':'obligation','phase':'sampling','state_scope':'request',
 'conditions':['sampling_statistical_profile','sampling_mode','numeric_format','execution_layout'],
 'property':'在声明logits、有限支持集、随机种子集合和样本量下，采样token的经验频率在指定统计阈值内符合目标softmax概率；既检查整体分布，也检查头部、尾部或等概率极值输入。',
 'observable':['sampled_token_counts','reference_probabilities','sample_count','chi_square_p_value','head_frequency','tail_frequency','frequency_error_threshold'],
 'oracle':'100维/100000次采样按期望频数>5的桶计算卡方p>0.001；五个有限支持token的20000次样本卡方p>1e-7且屏蔽token计数0；8192维Zipf头/尾16384次频率偏差<8个标准误；两个FP32最大logit的4096次样本须含两ID且ID0频率在(0.35,0.65)。这些是有界统计检查，不证明任意logits上的分布正确。',
 'execution_boundary':'固定logits与种子集合→累计token计数/频率；与单次成员约束、逐seed重复性分开。',
 'framework_scope':['vllm'],
 'evidence':evidence(S,['TestFusedGumbelArgmax::test_distribution_chi_squared','TestFusedGumbelArgmax::test_masked_support_matches_softmax','TestFusedGumbelArgmax::test_zipf_head_matches_softmax','TestFusedGumbelArgmax::test_zipf_tail_matches_softmax','TestFusedGumbelArgmax::test_tied_huge_logits_not_index_biased'],'csrc/cpu/sampling_kernels.cpp',[28,39,144]),
 'framework_difference':'只核对vLLM CPU fused_gumbel_argmax；SGLang未审。与sampling.policy现有统计oracle有交集，可选择扩充其条件与可观察量后直接映射，或将本unit作为其统计细分，勿重复计覆盖。'
}, {
 'id':'sampling.randomness_relation','name':'声明种子变化下的样本关系','kind':'obligation','phase':'sampling','state_scope':'request',
 'conditions':['sampling_statistical_profile','sampling_mode','execution_layout'],
 'property':'对测试声明的种子对或种子序列，采样结果满足指定的差异或相关性界限；该关系只约束所声明的模式，不能解释为完整随机独立性证明。',
 'observable':['seed_vectors','sampled_token_vectors','vector_inequality','consecutive_seed_shift_fraction'],
 'oracle':'16x50000均匀logits，种子向量arange(16)与其加1000000的采样向量必须不完全相同；2048个连续seed在128维均匀logits上，后一个winner等于前一个winner减1取模的比例<0.05。',
 'execution_boundary':'同一logits下两组/连续种子的token向量→特定差异或相关性统计。',
 'framework_scope':['vllm'],
 'evidence':evidence(S,['TestFusedGumbelArgmax::test_different_seeds_differ','TestFusedGumbelArgmax::test_noise_not_sliding_window'],'csrc/cpu/sampling_kernels.cpp',[28,33,39]),
 'framework_difference':'只核对vLLM CPU SplitMix64(seed,token)路径，未核对SGLang。'
}]
dims=[{'id':'recurrent_state_profile','type':'structured','fields':['recurrence_kind','state_layout','state_dtype','initial_state_flags','slot_indices','chunk_boundaries','accepted_token_counts','active_state_columns','padding_columns','write_rounding_interval'],'observed_values':[],'constraint':'State input, carry, rollback position and write extent must be explicit; final-state comparison, untouched-storage comparison and subsequent output comparison are separate observations.'}, {'id':'sampling_statistical_profile','type':'structured','fields':['logits_profile','allowed_support','seed_vectors','seed_relation','sample_count','target_distribution','tested_bins','frequency_statistic','significance_or_error_threshold'],'observed_values':[],'constraint':'Keep the concrete support, sample count, seed relation, measured statistic and threshold. A finite goodness-of-fit or correlation check does not establish unrestricted sampling correctness or full independence.'}]
paths=[G,S,'vllm/model_executor/layers/mamba/ops/cpu/gdn_attention.py','csrc/cpu/sampling_kernels.cpp']
payload={'status':'ready_for_parent_ontology_review_not_seed_write','condition_dimensions':dims,'units':units,'source_hashes':[{'path':p,'source_sha256':hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()} for p in paths], 'scope':'source-review only; no GPU/CPU test execution; candidate definitions not active until accepted into seed'}
(HERE/'units-ready.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(len(units),len(dims))
