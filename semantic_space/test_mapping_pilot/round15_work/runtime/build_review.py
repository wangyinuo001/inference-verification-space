"""Manual source-reviewed assignments; AST exports exact evidence and helper checks."""
import ast,csv,json,hashlib
from pathlib import Path
ROOT=Path('/Users/victoria/Documents/WYN_COLLEGE/THU/Agent4Sys/推理系统的最小测试子集');PUB=ROOT/'github_preview/inference-verification-space';SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105';OUT=Path(__file__).resolve().parent
assign={};helpers={};proposals=[];product_refs={}
def group(path,lines,unit,condition):
 for line in lines:
  assert (path,line) not in assign
  assign[path,line]=(unit,condition)
def all_lines(p):
 return [n.lineno for n in ast.walk(ast.parse((SRC/p).read_text())) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test_')]
def unit(id,name,phase,conditions,prop,obs,refs):
 proposals.append(dict(id=id,name=name,kind='obligation',framework_scope=['vllm'],phase=phase,state_scope='shared',conditions=conditions,property=prop,observable=obs,oracle='逐项比较测试声明的值、结构、异常或交互记录；实际检查表达式与替身条件见各入口证据。',evidence={'vllm':[dict(path=p,line=l,kind='source') for p,l in refs]}));product_refs[id]=refs
B='tests/v1/worker/test_gpu_batch_ordering.py'
group(B,[86,92,119,133,157,171,192,239,279],'compute.decode_mode_eligibility','CPU NumPy 请求状态与真实 gather/判定函数；GPUModelRunner 由 __new__ 最小字段构造，adaptive manager 采用返回固定 token 数的替身；没有图捕获或模型执行。')
group(B,[206,212,219],'compute.batch_partition_relation','真实排序与 decode/prefill 切分函数、CPU metadata；按请求 token 数与草稿字段比较顺序和分段计数。')
W='tests/v1/worker/test_gpu_warmup_blocks.py'
group(W,[179,193,217,237,251,278,295,309,368,442],'cache.capacity_plan','按入口实际构造：warmup 执行回调为 _StepRecorder，不计算模型；_StubExtensibleCache 只记录 commit。368/442 实际驱动 KVCacheManager 分配块，并与预约公式逐步比较；不测 GPU 物理容量。')
group(W,[511,537,544],'deployment.effective_configuration','按入口核查 SpeculativeConfig 派生 lookahead 或 warmup 请求 prompt_logprobs 字段；模型配置与执行回调为最小替身，不运行生成。')
K='tests/v1/worker/test_gpu_block_table.py'
group(K,all_lines(K),'cache.block_table_addressing','真实 CUDA BlockTables、暂存写入/聚集/槽位计算；按入口启用融合组、DCP、窗口或 CUDAGraph，参考为手工索引或前次张量副本。本轮只审源码未执行。')
E='tests/v1/worker/test_encoder_runner.py'
group(E,[48,64],'compute.kernel_dispatch_binding','CPU EncoderRunner/ModelState 配置分派；model/图管理器采用 MagicMock，核查 graph 启用字段或非 image 走 eager 输出，不执行编码网络。')
group(E,[157,175,194,217,228,239,277,326,362,389,402],'model.encoder_feature_binding','真实 CPU EncoderCache/EncoderRunner 与手工 embedding 张量；模型编码和模态分组部分采用替身。检查按请求区间及模态选择张量/标记、缓存缺失规则或预编码输入直通，不评价编码数值质量。')
group(E,[300,341,431,463],'model.encoder_cache_lifecycle','真实 CPU EncoderCache；ModelState encoder_runner 部分 MagicMock；核查存入、输入释放/重建及请求局部对象，463 使用 weakref 检查 CPU 存储回收。')
group(E,[415],'diagnostic.encoder_operation_metrics','真实 timed_encoder_operation 与 registry，在空上下文中记录计数和非负耗时；不以此判断 encoder 性能。')
T='tests/v1/worker/test_gpu_thinking_budget.py'
group(T,all_lines(T),'sampling.reasoning_budget_state','真实 CUDA ThinkingBudgetState/RequestState 或 Sampler，人工 token 历史与小词表 logits；推理标记配置为测试类，核查强制结束值/禁用/扫描位置；本轮未运行。')
C='tests/v1/worker/cpu/test_cpu_sampler.py'
group(C,[135,157,177,189,201,212,224,237,248,264,276],'sampling.reasoning_budget_state','真实 CPUThinkingBudgetState/RequestState；非 x86 CPU 且无 Triton 的平台门槛；人工 token 历史和 logits，276 使用临时 Llama 配置与替身 tokenizer，不加载模型。')
group(C,[306,319,348],'sampling.trace_token_binding','真实 CPUTraceReplayState；人工请求历史、声明 token trace 和索引映射；核查按 decode 步替换结果及槽位复用后清零。')
group(C,[360,371,390],'sampling.logit_support_record','CPU 张量含有限值、NaN、Inf 和宽词表；真实统计与 mask 编码，比较计数、紧凑 token 与 bit mask，不执行随机采样。')
group(C,[450,470],'sampling.logit_transform_relation','真实 CPUSampler 在人工 logits 上应用声明 top-k/top-p/temperature，逐项比较有限支持集或缩放张量；非 x86 CPU 无 Triton 门槛。')
group(C,[488,519],'sampling.per_request_modifier_binding','真实 CPU sampler/请求槽位；切换活动请求或复用槽位，核查独立温度与清除旧修饰状态。')
group(C,[539],'sampling.policy','真实 CPU sampler，temperature=0，人工 logits，比较 argmax token。')
group(C,[559,623,683,702],'sampling.watermark_context_policy','真实 CPUWatermarkSampler 与 Gumbel/DualKey watermarker，人工请求历史/种子/逐行开关；核对相同调用重现、普通行及贪心行、上下文重复范围；不证明水印可检测性。')
M='tests/distributed/test_eplb_migration_scheduler.py'
group(M,[49,54],'deployment.effective_configuration','真实 EPLBConfig 的默认及异步 batching 生效属性；不启动分布式执行。')
group(M,[67,93,111,131,141,149,163,174,204,328],'distributed.expert_migration_plan','CPU NumPy placement/真实调度与 move_to/from_buffer；通信器为记录 send/recv 的替身，收到权重手工 fill，核查通信计划/行位置和本地复制，不实际通信。')
A='tests/distributed/test_eplb_algo.py'
group(A,[16],'compute.operator_reference_relation','真实 CPU tensor 负载平均值/最大值统计；手工张量与期望数值，不执行跨 rank allreduce。')
group(A,[32,95,118,142,165,198,216,246,268,292,324,492],'distributed.expert_placement','真实 EPLB policy 和逻辑/物理映射；CPU 算法及可选 CUDA/XPU 输入转 CPU，不启动多 rank。每项只检查其具体形状、计数、索引或保留槽位断言。')
R='tests/distributed/test_sharded_rdt_trainer.py'
group(R,[156,159,166],'deployment.effective_configuration','真实 init info/factory，检查 backend 名、rank 派生角色和工厂登记；不启动 Ray。')
group(R,[171],'deployment.configuration_admissibility','trainer_init 缺 WeightSource 输入，检查明确拒绝；不启动服务。')
group(R,[180,329,354,371,389,424,429,434,440,450,460,478],'model.weight_publish_ownership','真实分组/所有者/存活消费者解析；_ListSource/_DeclaredSource 提供 CPU 人工参数，server/client 与 RPC 是进程内记录替身，reduce_tensor 被替换。354/389 带 CUDA 门槛但不运行 Ray/NIXL。')
group(R,[207,227,252,281],'model.weight_update_session','真实 trainer send_weights 包装与 gather；producer/client 为记录替身、IPC reduce_tensor 替换；部分搬张量需 CUDA。核查 start/update/finish、错误传播和非 sender 不调用 client，不证明真实权重传输成功。')
group(R,[536,558],'model.parameter_representation_relation','实际 CUDA tensor/_RDTProducerServer 字节打包，预置 serve ring/cache，直接调用生产方法；未运行 Ray/NIXL。检查同名不同形状/dtype 的连续请求回读字节。')
CG='tests/v1/worker/cpu/test_cpu_gumbel_sample.py'
group(CG,[216],'compute.operator_reference_relation','非 x86 CPU 无 Triton；真实 CPU gumbel kernel，使用测试 Python Murmur3/uniform/Gumbel 参考和请求映射/种子逐 token 比较。')
group(CG,[248,370,384,395],'sampling.policy','真实 CPU gumbel，人工 logits/种子/位置，检查贪心或有限支持集；多次样本不能证明任意输入永不违反。')
group(CG,[266],'compute.execution_relation','相同种子与位置，比较显式预缩放和内部温度缩放两条 CPU gumbel 路径。')
group(CG,[294,308],'sampling.logit_transform_relation','真实 CPU SamplingStates 依据逐请求索引应用温度/min_p，比较完整张量。')
group(CG,[337,349,405],'sampling.distribution_relation','真实 CPU gumbel；固定种子、递增位置和声明样本数，比较尾部z分数或全词表卡方界限；测试平台门槛为非x86 CPU且无Triton。')
GG='tests/v1/worker/test_gpu_gumbel_sample.py'
group(GG,[78],'compute.operator_reference_relation','真实 Triton Murmur3 kernel，固定种子/位置/offset 与已知整数输出，未进行随机统计。')
group(GG,[125],'compute.output_constraints','真实 Triton uniform64 转换，对输入端点检查有限、开区间及精确数值。')
group(GG,[231,253,375],'sampling.distribution_relation','真实 CUDA/Triton gumbel；声明固定 seed 和按样本递增 pos，尾部 z 分数或全词表卡方界限；不运行模型。')
group(GG,[287],'sampling.randomness_relation','目标/草稿使用同 logits 和种子，核查指定样本的不同率、各自尾部质量和重现性；不证明随机流独立。')
group(GG,[336,352,364,536],'sampling.policy','真实 CUDA/Triton gumbel/greedy_argmax，在手工或随机 logits 下核查 argmax、NaN策略或有限支持集。')
group(GG,[407,464,510],'sampling.draft_logit_snapshot','真实 CUDA/Triton gumbel 写入 logits_cache；比较整型位表示、逐 step 列和额外尾列，宽度不足检查拒绝；不验证完整推测解码分布。')
# New obligations are grounded in production functions rather than test directory names.
unit('compute.decode_mode_eligibility','执行批次的阶段与图模式资格','model_execution',['request_phase','speculative_mode','execution_variant','request_batch_profile'],'请求进度、上下文和草稿字段决定本批属于 prefill、decode 或可用 decode 图的扩展；批次排列后仍按请求状态索引判断，形状一致本身不代替阶段条件。',['request_ids','is_prefilling','decode_graph_eligible','uniform_token_count'],[('vllm/v1/worker/gpu/model_runner.py',1374)])
unit('cache.block_table_addressing','缓存块表与请求槽位对应','cache',['execution_layout','parallel_topology','cache_state','forward_buffer_context'],'按请求行、缓存组、块大小与并行布局，将逻辑位置映射到相应物理槽位；追加和覆盖更新保持计数，失效、填充、窗口外及退出批次的行按规则清零或使用无效槽位，外部输出不改写共享表。',['block_ids','row_counts','slot_mappings','padding','table_contents','data_ptr'],[('vllm/v1/worker/gpu/block_table.py',125),('vllm/v1/worker/gpu/block_table.py',160),('vllm/v1/worker/gpu/block_table.py',213),('vllm/v1/worker/gpu/block_table.py',229)])
unit('model.encoder_feature_binding','编码特征按请求区间与模态绑定','model_execution',['modality','input_form','request_phase','speculative_mode','cache_state'],'根据请求当前处理区间选择相应编码特征，保留位置、模态和批次顺序；预编码输入按规则直通缓存，目标与草稿边界的缺失特征分别遵守声明的拒绝或回退条件。',['embedding_values','modality_tags','placeholder_mask','feature_ids','encoder_inputs','missing_feature_error'],[('vllm/v1/worker/gpu/mm/encoder_runner.py',81),('vllm/v1/worker/gpu/mm/encoder_runner.py',150),('vllm/v1/worker/gpu/mm/encoder_runner.py',207)])
unit('model.encoder_cache_lifecycle','编码输入与结果的请求局部生命周期','model_execution',['modality','cache_state','request_identity_lifecycle_context'],'编码结果按标识保存，未调度输入不调用编码；释放某请求的原始输入不改变其他请求或保留的占位/结果，重建输入后可再次编码。',['feature_data','request_features','encoder_outputs','source_identity','weak_reference','encode_calls'],[('vllm/v1/worker/gpu/mm/encoder_cache.py',20),('vllm/v1/worker/gpu/mm/encoder_cache.py',29),('vllm/v1/worker/gpu/mm/encoder_cache.py',52)])
unit('diagnostic.encoder_operation_metrics','编码操作统计按请求累积并读取清空','service',['diagnostic_record_contract','concurrent_request_profile'],'启用统计后，编码操作次数与计时按声明规则记入参与请求；读取汇总后清除已交付的记录。',['request_ids','num_encoder_calls','encoder_forward_secs','registry_after_read'],[('vllm/v1/worker/gpu/mm/encoder_runner.py',178),('vllm/v1/worker/gpu/mm/encoder_runner.py',198)])
unit('sampling.reasoning_budget_state','推理预算与标记状态控制采样','sampling',['sampling_mode','logit_modifier_profile','request_phase','stop_pattern_kind'],'根据请求 token 历史的最近推理开始/自然结束标记与预算决定是否强制下一个结束标记；多 token 标记可续写，历史扫描状态按提交进度推进，槽位复用清除旧请求策略。',['logits','forced_end_token','cached_scan_position','request_budget','active_request'],[('vllm/v1/worker/cpu/sampler.py',90),('vllm/v1/worker/cpu/sampler.py',119),('vllm/v1/worker/cpu/sampler.py',150),('vllm/v1/worker/gpu/sample/thinking_budget.py',113)])
unit('sampling.trace_token_binding','采样重放 token 按请求与步数绑定','sampling',['request_phase','sampling_mode','sampling_batch_transition_context'],'声明重放序列的请求在相应 decode 步使用对应 token；无效映射、未启用及超出序列范围的行保持原结果，请求槽位复用更新有效长度。',['sampled_tokens','request_indices','decode_steps','trace_lengths'],[('vllm/v1/worker/cpu/sampler.py',234),('vllm/v1/worker/cpu/sampler.py',244)])
unit('sampling.logit_support_record','采样支持集与非有限值记录','sampling',['numeric_format','logit_modifier_profile','request_batch_profile'],'逐行记录非有限值或有限 token 支持集；紧凑 token、位图和数量按词表及有效采样行对应，支持集超过紧凑容量时保留完整集合表示。',['nan_counts','support_counts','token_ids','packed_mask','active_rows'],[('vllm/v1/worker/cpu/sampler.py',47),('vllm/v1/worker/cpu/sampler.py',437)])
unit('sampling.watermark_context_policy','水印采样的上下文与请求策略','sampling',['sampling_mode','logit_modifier_profile','request_phase','sampling_statistical_profile'],'水印采样按逐请求开关、温度与声明上下文执行；重复上下文依据历史范围决定跳过策略，普通行与贪心行采用其对应路径，相同输入与种子遵守声明重现关系。',['context_tokens','repeated_contexts','skip_mask','sampled_tokens','output_logits'],[('vllm/v1/worker/cpu/sampler.py',891),('vllm/v1/worker/cpu/sampler.py',961),('vllm/v1/worker/cpu/sampler.py',977)])
unit('distributed.expert_placement','专家逻辑与物理放置关系','model_execution',['parallel_topology','parameter_mapping','execution_layout'],'给定负载和拓扑，专家放置及反向映射保持索引范围、覆盖和副本总数；具体均衡策略由声明算法决定，局部重排保留同设备已有专家并保持新的专家多重集合。',['physical_to_logical','logical_to_physical','replica_counts','retained_slots'],[('vllm/distributed/eplb/policy/default.py',191),('vllm/distributed/eplb/policy/default.py',273),('vllm/distributed/eplb/eplb_state.py',1316)])
unit('distributed.expert_migration_plan','专家迁移的端点与接收位置','model_execution',['parallel_topology','parameter_mapping','tensor_transport_profile'],'新旧放置决定迁移源、目的、专家与接收行；按策略合并同端点流并避开批内端点冲突，本地已有副本不重复传输，重复目标从指定接收行复制。',['migration_flows','batches','source_rank','destination_rank','expert_id','receive_row'],[('vllm/distributed/eplb/migration_scheduler.py',20),('vllm/distributed/eplb/migration_scheduler.py',106)])
unit('model.weight_publish_ownership','权重发布的所有者与参与者元数据','model_execution',['parameter_mapping','parameter_lifecycle','parallel_topology'],'权重名称按元数据归属生产 rank 与发布组；各 rank 对整体元数据形成一致计划，错误归属拒绝；存活消费者标识去重后与计数一致，并作为本轮发布与释放门禁参数。',['group_names','owner_sets','name_owner_class','live_consumer_ids','live_count','published_groups','metadata_digest'],[('vllm/distributed/weight_transfer/sharded_rdt_trainer.py',1063),('vllm/distributed/weight_transfer/sharded_rdt_trainer.py',1106),('vllm/distributed/weight_transfer/sharded_rdt_trainer.py',1201),('vllm/distributed/weight_transfer/sharded_rdt_trainer.py',1227)])
unit('sampling.draft_logit_snapshot','草稿概率缓存保留对应步的输入 logits','sampling',['numeric_format','execution_layout','speculative_mode'],'草稿采样缓存按请求和步数保存温度变换前的输入 logits 位值；跨步写入不覆盖既有列，宽度与布局满足目标缓存要求。',['logits_bits','cache_columns','cache_strides','untouched_values','rejection'],[('vllm/v1/worker/gpu/sample/gumbel.py',349)])

for u in proposals:
 if u['id']=='diagnostic.encoder_operation_metrics':u.update(phase='diagnostic',state_scope='diagnostic_artifact',semantic_domain='diagnostic_support')
 if u['id']=='distributed.expert_migration_plan':u['evidence']['vllm'] += [dict(path='vllm/distributed/eplb/rebalance_execute.py',line=l,kind='source') for l in [305,501]]
helper_links={}
for line in [179,193,251]:helper_links[W,line]=[(W,'_assert_covers_lookahead')]
for line in [67,111]:helper_links[M,line]=[(M,'_assert_no_endpoint_conflict')]
helper_links[A,492]=[(A,'_validate_intragpu_rearrangement')]
pending_units=json.loads((PUB/'semantic_space/test_mapping_pilot/round12_work/runtime/proposed-units.json').read_text())
existing=json.loads((PUB/'semantic_space/space_seed.json').read_text());labels={u['id']:u['name'] for u in existing['units']+pending_units+proposals};valid=set(labels)
rows=[];reviews=[];hashes={};skipped=[];missing=[];trees={}
index=list(csv.DictReader((PUB/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
def extract_checks(node,origin='entry'):
 result=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Assert):result.append(dict(line=n.lineno,expression=ast.unparse(n),origin=origin))
  elif isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and (n.func.attr.startswith('assert') or n.func.attr=='raises'):result.append(dict(line=n.lineno,expression=ast.unparse(n),origin=origin))
 return result
for entry in index:
 if entry['snapshot']!='vllm':continue
 path=entry['path'];line=int(entry['line']);key=(path,line)
 if key not in assign:continue
 if entry['review_status']=='property_reviewed':skipped.append(key);continue
 unit,cond=assign[key];assert unit in valid,unit
 if path not in trees:
  data=(SRC/path).read_bytes();source=data.decode();tree=ast.parse(source);trees[path]=(source,tree);hashes[path]=hashlib.sha256(data).hexdigest()
 source,tree=trees[path]
 nodes=[n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno==line];assert len(nodes)==1,key
 node=nodes[0];checks=extract_checks(node)
 for hp,helper in helper_links.get(key,[]):
  ht=ast.parse((SRC/hp).read_text());hn=[n for n in ast.walk(ht) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==helper];assert len(hn)==1,(key,helper)
  checks += [dict(**x,path=hp) for x in extract_checks(hn[0],helper)]
 assert checks,key
 prop=labels[unit]+'。本项断言：'+'；'.join(dict.fromkeys(c['expression'] for c in checks))
 oracle=f"{path}:{checks[0]['line']}";decorators='; '.join(ast.unparse(n) for n in node.decorator_list)
 calls=[dict(line=n.lineno,expression=ast.unparse(n)) for n in ast.walk(node) if isinstance(n,ast.Call)]
 rows.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,unit_id=unit,property=prop,oracle_source=oracle,conditions=cond+' 参数化：'+decorators,rule_id='round15_source_review_runtime'))
 reviews.append(dict(snapshot='vllm',path=path,test_name=entry['test_name'],line=line,end_line=node.end_lineno,unit_id=unit,property=prop,source_sha256=hashes[path],definition_sha256=hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest(),checks=checks,calls=calls,source_excerpt=ast.get_source_segment(source,node),helper_checks=helper_links.get(key,[]),review_method='逐项核对调用、fixture/辅助函数与断言；AST 仅导出证据；未执行测试'))
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
file_reviews=[]
for p,h in hashes.items():
 s,t=trees[p];helpers=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(s,n)) for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('test')]
 file_reviews.append(dict(snapshot='vllm',path=p,source_sha256=h,helper_sources=helpers))
for helper_path in ['tests/distributed/test_weight_transfer.py']:
 helper_data=(SRC/helper_path).read_bytes();helper_source=helper_data.decode();helper_tree=ast.parse(helper_source)
 file_reviews.append(dict(snapshot='vllm',path=helper_path,source_sha256=hashlib.sha256(helper_data).hexdigest(),review_role='shared_fixture',helper_sources=[dict(name=n.name,line=n.lineno,end_line=n.end_lineno,source=ast.get_source_segment(helper_source,n)) for n in helper_tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('test')]))
(OUT/'evidence.json').write_text(json.dumps(dict(file_reviews=file_reviews,entry_reviews=reviews),ensure_ascii=False,indent=2))
source_reviews=[]
for u in proposals:
 for v in u['evidence']['vllm']:
  data=(SRC/v['path']).read_bytes();s=data.decode();t=ast.parse(s);candidates=[n for n in ast.walk(t) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno==v['line']]
  excerpt=ast.get_source_segment(s,candidates[0]) if candidates else '\n'.join(s.splitlines()[max(0,v['line']-1):v['line']+60])
  source_reviews.append(dict(**v,snapshot='vllm',source_sha256=hashlib.sha256(data).hexdigest(),source_excerpt=excerpt))
(OUT/'product-source-evidence.json').write_text(json.dumps(source_reviews,ensure_ascii=False,indent=2))
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2))
(OUT/'source_hashes.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2))
summary=dict(mapped_entries=len(rows),reviewed_files=len(hashes),already_reviewed_excluded=len(skipped),by_file={p:sum(r['path']==p for r in rows) for p in hashes},by_unit={u:sum(r['unit_id']==u for r in rows) for u in sorted({r['unit_id'] for r in rows})},runtime_execution=False)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))

assert set(assign)=={(r['path'],int(r['line'])) for r in rows}|set(skipped),sorted(set(assign)-{(r['path'],int(r['line'])) for r in rows}-set(skipped))
(OUT/'excluded-reviewed.json').write_text(json.dumps([dict(snapshot='vllm',path=B,line=287,reason='静态 AST 源码结构检查，不直接检查运行期 decode 资格')],ensure_ascii=False,indent=2))
