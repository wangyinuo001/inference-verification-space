import ast,csv,json,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
P=OUT.parents[1];ROOT=P.parents[3];SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'
D='compute.kernel_dispatch_binding';N='compute.operator_reference_relation'
S={}
def add(path,impl,items):
 for name,unit,desc in items:S[path,name]=(unit,desc,impl)
p='tests/kernels/moe/test_unquantized_backend_selection.py'
impl=['vllm/model_executor/layers/fused_moe/oracle/unquantized.py:231','vllm/model_executor/layers/fused_moe/experts/cpu_moe.py:347']
names={
'x86_cpu_unquantized_kernel_selection':'模拟x86与AMX能力，在128维输入及BF16/FP16组合中选择AMX或通用CPU专家类。',
'arm_cpu_unquantized_kernel_selection':'模拟ARM与linux/darwin，在BF16/FP16组合中选择NEON或通用CPU专家类。',
'select_default_backend_by_platform':'禁用FlashInfer与AITER探针，模拟各平台检查默认后端；TPU/OOT无专家类，其他后端专家类非空。',
'select_rocm_aiter_backend':'模拟ROCm及AITER已启用，选中AITER且专家类非空。',
'select_rocm_aiter_backend_non_gated_activation_falls_back':'环境变量启用AITER但激活SILU_NO_MUL时回退到其他可用专家实现。',
'explicit_aiter_backend_non_gated_activation_still_raises':'显式固定aiter且非门控激活时抛ValueError。',
'select_cuda_flashinfer_trtllm_backend':'模拟CUDA和monolithic支持，显式TRTLLM及EP选择monolithic专家类。',
'select_cuda_flashinfer_trtllm_modular_backend':'模拟monolithic不支持而modular支持，显式TRTLLM选择modular。',
'select_cuda_flashinfer_trtllm_modular_for_standard_all2all':'模拟设备可用，DP/EP=2和指定标准all2all后端选择TRTLLM modular。',
'require_decomposed_backend_rejects_monolithic_kernels':'模拟设备可用，require_decomposed_backend为真时monolithic支持谓词拒绝并说明routes adjusted。',
'select_cuda_deepep_ht_falls_back_from_trtllm':'模拟CUDA，DeepEP高吞吐且DP/EP=2时auto选择TritonExperts。',
'select_cuda_flashinfer_trtllm_ag_rs_uses_monolithic':'模拟CUDA，AG/RS通信及Renormalize/MiniMax2路由时auto或显式TRTLLM选择monolithic。',
'select_cuda_flashinfer_cutlass_backend':'模拟TRTLLM不支持而CUTLASS支持，显式CUTLASS及EP选中该后端。',
'select_lora_backend_prefers_triton':'dummy配置启用LoRA时选择Triton且专家类非空。',
'select_lora_explicit_non_triton_backend':'dummy配置启用LoRA覆盖显式CUTLASS请求，返回Triton。',
'select_explicit_triton_backend':'dummy配置显式Triton，在LoRA启用或关闭时均返回Triton。',
'quantization_only_backend_falls_back_to_auto':'未量化层请求humming/marlin时选择结果与默认auto完全相同。'}
add(p,impl,[('test_'+k,D,v+' 本项只检查支持/选择，不执行专家数值内核。') for k,v in names.items()])
p='tests/kernels/moe/test_topk_softplus_sqrt.py'
impl=['vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:148','vllm/model_executor/layers/fused_moe/router/dsv4_topk.py:134','vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:783']
add(p,impl,[(f'test_{k}',u,v) for k,u,v in [
('hash_routing_backend_selection',D,'模拟gfx950/AITER能力，构造FusedMoEFactory并检查指定MXFP4 backend专家类名及hash表随配置保留；未执行专家计算。'),
('hash_routing_rejects_monolithic_without_support',D,'将monolithic hash支持谓词置False，显式指定该实现且有hash表时拒绝并说明hash routing。'),
('aiter_w4a16_expert_parallel_selection',D,'模拟gfx950，hash路由允许naive EP8但拒绝EPLB和DP all2all组合。'),
('sqrtsoftplus_bias_uses_deepseek_v4_routing_method',D,'sqrtsoftplus带bias、topk8，归一化启用映射DeepseekV4，否则Unspecified。'),
('fused_topk_softplus_sqrt',N,'fused_topk_bias sqrtsoftplus结果对照FP32 softplus平方根参考：bias只决定专家选择，原score用于权重并按参数归一化/缩放；按专家ID排序后比较ID与权重。'),
('fused_topk_softplus_sqrt_hash',N,'hash表指定每词topk专家；fused_topk_bias与同表查找、FP32 softplus平方根权重参考比较，显式归一化和缩放。'),
('dsv4_fast_topk',N,'FP32 256/384专家topk6、归一化缩放1.5的快速路由对独立PyTorch参考；显式padding行置ID=-1权重0，图replay全padding后重新检查哨兵；包含0 token空输出。'),
('fused_topk_softplus_sqrt_padding',N,'直接CUDA算子对PyTorch参考比较有效行ID与权重；单独检查显式padding ID=-1、权重0及NaN行有限零权重；覆盖hash/bias开关和同分截断。'),
('fused_topk_softplus_sqrt_bias_vl',N,'混合普通词、五个图像哨兵和哨兵上界之外token，融合路由对照按图像区间选择bias_vl的FP32参考；helper按ID排序检查权重。'),
('fused_topk_softplus_sqrt_hash_bias_vl',N,'图像哨兵使用bias_vl topk、文本走hash表；结果对照PyTorch参考并额外逐项确认文本ID直接来自hash表。'),
('dsv4_fast_topk_bias_vl',N,'直接dsv4_topk处理33行256专家混合VL输入，与FP32参考比较有序ID和权重并确认int64输出。')]])
p='tests/kernels/attention/test_attention_selector.py'
impl=['vllm/v1/attention/selector.py:110','vllm/v1/attention/backends/fa_utils.py:74','vllm/v1/attention/backends/flash_attn.py:357']
names={
'backend_selection':'平台对象切换，显式注意力backend在MLA/block_size/DCP条件下选中声明类或拒绝不支持组合；仍受真实设备能力和可选依赖skip限制。',
'supported_backend_preserves_dcp_eligibility':'同一支持后端配置启用DCP前后validate_configuration返回相同的无效原因列表。',
'fp32_fallback':'FP32 head16在CPU选择CPU_ATTN、CUDA选择FLEX_ATTENTION，ROCm拒绝无有效backend。',
'auto_backend_string':'auto大小写变体解析为None后端。',
'auto_backend_selection_behavior':'auto与None配置均解析None，CPU选择得到相同后端名。',
'per_head_quant_scales_backend_selection':'显式per-head FP8 scale资格仅允许测试声明的FA3组合，其余指定后端拒绝。',
'int4_per_token_head_head_size_selection':'INT4 per-token-head KV和64/80/96/128/192/256 head尺寸选中Triton注意力。',
'non_causal_backend_selection':'AttentionConfig非因果标志组合仍选择支持的显式FA/FlashInfer后端。',
'non_causal_autoselect_backend':'CUDA auto及非因果要求选出的backend声明支持non-causal。',
'flash_attn_rejects_unhandled_kv_cache_dtypes':'支持谓词拒绝未处理的FP8变体、NVFP4及per-token-head量化格式。',
'flash_attn_accepts_handled_fp8_variants':'模拟fa_utils XPU平台，支持谓词接受fp8与fp8_e4m3。',
'fa4_hd256_fallback_matrix':'模拟Blackwell和FA可用，根据head维、block、sink/softcap/MM prefix/RSWA/DCP/FP8配置返回FA4或FA2。',
'fa4_hd256_block_size_advertisement':'模拟Blackwell，普通hd256 FA公布128块，其余配置公布16倍数；首选块与期望相同。',
'fa4_hd256_mm_prefix_deselects_flash_attn':'模拟Blackwell hd256与MM prefix组合返回非空不支持原因。',
'fa4_hd256_impl_selection':'真实Blackwell构造FlashAttentionImpl，decoder和全局encoder选择FA4，局部encoder回退FA2；仅检查版本与标志。',
'hopper_mm_prefix_selects_triton_flash_attn':'模拟Hopper能力与FA可用，Gemma4配置按MM prefix选择TRITON_FLASH_ATTN或FLASH_ATTN。',
'mm_prefix_selects_composite_without_changing_causal_default':'模拟Blackwell能力与FA可用，Gemma4配置按MM prefix选择TRITON_FLASHINFER或FLASHINFER，覆盖FP8 KV。',
'composite_rejects_features_not_shared_by_both_routes':'Gemma4 composite配置拒绝RSWA、INT8 KV、小块、DCP、PCP、设备决定query长度，检查具体拒绝原因。',
'rswa_selection_does_not_reuse_causal_result':'模拟Blackwell，同一配置先缓存causal后开启RSWA必须拒绝旧composite选择；改用Triton后成功。'}
add(p,impl,[('test_'+k,D,v+' 本项未运行注意力数值计算。') for k,v in names.items()])
add(p,impl,[('test_invalid_backend','deployment.configuration_admissibility','无效backend枚举名在配置构建前被拒绝。')])

p='tests/kernels/quantization/test_scaled_mm_kernel_selection.py'
impl=['vllm/model_executor/kernels/linear/__init__.py:257','vllm/model_executor/kernels/linear/scaled_mm/aiter.py:64','vllm/model_executor/kernels/linear/scaled_mm/cpu.py:212','vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:46','vllm/model_executor/layers/quantization/utils/humming/priority.py:26']
add(p,impl,[(f'test_{k}',u,v) for k,u,v in [
('auto_linear_backend_is_unchanged_without_overrides',D,'多个量化键无override且auto时保留候选列表身份和auto配置。'),
('linear_backend_override_is_quantization_specific',D,'全局cutlass但NVFP4 W4A16特定override为humming，其他量化键仍用cutlass。'),
('fp8_linear_backend_override',D,'模拟CUDA和Humming支持谓词，FP8特定override实际构造Humming kernel对象，未执行矩阵乘法。'),
('aiter_int8_kernel_can_implement_scale_layouts',D,'对称INT8按静态激活、channelwise和fused标志检查支持谓词；仅per-token激活与per-channel或fused权重组合受支持。'),
('aiter_int8_per_tensor_weight_selection_and_loading',D,'模拟ROCm和AITER/Triton均支持，通过compressed-tensors或Quark构造权重，fused分片选AITER，单片选Triton；fused加载后检查转置布局及0.5/2.0分片scale按输出通道重复；无GEMM执行。'),
('cpu_kernel_accepts_all_configs',D,'CPU支持谓词仅核对列出的两种INT8配置返回可实现，不推及全部配置。'),
('cpu_sgl_int8_linear_accepts_3d_input','compute.execution_relation','CPU SGL INT8相同权重/bias，三维输入调用与先展平二维再恢复输出逐元素精确一致；参考是同一内核变形执行，不是独立数值参考。'),
('register_oot_linear_kernel',D,'注册自定义空实现INT8 kernel并模拟OOT平台后，初始化选择该类实例；不验证空apply实现数值。'),
('register_oot_fp8_block_linear_kernel',D,'注册自定义空实现block FP8 kernel并模拟OOT，指定动态128激活/静态128块权重后构造该类实例。'),
('humming_priority_only_reorders_humming_and_marlin',D,'能力80保留列表顺序，能力90将Humming移至Marlin之前，其他候选相对顺序及输入列表不变。')]])

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
for rd in ['round12_work','round13_work']:
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
    if isinstance(a,ast.Call) and isinstance(a.func,ast.Name) and a.func.id in dd and a.func.id not in seen and not a.func.id.startswith('test_'):
     seen.add(a.func.id);helpers.append(dd[a.func.id]);scan(dd[a.func.id])
  scan(fn)
  checks=[]
  for f in [fn]+helpers:
   for a in ast.walk(f):
    if isinstance(a,ast.Assert) or (isinstance(a,ast.Call) and (ast.unparse(a.func).split('.')[-1].startswith('assert_') or ast.unparse(a.func).endswith('.raises'))):checks.append({'source':f'{path}:{a.lineno}','expression':ast.unparse(a)})
  assert checks
  cond={'test_range':f'{path}:{fn.lineno}-{fn.end_lineno}','module_conditions':[ast.unparse(x) for x in tree.body if isinstance(x,(ast.Assign,ast.AnnAssign))],'decorators':[ast.unparse(x) for x in fn.decorator_list],'input_and_call_source':ast.get_source_segment(txt,fn),'helpers':[{'source':f'{path}:{h.lineno}','body':ast.get_source_segment(txt,h)} for h in helpers],'implementation_anchors':impl,'external_helper_anchors':(['tests/kernels/moe/utils.py:67'] if 'make_dummy_moe_config' in ast.unparse(fn) else []),'validation':'仅静态源码核对，未导入或执行上游，未使用GPU'}
  row={k:r[k] for k in ['snapshot','path','test_name','line']};row.update(unit_id=unit,property=desc+' 实际断言：'+'；'.join(x['expression'] for x in checks),oracle_source=';'.join(dict.fromkeys(x['source'] for x in checks)),conditions=json.dumps(cond,ensure_ascii=False),rule_id='round13_compute_source_'+Path(path).stem)
  rows.append(row);ev.append({**{k:row[k] for k in ['snapshot','path','test_name','line','unit_id']},'review':desc,'assertions':checks,'conditions':cond})
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
impl_paths=sorted({x.rsplit(':',1)[0] for _,_,im in S.values() for x in im} | {
 'tests/kernels/moe/utils.py',
 'vllm/platforms/cuda.py',
 'vllm/v1/attention/backends/composite.py',
 'vllm/v1/attention/backends/triton_flashinfer.py',
 'vllm/model_executor/layers/fused_moe/config.py',
 'vllm/model_executor/layers/fused_moe/modular_kernel.py',
 'vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py',
 'vllm/model_executor/layers/quantization/quark/schemes/quark_w8a8_int8.py',
})
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':reviews,'entry_reviews':ev,'supporting_source_reviews':[{'snapshot':'vllm','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()} for p in impl_paths],'confirmed_rows':len(rows),'deferred':[{'snapshot':'vllm','path':'tests/kernels/moe/test_topk_softplus_sqrt.py','test_name':'test_torch_topk_softplus_sqrt_breaks_ties_by_expert_id','reason':'仅验证测试文件PyTorch参考helper，未调用生产路由器；不作为生产算子数值确认'}]},ensure_ascii=False,indent=2))
(OUT/'notes.md').write_text(f'确认{len(rows)}条。逐项阅读输入、调用和断言，相关实现只审阅涉及分支。mock资格选择与真实数值路由分别映射；参考helper自测暂不确认。未执行测试或使用GPU。无新增性质。\n')
print(ROOT,len(rows))
