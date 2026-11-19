import ast,csv,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[4]
ROOT=P.parents[1]
SRC=ROOT/'experiments/third_party/reference_repos/vllm-20261105'
OUT=Path(__file__).parent
rows=list(csv.DictReader((P/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
files={
 'tests/model_executor/layers/test_pooler_activations.py':'vllm/model_executor/layers/pooler/activations.py',
 'tests/kernels/moe/test_moe_weight_loading_padded.py':'vllm/model_executor/layers/fused_moe/routed_experts.py',
 'tests/model_executor/layers/test_mm_input_norm.py':'vllm/model_executor/layers/fusion/mm_input_norm.py',
}
review=[];deferred=[];evidence=[]
for path,product in files.items():
 src=(SRC/path).read_text(); tree=ast.parse(src); funcs={};classes={}
 for cls in tree.body:
  if isinstance(cls,ast.ClassDef):
   for fn in cls.body:
    if isinstance(fn,(ast.FunctionDef,ast.AsyncFunctionDef)):
     funcs[cls.name+'::'+fn.name]=fn;classes[cls.name+'::'+fn.name]=cls
 for r in rows:
  if r['snapshot']!='vllm' or r['path']!=path or r['review_status']=='property_reviewed':continue
  fn=funcs.get(r['test_name']); cls=classes.get(r['test_name'])
  if not fn:continue
  n=fn.name;c=cls.name;unit=None;description=''
  if path.endswith('test_pooler_activations.py'):
   if c in ['TestGetActFn','TestResolveClassifierActFn'] or n=='test_default_num_labels_is_none':
    # Factory selection is neither a numerical pooling assertion nor deployment configuration.
    deferred.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'proposed_unit':'model.output_transform_selection','reason':'池化配置选择的激活模块类型或对象身份；需要独立于数值变换的接口选择性质。'});continue
   unit='model.pooling_transform_relation'
   descriptions={'TestPoolerIdentity':'池化恒等变换保留输入张量值，列表输入逐元素保留对应。','TestPoolerNormalize':'池化归一化输出的末维L2范数在给定容差内等于1。','TestPoolerMultiLabelClassify':'多标签输出满足本项断言的区间或正负饱和值要求。','TestPoolerClassify':'按标签数选择的分类变换满足零输入的sigmoid值或输出行和为1的约束。','TestLambdaPoolerActivation':'池化自定义ReLU在给定正负输入上逐元素产生声明输出。'}
   description=descriptions[c]
  elif path.endswith('test_moe_weight_loading_padded.py'):
   if c=='TestGetHiddenDim':
    deferred.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'proposed_unit':'model.parameter_axis_mapping','reason':'仅计算权重hidden轴编号及非法shard_dim异常，没有执行参数加载。'});continue
   if c=='TestOrientFusedWeight':
    unit='model.parameter_representation_relation' if n=='test_qwen3_vl_transposed_block_scale_uses_explicit_layout' else 'model.parameter_storage_layout'
    description='融合专家权重/缩放量按显式转置标志确定输出布局；本项只确认下面断言实际比较的形状、存储身份或转置后数值。'
   elif c=='TestNarrowExpertDataForPadding':
    unit='model.parameter_binding' if n=='test_narrowed_data_shares_storage' else 'model.parameter_storage_layout'
    description='补齐后的专家权重缓冲区按指定hidden或shard轴取得视图，检查本项给定尺寸下的形状与别名；写入测试另外核对逻辑区域值和padding区域保持。'
   elif c=='TestWeightLoadingWithPaddedHiddenSize':
    unit='model.parameter_binding';description='专家权重写入补齐缓冲区后，声明逻辑切片与来源权重对应，保留本项实际检查的padding区域；CPU至CUDA项还比较峰值临时显存小于单分片字节数。'
   elif c=='TestUnquantizedTrtLlmPrePadding':
    unit='model.parameter_storage_layout';description='非量化MoE权重分配前的维度取整按TRTLLM/TRITON后端与给定原始维度返回预期hidden及intermediate。'
   elif c=='TestLoadWeightsExpertBias':
    unit='model.parameter_binding';description='专家检查点名称映射到存在的目标权重或bias加载入口；无bias层跳过可忽略bias，缺失的非bias及未映射融合bias按指定异常拒绝。加载器为记录用替身，本项仅检查目标名称和拒绝行为。'
   elif c=='TestLoadWeightsExpertMapping':
    unit='model.parameter_binding';description='专家检查点经名称映射、逻辑到物理专家复制及融合gate/up拆分后，将本项断言指定的目标名、分片名、专家号和张量传给记录加载器；仅映射表断言不声称真实权重已经加载。'
   elif c=='TestPerTensorScaleCoercion':
    unit='compute.input_admissibility' if n=='test_rejects_non_scalar' else 'model.parameter_representation_relation'
    description='单元素量化尺度转换为0维标量并保留0.5值；多元素输入按本项异常检查拒绝。'
  else:
   source=ast.get_source_segment(src,fn)
   if c in ['TestBuildMMInputNorm','TestFusedMMInputInputLayoutConfig'] or n in ['test_input_norm_is_always_enabled','test_is_identity_params']:
    if n in ['test_invalid_layout_raises']:
     unit='compute.input_admissibility';description='图像归一化模块拒绝未声明的输入布局nchw。'
    else:
     deferred.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'proposed_unit':'compute.normalization_configuration_binding','reason':'归一化配置、启用状态、布局或构造类型判断；没有实际运行像素数值计算。'});continue
   elif n.startswith('test_rejects_'):
    unit='compute.input_admissibility';description='图像归一化调用边界对本项非法维数、通道数、不可整除PCL尺寸或不匹配输出形状抛出AssertionError。'
   elif n=='test_buffers_initialized_on_default_device':
    unit='model.parameter_storage_layout';description='图像归一化weight/bias缓冲区在指定cuda或meta默认设备上创建，比较实际设备字段。'
   elif n in ['test_identity_config_buffers','test_non_identity_buffers']:
    unit='model.parameter_representation_relation';description='图像归一化的mean/std/rescale转换为weight=rescale/std与bias=-mean/std；对本项恒等或RGB参数比较构造缓冲区的值及形状。'
   elif n in ['test_bchw_matches_pcl_equivalent','test_forward_oot_matches_native']:
    unit='compute.execution_relation';description='同一图像输入在BCHW/PCL布局或OOT/native两条路径产生的输出经对应reshape后满足assert_close。'
   else:
    unit='compute.operator_reference_relation';description='设备图像归一化输出与FP32按通道计算(input*rescale-mean)/std后转目标dtype的参考比较；identity路径比较原输入的dtype转换，原始kernel路径比较x*w+b。'
  if not unit:raise ValueError((c,n))
  assertions=[]
  for a in ast.walk(fn):
   if isinstance(a,ast.Assert):assertions.append((a.lineno,ast.unparse(a)))
   elif isinstance(a,ast.Call) and isinstance(a.func,ast.Attribute) and (a.func.attr=='assert_close' or a.func.attr=='raises'):
    assertions.append((a.lineno,ast.unparse(a)))
  helper=[]
  if path.endswith('test_mm_input_norm.py') and any(isinstance(a,ast.Call) and isinstance(a.func,ast.Name) and a.func.id=='_check_against_reference' for a in ast.walk(fn)):
   for a in tree.body:
    if isinstance(a,ast.FunctionDef) and a.name in ['_check_against_reference','_reference_input_norm','_reference_input_norm_nchw']:
     helper.append(f'{path}:{a.lineno}')
   helper_fn=next(a for a in tree.body if isinstance(a,ast.FunctionDef) and a.name=='_check_against_reference')
   helper_assert=next(a for a in ast.walk(helper_fn) if isinstance(a,ast.Call) and isinstance(a.func,ast.Attribute) and a.func.attr=='assert_close')
   assertions.append((helper_assert.lineno,'_check_against_reference: '+ast.unparse(helper_assert)))
  assertions.sort()
  if not assertions:raise ValueError(('no assertion',r))
  checks='；'.join(f'L{line}: {expr}' for line,expr in assertions)
  setup=[]
  for a in cls.body:
   if isinstance(a,(ast.Assign,ast.AnnAssign)) or isinstance(a,ast.FunctionDef) and a.name in ['setup_method','_make_method','_make_experts','_mapping','_load','_checkpoint_weights','_norm']:
    setup.append(f'{path}:{a.lineno}-{a.end_lineno}')
  decorators=[ast.unparse(d) for d in cls.decorator_list+fn.decorator_list]
  review.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'unit_id':unit,'property':'验证类别：'+description+' 本项实际检查：'+checks,'oracle_source':'; '.join([f'{path}:{line}' for line,_ in assertions]+helper),'conditions':json.dumps({'parameters_and_inputs':f'{path}:{fn.lineno}-{fn.end_lineno}','class_setup':setup,'decorators':decorators,'implementation':product,'evidence':'源码核对，未在本轮执行测试；只确认明确断言检查的性质'},ensure_ascii=False),'rule_id':'round10_source_review_'+c})
 evidence.append({'snapshot':'vllm','path':path,'source_sha256':hashlib.sha256((SRC/path).read_bytes()).hexdigest(),'implementation_path':product,'implementation_sha256':hashlib.sha256((SRC/product).read_bytes()).hexdigest(),'method':'人工逐段阅读测试构造、调用、参考与断言，程序仅提取行号/断言及排除已确认入口'})
cols=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=cols,delimiter='\t');w.writeheader();w.writerows(review)
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':evidence,'confirmed_rows':len(review),'deferred':deferred},ensure_ascii=False,indent=2))
(OUT/'proposed-units.json').write_text(json.dumps({'proposals':[{'id':'model.output_transform_selection','property':'按池化任务配置与显式覆盖选择输出变换的模块类型和参数，保留显式提供的可调用对象。'},{'id':'model.parameter_axis_mapping','property':'权重布局、张量阶数与分片轴确定参数的逻辑hidden轴；拒绝不在数据轴范围内的分片轴。'},{'id':'compute.normalization_configuration_binding','property':'预处理归一化配置和布局声明确定归一化实现、启用状态及布局字段。'}],'pending_entry_count':len(deferred)},ensure_ascii=False,indent=2))
(OUT/'notes.md').write_text(f'本轮实际核对3个测试文件和相应实现，输出{len(review)}项已有单元映射；{len(deferred)}项需要更细的配置/轴映射单元，未加入确认表。没有导入上游或使用GPU。\n逐项属性保留原断言和行号。形状检查不扩充为数值正确性，模拟weight_loader只说明传入参数和目标对应，范数/区间检查不扩充为完整参考等价。源码散列见evidence.json。\n另完整阅读EP权重过滤测试及实现，其主要检查名称到专家ID和rank归属、是否跳过权重，现有参数加载单元不足以精确表达，暂未归入确认表。\n')
print('confirmed',len(review),'deferred',len(deferred))
