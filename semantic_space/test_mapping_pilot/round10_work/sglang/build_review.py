import ast,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
PROJECT=ROOT.parents[1]
SRC=PROJECT/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
OUT=Path(__file__).parent
INDEX=ROOT/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv'
rows=list(csv.DictReader(INDEX.open(),delimiter='\t'))
units={x['id'] for x in json.load((ROOT/'semantic_space/space_seed.json').open())['units']}
hf='test/registered/unit/utils/test_hf_transformers.py'
chat='test/registered/unit/entrypoints/openai/test_serving_chat.py'
# Classes below were read in full together with their called source functions.
classes={
'TestNormalizeRopeScalingCompat':'RoPE 配置兼容规范化按缺失字段及嵌套配置规则添加别名并保留已有值',
'TestGetRopeConfig':'RoPE 参数按 v5 字段与 v4 回退规则返回 theta 和参数对象',
'TestPatchTextConfig':'父配置和文本子配置同步缺失 token 字段并保留各自已有值',
'TestGetContextLength':'上下文容量按配置字段优先级、RoPE 缩放例外及默认值计算',
'TestOverrideVHeadDimIfZero':'仅在子配置 v_head_dim 为零时设置默认或显式修补值',
'TestGetHfTextConfig':'按模型配置结构选择文本子配置，传播属性并进行兼容规范化',
}
sel={}
for cls,prop in classes.items():sel[(hf,cls)] = ('deployment.effective_configuration',prop)
chatmap={
'test_stop_str_isolation_between_requests':('request.effective_sampling_parameters','本请求的停止串合并模板默认值，同时保持模板对象及下一请求的停止串不受污染'),
'test_unstreamed_tool_args_completion':('response.stream_payload_consistency','已发送工具实参不完整时返回剩余实参增量且不附加结束原因'),
'test_unstreamed_tool_args_no_completion_needed':('response.stream_payload_consistency','工具实参已全部发送时不重复生成补全增量'),
'test_unstreamed_tool_args_no_parser_data':('response.stream_payload_consistency','解析器没有工具调用数据时不生成补全增量'),
'test_kimi_k2_non_streaming_tool_call_id_format':('response.delivery','非流式工具调用响应保留解析出的名称并按格式构造工具调用 ID'),
'test_kimi_k2_streaming_tool_call_id_format':('response.delivery','首个流式工具调用事件按解析名称和索引构造工具调用 ID'),
'test_kimi_k2_non_streaming_tool_call_id_with_history':('response.delivery','非流式工具调用响应 ID 包含历史调用数偏移并保留解析名称'),
'test_kimi_k2_streaming_tool_call_id_with_history':('response.delivery','流式工具调用事件 ID 包含历史调用数偏移'),
'test_dsv4_content_parts_list_normalized':('input.prompt_text_rendering','消息内容归一化后渲染包含用户文本和角色标记，丢弃的非文本输入不进入文本结果'),
'test_dsv4_task_and_reminder_encode_end_to_end':('input.prompt_text_rendering','任务、提醒及思考模式控制生成的任务标记、角色前缀及消息顺序'),
'test_streaming_abort_yields_error':('response.delivery','带错误状态的中止输出错误事件和 DONE，检查错误码与消息且仅交付这两个事件'),
'test_non_streaming_cached_tokens_details_emits_sglext':('response.delivery','非流式响应按请求标志保留设备、主机、存储层缓存元数据字段'),
'test_non_streaming_chat_response_returns_requested_prompt_ids_and_meta_info':('response.delivery','响应对象及序列化结果保留请求要求的提示 token ID 和原始元信息'),
'test_streaming_cached_tokens_details_emits_sglext':('response.delivery','缓存元数据以一个独立流事件交付，choices 为空并保持各层字段'),
'test_incremental_streaming_output_delta':('response.stream_payload_consistency','增量模式下各 SSE 文本片段拼接为模拟生成器的完整文本'),
'test_extract_routed_dp_rank_from_header_invalid':('input.error_response_classification','无法解析为整数的路由 header 抛出 400 错误并说明字段类型要求'),
'test_non_stream_reasoning_response_preserves_payload_whitespace':('response.reasoning_content_partition','按思考标记分离输出，保留推理内容和普通正文的换行与空白'),
'test_build_chat_response_qwen3_thinking_forces_reasoning':('response.reasoning_content_partition','强制思考解析配置将未带标记的内容放入推理字段并返回空普通正文'),
'test_decode_response_returns_text':('response.delivery','默认响应解码从返回字典提取原 text 字段'),
'test_get_parsed_response_fields_passthrough':('response.delivery','默认响应后处理保留推理字段和工具调用字段'),
'test_required_with_parser_uses_function_call_parser':('tool.call_parse_relation','required 选择下采用已配置工具解析器的名称结果并返回 tool_calls 结束类型'),
'test_required_without_parser_falls_back_to_json':('tool.call_parse_relation','required 且无工具解析器时解析 JSON 数组并返回名称'),
'test_required_without_parser_invalid_json_returns_none':('tool.call_parse_relation','required 且无工具解析器时非法 JSON 回退为空工具调用'),
}
functions={}
for path in (hf,chat):
 text=(SRC/path).read_text();tree=ast.parse(text)
 for cls in tree.body:
  if isinstance(cls,ast.ClassDef):
   for n in cls.body:
    if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
     functions[(path,cls.name+'::'+n.name)]=(n,text)
results=[];details=[]
for r in rows:
 if r['snapshot']!='sglang' or r['review_status'] in ('property_reviewed','supporting_record_reviewed','outside_current_scope'):continue
 path,name=r['path'],r['test_name'];key=(path,name)
 if key not in functions:continue
 n,text=functions[key];cls,method=name.split('::')
 spec=sel.get((path,cls)) if path==hf else chatmap.get(method)
 if not spec or method=='test_no_sub_config':continue
 unit,prop=spec;assert unit in units
 assertions=[]
 for a in ast.walk(n):
  if isinstance(a,ast.Assert) or isinstance(a,ast.Call) and isinstance(a.func,ast.Attribute) and a.func.attr.startswith('assert'):
   assertions.append({'line':a.lineno,'expression':ast.get_source_segment(text,a)})
 if not assertions:raise RuntimeError(name)
 oracle='; '.join(f'{path}:{a["line"]}' for a in assertions)
 prop='类别：'+prop+'；本项实际断言：'+'；'.join(a['expression'].replace('\n',' ') for a in assertions)
 conditions='源码审查，未新增运行；'+('纯配置对象与显式字段，按测试体所列优先级、缺失值和边界条件检查；不代表模型实际执行正确' if path==hf else '受控请求及模拟 tokenizer/generator/parser；核对本测试实际断言字段，模拟结果不作为模型语义或真实缓存正确性证明')
 results.append(dict(snapshot='sglang',path=path,test_name=name,line=r['line'],unit_id=unit,property=prop,oracle_source=oracle,conditions=conditions,rule_id='round10_sglang_body_review_config_response_v1'))
 details.append({'snapshot':'sglang','path':path,'test_name':name,'line':int(r['line']),'unit_id':unit,'assertions':assertions,'test_body_sha256':hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest()})
fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(results)
paths=[hf,chat,'python/sglang/srt/utils/hf_transformers/common.py','python/sglang/srt/utils/hf_transformers_patches.py','python/sglang/srt/entrypoints/openai/serving_chat.py','python/sglang/srt/entrypoints/openai/serving_base.py','python/sglang/srt/entrypoints/openai/encoding_dsv4.py']
evidence={'file_reviews':[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()} for p in paths],'entries':details,'method':'Explicitly reviewed class/function bodies and relevant implementation sections; only pending index entries admitted; assertions preserved verbatim with source anchors. No test execution.', 'count':len(results)}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(len(results));print({p:sum(x['path']==p for x in results) for p in (hf,chat)})
