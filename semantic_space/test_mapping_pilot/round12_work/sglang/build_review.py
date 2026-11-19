"""Emit source reviews of explicitly read API/parser files; not a name classifier."""
import ast,csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];PROJECT=ROOT.parents[1]
SRC=PROJECT/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
OUT=Path(__file__).parent;BASE='test/registered/unit/'
FIELD='request.field_binding';ADMIT='request.declared_admission_contract'
PAYLOAD='response.observed_payload_contract';EVENT='response.stream_event_lifecycle'
ERROR='input.error_response_classification';REASON='response.reasoning_content_partition'
USAGE='response.usage_accounting';MM='input.multimodal_binding'
TEXT='input.prompt_text_rendering';PARAM='request.effective_sampling_parameters'
ASR='response.transcription_text_projection';INIT='response.reasoning_parser_initial_state'
FORMAT='input.content_format_selection';CONFIG='deployment.configuration_structure'
mapping={}
def add(file,unit,names):
 for name in names.split():
  assert (file,name) not in mapping,(file,name)
  mapping[(file,name)]=unit

f='entrypoints/anthropic/test_serving.py'
add(f,EVENT,'''test_stream_closes_tool_block_before_text_delta test_stream_reasoning_content_uses_thinking_block test_stream_text_then_tool_use_closes_text_block test_stream_tool_use_without_arguments_is_not_empty_completion test_stream_no_usage_chunk_emits_error_event test_stream_text_then_thinking_closes_text_block test_stream_consecutive_tool_calls_get_separate_blocks test_stream_finish_chunk_with_payload_emits_delta test_stream_empty_completion_with_finish_reason_emits_message_delta test_stream_no_finish_no_content_still_emits_api_error test_stream_upstream_error_envelope_is_forwarded test_stream_parse_failure_closes_open_content_block test_stream_pre_first_chunk_value_error_emits_envelope''')
add(f,USAGE,'''test_stream_usage_subtracts_cache_read_and_omits_final_input_tokens test_non_streaming_usage_subtracts_cache_read_tokens test_cache_read_exceeds_prompt_tokens_clamps_to_zero test_usage_without_prompt_tokens_details''')
add(f,FIELD,'''test_tool_result_search_result_content_is_flattened test_builtin_web_search_tool_without_schema_is_skipped test_request_thinking_enabled_invokes_apply_reasoning_enabled test_request_thinking_disabled_invokes_apply_reasoning_enabled test_request_thinking_enabled_with_budget_tokens_logs_warning test_request_thinking_adaptive_is_treated_as_enabled test_request_thinking_display_omitted_logs_warning_but_still_enables test_request_output_config_effort_maps_to_reasoning_effort test_request_output_config_task_budget_is_logged_not_enforced test_request_task_budget_with_remaining_is_accepted test_request_betas_is_accepted_and_logged test_server_tool_only_with_tool_choice_auto_is_allowed test_tool_choice_named_custom_tool_is_resolved test_user_message_text_tool_text_preserves_order test_empty_text_assistant_turn_preserves_role_alternation test_in_messages_system_role_folded_to_top_level test_in_messages_system_role_merged_with_top_level test_top_level_system_only_is_unchanged test_validator_handles_constructed_message_objects test_thinking_history_drop_on_missing_detector''')
add(f,ADMIT,'''test_custom_tool_without_schema_is_rejected test_request_thinking_enabled_requires_budget_tokens test_request_thinking_enabled_budget_below_min_is_rejected test_request_thinking_disabled_with_display_is_rejected test_request_thinking_disabled_with_budget_is_rejected test_request_thinking_adaptive_with_budget_is_rejected test_redacted_thinking_history_is_rejected test_server_tool_only_with_tool_choice_any_raises_400 test_tool_choice_named_unknown_tool_raises_400''')
add(f,ERROR,'''test_non_streaming_openai_error_response_is_forwarded test_non_streaming_error_with_non_json_body test_non_streaming_error_5xx_scrubs_message test_error_response_does_not_leak_exception_name''')
add(f,PAYLOAD,'''test_non_streaming_response_includes_thinking_block test_convert_response_non_streaming_empty_content_keeps_block test_stop_reason_content_filter_falls_back_with_warning''')
add(f,TEXT,'test_assistant_thinking_history_is_rewrapped_for_chat_template')

f='entrypoints/openai/test_protocol.py'
add(f,PAYLOAD,'''test_model_card_serialization test_empty_model_list test_model_list_with_cards test_hidden_states_excluded_when_none test_hidden_states_included_when_not_none test_prompt_token_ids_and_meta_info_serialization''')
add(f,FIELD,'''test_basic_completion_request test_completion_request_sglang_extensions test_basic_chat_completion_request test_chat_completion_tool_choice_validation test_chat_completion_sglang_extensions test_chat_completion_tito_extensions test_chat_completion_reasoning_effort test_chat_completion_reasoning_effort_none test_chat_completion_reasoning_effort_none_from_reasoning_dict test_chat_completion_reasoning_effort_max test_chat_completion_json_format test_function_defaults_preserve_strict test_function_defer_loading_true_serialized test_function_defer_loading_false_serialized test_tool_level_defer_loading_propagates_to_function test_function_level_defer_loading_wins_over_tool_level test_tool_reference_content_part_accepted test_model_serialization_roundtrip''')
add(f,ADMIT,'test_completion_request_validation_errors test_invalid_tool_choice_type test_negative_token_limits')

f='entrypoints/openai/test_serving_completions.py'
add(f,FIELD,'test_single_string_prompt test_single_token_ids_prompt')
add(f,PAYLOAD,'''test_echo_with_string_prompt_streaming test_echo_with_list_of_strings_streaming test_echo_with_token_ids_streaming test_echo_with_multiple_token_ids_streaming test_prepare_echo_prompts_non_streaming test_logprobs_false_non_streaming test_non_streaming_cached_tokens_details_emits_sglext test_streaming_cached_tokens_details_emits_sglext''')
add(f,PARAM,'test_response_format_json_object test_response_format_json_schema test_response_format_structural_tag test_response_format_none')
add(f,ADMIT,'test_response_format_json_schema_missing_schema')
add(f,EVENT,'test_streaming_abort_yields_error')

f='entrypoints/openai/test_serving_embedding.py'
add(f,FIELD,'test_convert_single_string_request test_convert_list_string_request test_convert_token_ids_request')
add(f,MM,'''test_convert_multimodal_request test_convert_multimodal_request_with_jinja_chat_template test_convert_image_only_multimodal_request_with_jinja_chat_template test_convert_video_multimodal_request_with_jinja_chat_template test_multimodal_request_falls_back_when_no_chat_template test_multimodal_request_with_no_tokenizer_uses_fallback''')
add(f,ERROR,'test_jinja_template_errors_are_raised_as_value_error test_jinja_template_syntax_error_includes_location test_non_jinja_template_errors_are_raised_as_value_error')

f='parser/test_jinja_template_utils.py'
add(f,TEXT,'''test_detect_llama4_openai_format test_detect_deepseek_string_format test_detect_invalid_template test_detect_empty_template test_detect_msg_content_pattern test_detect_m_content_pattern test_process_content_already_string test_process_content_filter_none_values test_detect_template_with_filter test_detect_template_with_is_test test_detect_template_with_slice test_detect_template_no_content_loop_is_string test_detect_msg_content_without_multimodal_keywords''')
add(f,MM,'''test_process_content_openai_format test_process_content_string_format test_process_content_with_audio test_process_content_with_modalities test_process_content_with_video test_process_content_video_with_max_dynamic_patch test_process_content_v32_encoding test_process_content_v32_encoding_accepts_responses_input_text test_process_content_video_with_modalities''')
add(f,ADMIT,'test_process_content_invalid_format_raises')

# These families were read method by method. Their remaining entries all test
# the explicitly named operation; exact per-test assertions are retained below.
family_units={'managers/test_mm_process_config.py':'deployment.effective_configuration',
              'utils/test_field_validators.py':ADMIT}
sources={
'entrypoints/anthropic/test_serving.py':['entrypoints/anthropic/serving.py','entrypoints/anthropic/protocol.py'],
'entrypoints/openai/test_protocol.py':['entrypoints/openai/protocol.py'],
'entrypoints/openai/test_serving_completions.py':['entrypoints/openai/serving_completions.py','entrypoints/openai/protocol.py'],
'entrypoints/openai/test_serving_embedding.py':['entrypoints/openai/serving_embedding.py'],
'parser/test_jinja_template_utils.py':['parser/jinja_template_utils.py'],
'managers/test_mm_process_config.py':['server_args.py','multimodal/processors/base_processor.py','multimodal/processors/ernie45_vl.py','multimodal/processors/midashenglm.py'],
'utils/test_field_validators.py':['utils/field_validators.py','managers/io_struct.py'],
'entrypoints/openai/test_whisper_adapter.py':['entrypoints/openai/transcription_adapters/whisper.py','multimodal/processors/whisper.py'],
'parser/test_reasoning_parser.py':['parser/reasoning_parser.py']}
pending=list(csv.DictReader((ROOT/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
cache={};nodes={};helpers={}
for suffix in sources:
 p=BASE+suffix;t=(SRC/p).read_text();cache[p]=t;tree=ast.parse(t);helpers[p]=[]
 for top in tree.body:
  if isinstance(top,ast.ClassDef):items=[(top.name+'::'+x.name,x) for x in top.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
  elif isinstance(top,(ast.FunctionDef,ast.AsyncFunctionDef)):items=[(top.name,top)]
  else:continue
  for name,n in items:
   if n.name.startswith('test'):nodes[(p,name,n.lineno)]=n
   else:helpers[p].append({'name':name,'line':n.lineno,'source':ast.get_source_segment(t,n)})

def unit_for(suffix,name):
 short=name.split('::')[-1]
 if suffix=='managers/test_mm_process_config.py' and short in ('test_top_level_non_dict_rejected','test_modality_non_dict_rejected_image','test_modality_non_dict_rejected_video','test_modality_non_dict_rejected_audio'):return CONFIG
 if suffix=='parser/test_jinja_template_utils.py' and short.startswith('test_detect_'):return FORMAT
 if suffix in family_units:return family_units[suffix]
 if suffix=='entrypoints/openai/test_whisper_adapter.py':
  if name.startswith('TestWhisperBuildFusedAutodetectParams::'):return PARAM
  if short in ('test_autodetect_codes_round_trip_through_input_validator','test_unknown_language_token_id_raises_clean_error'):return ADMIT
  return ASR
 if suffix=='parser/test_reasoning_parser.py':
  if name.startswith('TestMiniMaxAppendThinkDetector::'):return REASON
  if name.startswith('TestStreamingParseResult::'):return None
  return INIT
 return mapping.get((suffix,short))

results=[];entries=[];support=[];missing=[]
for r in pending:
 if r['snapshot']!='sglang' or r['review_status'] in ('property_reviewed','supporting_record_reviewed','outside_current_scope'):continue
 key=(r['path'],r['test_name'],int(r['line']))
 if key not in nodes:continue
 n=nodes[key];p=r['path'];suffix=p[len(BASE):];t=cache[p];unit=unit_for(suffix,r['test_name'])
 if unit is None:
  support.append({'snapshot':'sglang','path':p,'test_name':r['test_name'],'line':n.lineno,'role':'构造解析结果字段或检查 Python Protocol 结构相容性；不运行请求解析过程','source':ast.get_source_segment(t,n)});continue
 assertions=[];calls=[];inputs=[]
 for a in ast.walk(n):
  expr=ast.get_source_segment(t,a)
  if isinstance(a,ast.Assert):assertions.append({'line':a.lineno,'kind':'assert','expression':expr})
  elif isinstance(a,ast.Call):
   fn=ast.unparse(a.func);calls.append({'line':a.lineno,'target':fn,'expression':expr})
   if fn.split('.')[-1].startswith('assert') or fn.endswith('.raises'):
    assertions.append({'line':a.lineno,'kind':'assertion_call','expression':expr})
  elif isinstance(a,(ast.Assign,ast.AnnAssign)):inputs.append({'line':a.lineno,'expression':expr})
 assertions.sort(key=lambda x:x['line'])
 if not assertions:
  if n.name=='test_fused_params_survive_sampling_params_construction':
   a=next(x for x in calls if x['target']=='SamplingParams');assertions=[{'line':a['line'],'kind':'implicit_no_exception','expression':'SamplingParams 构造无异常：'+a['expression']}]
  else:missing.append({'path':p,'test_name':r['test_name'],'reason':'未提取显式检查，继续保留未确认'});continue
 unit_prop='；'.join(' '.join(a['expression'].split()) for a in assertions)
 result={k:r[k] for k in ['snapshot','path','test_name','line']}
 result.update(unit_id=unit,property='本项实际检查：'+unit_prop,oracle_source='; '.join(f'{p}:{a["line"]}' for a in assertions),conditions='使用测试体所构造的请求、文本、参数或模拟响应；完整输入、Mock 边界和辅助函数见 evidence。源码性质审查，未新增执行结果。',rule_id='round12_sglang_api_source_review_v1')
 results.append(result)
 entries.append({'snapshot':'sglang','path':p,'test_name':r['test_name'],'line':n.lineno,'unit_id':unit,'test_body_sha256':hashlib.sha256(ast.get_source_segment(t,n).encode()).hexdigest(),'assertions':assertions,'input_statements':inputs,'calls':calls,'test_source':ast.get_source_segment(t,n),'fixture_and_helper_sources':helpers[p],'implementation_files':['python/sglang/srt/'+x for x in sources[suffix]],'review_basis':'逐项阅读测试体及被测转换/解析实现，按本项实际断言归类；Mock 不视为真实模型执行。'})

fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
new={ASR,INIT,FORMAT,CONFIG}
for filename,items in [('crosswalk.tsv',results),('existing-units-crosswalk.tsv',[r for r in results if r['unit_id'] not in new]),('proposed-units-crosswalk.tsv',[r for r in results if r['unit_id'] in new])]:
 with (OUT/filename).open('w') as fp:
  w=csv.DictWriter(fp,fields,delimiter='\t');w.writeheader();w.writerows(items)
allpaths=set(cache)|{'python/sglang/srt/'+x for ss in sources.values() for x in ss}
reviews=[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()} for p in sorted(allpaths)]
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':reviews,'entries':entries,'supporting_records':support,'unconfirmed':missing,'count':len(results),'scope':'请求字段、API 适配、流式事件、多模态配置和推理/转写文本解析的源码检查。'},ensure_ascii=False,indent=2)+'\n')
proposals=[]
for uid,name,prop,observable,oracle,p in [
(ASR,'转写输出的语言识别与可见文本投影','按声明的转写前缀协议识别语言和完成状态，去除已知控制标记并保留普通文本；不完整或无效前缀不产生完整结果。',['language_code','visible_text','partial_prefix_status','control_token_set'],'构造含语言/任务/时间标记的解码字符串，检查语言、可见文本及不完整状态；同步核对语言表与约束表达式。','entrypoints/openai/transcription_adapters/whisper.py'),
(INIT,'推理解析器初始状态与请求配置','解析器根据模型标记约定、推理开关及上轮助手内容初始化状态，并传递流式与非空内容配置。',['start_end_markers','initial_reasoning_state','stream_reasoning','previous_content','force_nonempty_content'],'检查构造后的标记、状态和配置字段；文本分离结果由相应解析检查另外给出。','parser/reasoning_parser.py'),
(FORMAT,'提示模板内容表示选择','解析模板对消息内容的使用方式，选择结构化内容块或纯字符串输入表示；无效和无匹配模板按声明的默认表示处理。',['detected_content_format','template_parse_fallback'],'给定含循环、过滤器、切片等结构的模板文本，检查选择出的内容表示；不将格式选择等同于模板渲染正确。','parser/jinja_template_utils.py'),
(CONFIG,'预处理配置结构校验','服务启动配置中的多模态预处理设置采用顶层映射及逐模态映射；不符合声明结构时报告相应字段的类型错误。',['configuration_value','error_type','error_message'],'构造顶层和逐模态错误值，检查校验错误类型与消息。','server_args.py')]:
 path='python/sglang/srt/'+p
 tree=ast.parse((SRC/path).read_text());target={ASR:'parse_fused_output',INIT:'ReasoningParser',FORMAT:'detect_jinja_template_content_format',CONFIG:'_handle_multimodal'}[uid]
 anchor=next((n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==target),None)
 assert anchor,target
 proposals.append({'id':uid,'name':name,'kind':'obligation','phase':'preflight' if uid==CONFIG else 'ingress' if uid==FORMAT else 'output','state_scope':'deployment' if uid==CONFIG else 'request','conditions':{ASR:['transcription_projection_context'],INIT:['reasoning_parser_context'],FORMAT:['chat_template_profile'],CONFIG:['configuration_resolution_profile']}[uid],'property':prop,'observable':observable,'oracle':oracle,'framework_scope':['sglang'],'semantic_domain':'inference_request','evidence':{'sglang':[{'path':path,'line':anchor,'kind':'source','source_sha256':hashlib.sha256((SRC/path).read_bytes()).hexdigest()}]}})
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2)+'\n')
existing={u['id'] for u in json.loads((ROOT/'semantic_space/space_seed.json').read_text())['units']}
assert all(r['unit_id'] in existing|new for r in results),set(r['unit_id'] for r in results)-existing-new
assert len({(r['path'],r['test_name'],r['line']) for r in results})==len(results)
assert all(re.fullmatch(r'[^;]+:\d+(; [^;]+:\d+)*',r['oracle_source']) for r in results)
print(json.dumps({'count':len(results),'existing':sum(r['unit_id'] not in new for r in results),'proposed':sum(r['unit_id'] in new for r in results),'support':len(support),'unconfirmed':missing,'files':len(reviews),'by_unit':{u:sum(r['unit_id']==u for r in results) for u in sorted({r['unit_id'] for r in results})}},ensure_ascii=False,indent=2))

(OUT/'proposed-conditions.json').write_text(json.dumps([{'id':'transcription_projection_context','type':'structured','fields':['language_token_vocabulary','task_marker','timestamp_mode','prefix_completeness','decoded_output','known_control_markers','output_snapshot_sequence'],'observed_values':[],'constraint':'Fix the transcription prefix protocol and timestamp mode; distinguish incomplete prefixes, unknown language tokens, recognized control tokens and ordinary spoken text. Tests inspect decoded strings and declared language/regex tables, not acoustic recognition quality.'}],ensure_ascii=False,indent=2)+'\n')
