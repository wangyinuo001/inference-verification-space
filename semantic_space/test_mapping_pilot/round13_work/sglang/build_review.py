"""Export explicitly read SGLang unit-test/source families; not a classifier."""
import ast,csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; PROJECT=ROOT.parents[1]
SRC=PROJECT/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'; OUT=Path(__file__).parent
BASE='test/registered/unit/'
FIELD='request.field_binding'; ADMIT='request.declared_admission_contract'
BATCH='sampling.batch_state_transition'; ADAPT='spec.adaptive_budget_transition'
REG='deployment.extension_dispatch_contract'; PARSER='deployment.parser_selection'
LIFE='request.identity_state_lifecycle'; TEXT='input.prompt_text_rendering'
sources={
'sampling/test_penaltylib.py':['sampling/penaltylib/orchestrator.py','sampling/penaltylib/frequency_penalty.py','sampling/penaltylib/presence_penalty.py','sampling/penaltylib/min_new_tokens.py'],
'sampling/test_sampling_batch_info.py':['sampling/sampling_batch_info.py'],
'spec/test_spec_registry.py':['speculative/spec_registry.py','speculative/spec_info.py','arg_groups/speculative_hook.py'],
'spec/test_adaptive_spec_params.py':['speculative/adaptive_spec_params.py'],
'managers/test_template_manager.py':['managers/template_detection.py'],
'parser/test_code_completion_parser.py':['parser/code_completion_parser.py'],
'managers/test_tokenizer_manager_rid_cleanup.py':['managers/tokenizer_manager.py'],
'entrypoints/openai/test_serving_chat.py':['entrypoints/openai/serving_chat.py','entrypoints/openai/serving_base.py','entrypoints/openai/protocol.py','entrypoints/openai/encoding_dsv4.py','parser/reasoning_parser.py']}
contexts={
'sampling/test_penaltylib.py':'CPU，词表32；测试辅助函数构造含 penalty、min_new_tokens、stop_token_ids、EOS 的模拟请求及批次。检查 prepare/filter/merge/teardown 后的标记、字段或指定 logits；不把状态检查提升为采样输出分布验证。',
'sampling/test_sampling_batch_info.py':'CPU SamplingBatchInfo 与模拟 penalizer/grammar；输入批次、索引、标志和 Mock 调用见测试体。callback 的 called_once 只支持调用约束，不证明 grammar 生成的掩码数值。',
'spec/test_spec_registry.py':'隔离并恢复插件注册表；worker factory 返回 MagicMock。检查名称解析、注册冲突、接口、回调及配置传递，不执行 draft/target 模型。',
'spec/test_adaptive_spec_params.py':'以测试中候选步数、阈值、EMA、warmup、batch 路由和 CUDA Graph 批尺寸配置为条件，向控制器输入显式 acceptance 历史；不执行真实 GPU Graph 或验证推测输出分布。',
'managers/test_template_manager.py':'模板字符串、tokenizer 词表和 parser=auto 配置；部分方法声明加载 Qwen/Qwen3-0.6B tokenizer。仅源码核对，未在本轮下载或执行。规则次序检查只说明表项顺序。',
'parser/test_code_completion_parser.py':'已声明 FIM 前缀/后缀/中间标记及其位置、注册表与覆盖开关；检查提示串或注册状态，不执行补全模型。',
'managers/test_tokenizer_manager_rid_cleanup.py':'最小 TokenizerManager、模拟 scheduler 和 tokenization、真实本地 ReqState/event；构造 abort、finished/unfinished output 与派发前异常。检查 rid_to_state、finished/event 和重提请求，不覆盖跨进程调度竞态。',
'entrypoints/openai/test_serving_chat.py':'MockTokenizerManager/Tokenizer 及显式 ChatCompletionRequest；部分模板、消息处理和输出被 Mock。核对字段、模式选择、异常或 Mock 调用，不视为真实模型生成。'}
# Family membership below follows complete method/body and implementation reading.
def unit_for(s,n):
 short=n.split('::')[-1]
 if s=='sampling/test_penaltylib.py':return 'sampling.logit_transform_relation' if short=='test_blocks_custom_stop_tokens' else BATCH
 if s=='sampling/test_sampling_batch_info.py':return BATCH
 if s=='spec/test_adaptive_spec_params.py':return ADAPT
 if s=='spec/test_spec_registry.py':return 'deployment.effective_configuration' if short=='test_handle_speculative_decoding_invokes_custom_handle_server_args' else REG
 if s=='managers/test_template_manager.py':return PARSER
 if s=='managers/test_tokenizer_manager_rid_cleanup.py':
  if short=='test_abort_removes_rid_from_state':return 'abort.cleanup'
  if short=='test_abort_sets_finished_and_notifies':return 'abort.terminal_status'
  return LIFE
 if s=='parser/test_code_completion_parser.py':
  if n.startswith(('TestFimPosition::','TestCompletionTemplate::')):return None
  return TEXT if n.startswith(('TestGenerateCompletionPrompt::','TestGenerateCompletionPromptFromRequest::')) else REG
 if s=='entrypoints/openai/test_serving_chat.py':
  if short in ['test_convert_to_internal_request_rejects_stream_return_prompt_token_ids','test_convert_to_internal_request_rejects_stream_return_meta_info','test_dsv_encoders_accept_object_tool_call_arguments_string','test_dsv4_task_field_schema']:return ADMIT
  if short in ['test_dpsk_v32_encoding_path','test_encode_messages_returns_none_by_default']:return PARSER
  if short=='test_poolside_v1_does_not_double_prepend_think':return TEXT
  return FIELD
 raise AssertionError((s,n))

cache={}; nodes={}; helpers={}; contexts_source={}
def seg(t,n):return '\n'.join(t.splitlines()[n.lineno-1:n.end_lineno])
for suffix in sources:
 p=BASE+suffix;t=(SRC/p).read_text();cache[p]=t;tree=ast.parse(t);helpers[p]=[];contexts_source[p]=[]
 for top in tree.body:
  if isinstance(top,ast.ClassDef):
   items=[(top.name+'::'+x.name,x) for x in top.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))]
   for x in top.body:
    if isinstance(x,(ast.Assign,ast.AnnAssign)):contexts_source[p].append({'line':x.lineno,'source':seg(t,x)})
  elif isinstance(top,(ast.FunctionDef,ast.AsyncFunctionDef)):items=[(top.name,top)]
  else:
   if isinstance(top,(ast.Assign,ast.AnnAssign,ast.Import,ast.ImportFrom)):contexts_source[p].append({'line':top.lineno,'source':seg(t,top)})
   continue
  for name,n in items:
   if n.name.startswith('test'):nodes[(p,name,n.lineno)]=n
   else:helpers[p].append({'name':name,'line':n.lineno,'source':seg(t,n)})
rows=list(csv.DictReader((ROOT/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
results=[]; entries=[]; support=[]; missing=[]
implicit={
'TestBatchedFrequencyPenalizer::test_cumulate_when_not_prepared_is_noop':'pen.cumulate_output_tokens',
'TestBatchedPenalizerBase::test_filter_when_not_prepared_is_noop':'pen.filter',
'TestConformanceGuard::test_base_custom_spec_algo_conforms':'_assert_custom_spec_algo_conforms',
'TestConformanceGuard::test_conforming_subclass_passes':'_assert_custom_spec_algo_conforms',
'ServingChatTestCase::test_dsv_encoders_accept_object_tool_call_arguments_string':'self.chat._process_messages'}
for r in rows:
 if r['snapshot']!='sglang' or r['review_status'] in ['property_reviewed','supporting_record_reviewed','outside_current_scope']:continue
 key=(r['path'],r['test_name'],int(r['line']))
 if key not in nodes:continue
 p=r['path'];t=cache[p];n=nodes[key];s=p[len(BASE):];unit=unit_for(s,r['test_name']);body=seg(t,n)
 if unit is None:
  support.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'role':'枚举标识或数据容器字段的结构检查；不执行 FIM 提示转换','source':body});continue
 assertions=[];calls=[];inputs=[]
 for a in ast.walk(n):
  if isinstance(a,ast.Assert):assertions.append({'line':a.lineno,'kind':'assert','expression':ast.unparse(a)})
  elif isinstance(a,ast.Call):
   fn=ast.unparse(a.func);expr=ast.unparse(a);calls.append({'line':a.lineno,'target':fn,'expression':expr})
   if fn.split('.')[-1].startswith('assert') or fn.endswith('.raises'):
    assertions.append({'line':a.lineno,'kind':'assertion_call','expression':expr})
  elif isinstance(a,(ast.Assign,ast.AnnAssign,ast.For,ast.AsyncFor)):
   inputs.append({'line':a.lineno,'expression':ast.unparse(a)})
 if r['test_name'] in implicit:
  target=implicit[r['test_name']]
  candidates=[x for x in calls if x['target']==target] if target else [x for x in calls if 'encode_messages' in x['target']]
  for a in candidates:assertions.append({'line':a['line'],'kind':'implicit_no_exception','expression':'本项检查该调用完成且不抛异常：'+a['expression']})
 assertions.sort(key=lambda x:x['line'])
 if not assertions:
  missing.append({'path':p,'test_name':r['test_name'],'reason':'检查表达式未完整抽取，保留待核对','source':body});continue
 rr={k:r[k] for k in ['snapshot','path','test_name','line']}
 rr.update(unit_id=unit,property='本项实际检查：'+'；'.join(' '.join(a['expression'].split()) for a in assertions),oracle_source='; '.join(f'{p}:{a["line"]}' for a in assertions),conditions=contexts[s],rule_id='round13_sglang_state_and_dispatch_source_review_v1')
 results.append(rr)
 entries.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'unit_id':unit,'test_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'assertions':assertions,'input_statements':inputs,'calls':calls,'test_source':body,'fixture_and_helper_sources':helpers[p],'module_and_class_constants':contexts_source[p],'implementation_files':['python/sglang/srt/'+x for x in sources[s]],'review_basis':'逐项阅读本测试、共同输入辅助函数与被测实现，记录检查的字段/异常/数值或Mock调用，未运行测试。'})

new={BATCH,ADAPT,REG,PARSER,LIFE};fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
for file,rs in [('crosswalk.tsv',results),('existing-units-crosswalk.tsv',[r for r in results if r['unit_id'] not in new]),('proposed-units-crosswalk.tsv',[r for r in results if r['unit_id'] in new])]:
 with (OUT/file).open('w') as fp:w=csv.DictWriter(fp,fields,delimiter='\t');w.writeheader();w.writerows(rs)
paths=set(cache)|{'python/sglang/srt/'+x for xs in sources.values() for x in xs}
reviews=[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()} for p in sorted(paths)]
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':reviews,'entries':entries,'supporting_records':support,'unconfirmed':missing,'count':len(results),'scope':'采样批状态、推测预算控制、请求状态生命周期、插件/模板选择及请求适配。仅源码性质核对。'},ensure_ascii=False,indent=2)+'\n')

proposals=[]
def proposal(uid,name,phase,state,cond,prop,obs,oracle,anchors):
 ev=[]
 for p,target in anchors:
  p='python/sglang/srt/'+p;t=(SRC/p).read_text();tree=ast.parse(t)
  line=next(n.lineno for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==target)
  ev.append({'path':p,'line':line,'kind':'source','source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()})
 proposals.append({'id':uid,'name':name,'kind':'obligation','phase':phase,'state_scope':state,'conditions':[cond],'property':prop,'observable':obs,'oracle':oracle,'framework_scope':['sglang'],'semantic_domain':'inference_request','evidence':{'sglang':ev}})
proposal(BATCH,'采样批状态的初始化与成员变化','sampling','shared','sampling_batch_transition_context','采样配置、累计惩罚及约束状态随批次创建、过滤、合并和释放进行相应更新；由成员需求决定启用标志，操作保留声明的批次关联与回调约定。',['batch_members','required_flags','prepared_state','penalty_tensor_shape','grammar_callback','released_attributes'],'构造明确的批次成员和参数，核对每次操作后的字段、标志、形状、关联对象及回调实参；数值变换由独立 logits 检查给出。',[('sampling/penaltylib/orchestrator.py','BatchedPenalizerOrchestrator'),('sampling/sampling_batch_info.py','SamplingBatchInfo')])
proposal(ADAPT,'推测步数控制与批次路由','scheduling','shared','adaptive_speculation_context','推测步数由声明的候选集合、接受历史和控制参数更新；warmup、更新周期、阈值、滞回和上限约束生效，批尺寸和图配置决定对应控制槽及可用图集合。',['current_steps','ema_accept_len','warmup_counter','routed_slot','reachable_graph_sizes','configuration_error'],'输入固定历史序列，逐次核对步数、EMA、更新返回值和批次路由；检查非法候选配置被拒绝。',[('speculative/adaptive_spec_params.py','AdaptiveStepSlot'),('speculative/adaptive_spec_params.py','AdaptiveSpeculativeParams')])
proposal(REG,'扩展注册与执行入口选择','preflight','deployment','extension_registry_context','扩展名称按声明规范注册和解析；重复或保留名称按约定拒绝，覆盖行为显式启用，已注册对象经接口与能力检查绑定到工厂、模板或配置回调。',['resolved_object','registry_members','duplicate_error','factory_call','capability_rejection','callback_binding'],'检查名称到对象的对应、冲突异常、接口成员以及模拟工厂/回调的实参和返回值；模板注册另检查覆盖与单次设置约定。',[('speculative/spec_registry.py','register_algorithm'),('parser/code_completion_parser.py','register_completion_template')])
proposal(PARSER,'模型模板与解析方式选择','ingress','deployment','parser_selection_context','根据模板结构、词表标记、模型配置和显式解析开关选择对应解析或编码方式；按声明优先级处理重叠规则，并使用无匹配及加载失败的约定结果。',['reasoning_parser','tool_call_parser','chat_encoding_spec','rule_order','fallback_value'],'构造模板与词表组合，检查解析器/编码标识及规则优先级；显式配置和 auto 配置分别验证。',[('managers/template_detection.py','match_rules'),('managers/template_detection.py','resolve_auto_parsers'),('entrypoints/openai/serving_chat.py','_resolve_chat_encoding_spec')])
proposal(LIFE,'请求标识与活动状态生命周期','scheduling','shared','request_identity_lifecycle_context','活动请求标识与状态保持对应，重复活动标识按约定拒绝；完成、中止和派发前失败清除相应状态，使已终止标识可以重新使用，未完成请求继续保留状态。',['rid_to_state','finished_flag','notification_event','duplicate_error','resubmission_result'],'用同一请求标识串接创建、输出、失败或中止及再次提交，核对状态表成员和终止/通知字段；单条与批量清理分别检查。',[('managers/tokenizer_manager.py','_init_req_state'),('managers/tokenizer_manager.py','_discard_pending_req_states'),('managers/tokenizer_manager.py','_handle_batch_output')])
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2)+'\n')
conditions=[
('sampling_batch_transition_context',['request_members','sampling_parameters','penalty_history','filter_indices','grammar_states','prepared_state','operation'],'固定批次成员、采样参数和历史状态；区分数值断言、元数据断言与模拟回调检查。'),
('adaptive_speculation_context',['candidate_steps','initial_steps','accepted_token_history','ema_alpha','warmup_batches','update_interval','hysteresis_thresholds','ceiling','batch_size','cuda_graph_sizes'],'控制参数与接受历史共同定义步数演化；此维度不把预算控制视为输出分布等价性。'),
('extension_registry_context',['registry_snapshot','requested_name','reserved_names','case_policy','override_policy','factory','interface','capabilities','configuration_hook'],'隔离注册表与扩展对象；记录名称冲突、覆盖开关、能力约束和具体工厂/回调绑定。'),
('parser_selection_context',['template','tokenizer_vocabulary','rule_order','explicit_parser','auto_mode','model_architecture','tokenizer_load_outcome'],'按模板结构与词表签名选择解析方式，保留显式配置、默认配置和加载失败条件。'),
('request_identity_lifecycle_context',['request_ids','active_state_map','output_finished_reason','abort_event','dispatch_failure','single_or_batch','notification_state'],'区分活动、已结束和派发前失败的请求；同标识重提交必须以先前状态结束为前提。')]
(OUT/'proposed-conditions.json').write_text(json.dumps([{'id':i,'type':'structured','fields':f,'observed_values':[],'constraint':c} for i,f,c in conditions],ensure_ascii=False,indent=2)+'\n')
existing={u['id'] for u in json.loads((ROOT/'semantic_space/space_seed.json').read_text())['units']}
assert all(r['unit_id'] in existing|new for r in results),{r['unit_id'] for r in results}-existing-new
assert len({(r['path'],r['test_name'],r['line']) for r in results})==len(results)
assert all(re.fullmatch(r'[^;]+:\d+(; [^;]+:\d+)*',r['oracle_source']) for r in results)
print(json.dumps({'count':len(results),'existing':sum(r['unit_id'] not in new for r in results),'proposed':sum(r['unit_id'] in new for r in results),'support':len(support),'unconfirmed':[{'path':r['path'],'test_name':r['test_name']} for r in missing],'files':len(reviews),'by_unit':{u:sum(r['unit_id']==u for r in results) for u in sorted({r['unit_id'] for r in results})}},ensure_ascii=False,indent=2))
