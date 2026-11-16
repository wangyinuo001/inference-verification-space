from pathlib import Path
import json,ast,csv,hashlib
h=Path(__file__).resolve().parent;src=h.parents[5]/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
d=json.loads((h/'units-ready.json').read_text())
# Explicit source functions and property-specific observations/oracles after review.
updates={
'constraint.diagnostics':(['compilation_time','schema_count','ebnf_size','is_cache_hit','is_grammar_aborted','tree_traversal_time','dispatch_type','num_timeout'],'新实例字段等于断言的 None/False/空列表/0；修改一个实例的遍历列表不改变另一个；带统计对象的编译计时为非空正数。','base_grammar_backend.py',['GrammarStats','BaseGrammarBackend._init_value_dispatch']),
'constraint.object_state':(['is_terminated'],'构造基础语法对象后 is_terminated() 返回 False；本批尚无 token 推进后的终止状态检查。','base_grammar_backend.py',['BaseGrammarObject.is_terminated']),
'constraint.compile_outcome':(['dispatch_key_type','dispatch_key_string','grammar_result_type','error_message','raised_exception'],'用四种明确语法键检查基础分派返回 InvalidGrammarObject；未知键触发 ValueError 并保留键和值；无效对象保留默认或指定错误文本；mock 返回 None 时结果为 None。','base_grammar_backend.py',['InvalidGrammarObject.__init__','BaseGrammarBackend._init_value_dispatch','BaseGrammarBackend.dispatch_fallback']),
'constraint.compilation_cache':(['cache_key','cached_object_identity','returned_copy_identity','cache_hit','future_identity','reasoning_initialization_call'],'插入后键对应原对象；命中标志为真、copy 调用一次并返回副本，副本接收推理标志；重置删除旧条目并使后续查询未命中；重复未命中产生两个独立 Future。','base_grammar_backend.py',['BaseGrammarBackend.get_cached_or_future_value','BaseGrammarBackend.set_cache','BaseGrammarBackend.reset']),
'constraint.backend_selection':(['registry_initializer','selected_backend_identity','constructor_arguments','reasoner_wrapper_type','wrapped_backend_identity','effective_backend_name','raised_exception'],'将配置和注册函数的预期结果与后端身份、构造参数和推理包装对象逐项比较；禁用后端为 None；未知名称或 strict+none 报错；指定 tokenizer 不支持异常触发有效后端改为 none。','base_grammar_backend.py',['register_grammar_backend','create_grammar_backend']),
'constraint.request_dispatch':(['grammar_key','request_grammar_identity','enqueue_result','require_reasoning_argument','strict_initializer_calls'],'对请求 JSON/regex/EBNF/tag 字段检查实际 grammar_key；JSON 与 regex 同时指定时选择 JSON；后端收到请求推理标志；严格无约束请求绑定 strict initializer 结果，受约束请求走普通分派。','grammar_manager.py',['GrammarManager.process_req_with_grammar']),
'constraint.wait_queue':(['queued_request_ids','ready_request_ids','request_grammar_identity','grammar_wait_ct','queue_length','has_waiting_grammars','cache_set_calls'],'比较迁移前后的请求身份集合与队列长度：完成项返回并移出，未完成项保留并递增 wait_ct，已结束请求移出；缓存命中不入队，空队列返回空集合。','grammar_manager.py',['GrammarManager.get_ready_grammar_requests','GrammarManager.has_waiting_grammars','GrammarManager.process_req_with_grammar']),
'constraint.request_abort':(['request_id','future_cancel_calls','abort_calls','abort_reason','returned_grammar_type','timeout_cache_key','timeout_cache_result_type'],'按 abort_all 或 rid 前缀检查取消与终止调用；无后端、无效编译结果和 Future 异常产生对应终止原因；wait_ct 达上限时请求返回、终止且为原键缓存 InvalidGrammarObject。','grammar_manager.py',['GrammarManager.abort_requests','GrammarManager.get_ready_grammar_requests']),
'constraint.reasoning_budget':(['request_thinking_budget','request_grammar_max_think_tokens','cached_grammar_max_think_tokens','cache_key','request_grammar_identity'],'命中缓存、严格无约束及 Future 完成路径把请求预算 7/3/4 写入请求语法；Future 完成例缓存副本预算仍为 99，且缓存键保持原 schema。','grammar_manager.py',['GrammarManager._apply_request_reasoning_budget','GrammarManager.get_ready_grammar_requests']),
'constraint.rank_readiness':(['per_rank_ready_index_sets','per_rank_failed_index_sets','returned_request_ids','remaining_request_ids','abort_reason'],'两个 mock rank 的就绪集合 {0} 和 {0,1} 仅释放交集请求0并保留请求1；失败集合 {0} 与空集的并集使请求0按超时终止并移出。','grammar_manager.py',['GrammarManager.get_ready_grammar_requests']),
'diagnostic.weight_probe_generation':(['probe_dtype','probe_shape','probe_device','minimum_value','maximum_value','distinct_value_count','input_tensor_values'],'比较探针与输入元数据；浮点值在 [0,1)，整数位于 dtype 范围并有多个值，布尔样本含真和假；调用前后输入张量 assert_close。','weight_checker.py',['_random_like']),
'diagnostic.weight_comparison_projection':(['projected_tensor_names','projected_compare_flags','projected_tensor_values','projected_entry_order'],'对照显式预期三元组流逐项比较长度、名称、标志和 assert_close 数值；FP8 解量化项先于原始权重/尺度，原始量化项和排除缓冲区标志为 false；FP8 参考共用 block_quant_dequant。','weight_checker.py',['_postprocess_tensors','_is_non_persistent_buffer_name']),
'diagnostic.weight_comparison_verdict':(['expected_names','actual_names','compare_flags','stream_lengths','elementwise_equality','raised_exception','error_tensor_name','max_abs_error'],'名字/标志不一致触发 AssertionError，长度不一致触发 ValueError；compare=True 张量逐元素不等触发含名称和误差的异常；compare=False 差异及未变/恢复的模型通过同一异常门。','weight_checker.py',['_check_tensors','WeightChecker._compare']),
'diagnostic.weight_snapshot_isolation':(['snapshot_tensor_names','snapshot_tensor_devices','snapshot_tensor_values','live_tensor_values'],'快照名字集合等于两个参数加四个缓冲区；CUDA 模型快照移至 CPU；将 live w 填为99后快照 w 与原值 assert_close。','weight_checker.py',['WeightChecker._snapshot']),
'diagnostic.weight_probe_reset':(['parameter_storage_pointer','parameter_values_before_after','excluded_buffer_values_before_after'],'正常 w 的 data_ptr 不变且值改变；三个匹配排除模式的缓冲区在重置前后 assert_close。','weight_checker.py',['WeightChecker._reset_tensors']),
'diagnostic.weight_action_dispatch':(['action_name','handler_call_counts','returned_value','raised_exception_message'],'四个声明动作各调用对应 mock handler 一次；snapshot/compare 返回 None；未知动作异常包含 Unsupported。','weight_checker.py',['WeightChecker.handle']),
'diagnostic.weight_checksum_record':(['per_tensor_checksum','included_tensor_names','record_keys','parallelism_info','checksum_length','hex_parse_result','input_tensor_values'],'检查记录键和纳入名字集合、十六位可解析十六进制张量散列、fake runner 的 TP/DP/PP 字段；相同输入记录相同，指定权重修改使散列改变，散列调用不修改输入；Pydantic 验证嵌套拓扑类型。','weight_checker.py',['WeightChecker._compute_checksum','WeightChecker._parallelism_info','_hash_tensor']),
}
for u in d['units']:
 if u['id'] not in updates:continue
 obs,oracle,file,names=updates[u['id']];u['observable']=obs;u['oracle']=oracle
 if u['id']=='constraint.diagnostics':u['semantic_domain']='diagnostic_support'
 folder='utils/' if file=='weight_checker.py' else 'constrained/'
 path='python/sglang/srt/'+folder+file;tree=ast.parse((src/path).read_text());loc={}
 for node in tree.body:
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):loc[node.name]=node.lineno
  if isinstance(node,ast.ClassDef):
   for n in node.body:
    if isinstance(n,ast.FunctionDef):loc[node.name+'.'+n.name]=n.lineno
 ev=[e for e in u['evidence']['sglang'] if e['kind']!='source']
 u['evidence']['sglang']=[dict(path=path,line=loc[n],kind='source') for n in names]+ev
 if u['id']=='diagnostic.weight_comparison_projection':
  u['evidence']['sglang'] += [dict(path='test/registered/unit/utils/test_weight_checker.py',line=186,kind='existing_test'),dict(path='test/registered/unit/utils/test_weight_checker.py',line=241,kind='existing_test')]
(h/'units-ready.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
# Exact local helper links (avoid line guesses) in weight mapping conditions.
p=h/'proposed-crosswalk.tsv';rows=list(csv.DictReader(p.open(),delimiter='\t'))
wt=ast.parse((src/'test/registered/unit/utils/test_weight_checker.py').read_text());ln={n.name:n.lineno for n in wt.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
for r in rows:
 if r['rule_id']=='sglang_weight_checker_explicit_source_review':
  for name in ['_assert_triples_close','_build_fp8_quant_pair','_TinyModel','_FakeModelRunner','_WeightCheckerTestBase','_ChecksumTestBase']:
   import re
   r['conditions']=re.sub(r'\b'+re.escape(name)+r':\d+',name+':'+str(ln[name]),r['conditions'])
with p.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter='\t');w.writeheader();w.writerows(rows)
# Validate source refs/hash manifests and entry-key uniqueness for both new groups.
new=[r for r in rows if r['rule_id'] in ('sglang_grammar_lifecycle_explicit_source_review','sglang_weight_checker_explicit_source_review')]
assert len(new)==126
keys=[tuple(r[c] for c in ['snapshot','path','test_name','line']) for r in new];assert len(keys)==len(set(keys))
refs=0
for r in new:
 for ref in r['oracle_source'].split(';'):
  p,line=ref.rsplit(':',1);assert (src/p).read_text().splitlines()[int(line)-1].strip();refs+=1
for u in d['units']:
 if u['id'] not in updates:continue
 assert u['framework_scope']==['sglang']
 for e in u['evidence']['sglang']:assert (src/e['path']).read_text().splitlines()[e['line']-1].strip()
for name in ['grammar-lifecycle-review.json','weight-checker-review.json']:
 for p,digest in json.loads((h/name).read_text())['files_sha256'].items():assert hashlib.sha256((src/p).read_bytes()).hexdigest()==digest
report=dict(reviewed_entries=126,independent_predicate_references=refs,units=list(updates),source_hashes_verified=True,source_lines_nonempty=True,unique_entry_keys=True,framework_scope='sglang only',runtime_executed=False)
(h/'continuation-review-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(report)
