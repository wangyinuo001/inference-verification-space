# -*- coding: utf-8 -*-
"""Explicit human source review manifest; AST only transcribes already-read checks.

Both complete test files, all local fixture helpers, and both implementation
modules were read before authoring CASES. No discovery/name heuristic grants review.
Four assertion-free smoke entries remain pending. Re-running preserves old rows.
"""
import ast, csv, hashlib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
SOURCE = ROOT / "experiments/third_party/reference_repos/sglang-v0.5.14-complete"
BASE = "test/registered/unit/constrained/test_base_grammar_backend.py"
MGR = "test/registered/unit/constrained/test_grammar_manager.py"
IMPL = "python/sglang/srt/constrained/"
COLS = "snapshot path test_name line unit_id property oracle_source conditions rule_id".split()
# Explicit source-line manifest: unit suffix and actual checked case semantics.
CASES = {
BASE: {
39:("diagnostics", "Fresh GrammarStats exposes None timing/schema/size/dispatch fields, false hit/abort flags, empty traversal timings and zero timeouts."),
50:("diagnostics", "Appending 0.1 to one instance traversal timings leaves the other instance list empty."),
61:("object_state", "Fresh abstract grammar reports is_terminated() false; no accepted-token behavior is tested."),
73:("compile_outcome", "Default invalid grammar carries the literal Unknown grammar error diagnostic."),
77:("compile_outcome", "Invalid grammar preserves the supplied Regex compilation failed diagnostic."),
91:("compilation_cache", "Inserting JSON schema key stores the identical grammar object under that key."),
98:("compilation_cache", "Reset removes the previously inserted schema cache entry."),
103:("compilation_cache", "Cache hit reports true, invokes one copy and returns that copy rather than the stored object."),
117:("compilation_cache", "Cache-hit copied object receives maybe_init_reasoning(True)."),
127:("compilation_cache", "Uncached JSON schema returns hit=false and a Future whose base backend result is InvalidGrammarObject within five seconds."),
136:("compile_outcome", "Base dispatch for JSON, regex, EBNF and structural tag returns InvalidGrammarObject for the four explicit arguments."),
149:("compile_outcome", "Fallback dispatch of unknown/value raises ValueError."),
153:("compile_outcome", "Each of four grammar-type dispatch inputs reaches an unsupported base result; no per-handler mock call assertion exists."),
166:("compile_outcome", "Dispatching unknown_type/value raises ValueError."),
170:("diagnostics", "Mock JSON compiler with GrammarStats acquires non-None strictly positive compilation_time."),
188:("compilation_cache", "The same schema changes from cache hit to miss/Future after reset."),
203:("compile_outcome", "Fallback ValueError contains both custom_type and custom_value."),
210:("compile_outcome", "JSON compiler mocked to return None yields None without assuming a stats object."),
216:("compilation_cache", "Two same-key misses return distinct Futures, false hit flags and independently complete with invalid base grammar objects."),
249:("backend_selection", "Registering my_backend inserts its name into the registry; invocation is not asserted."),
254:("backend_selection", "Second registration of dup replaces first initializer; direct registry invocation returns second."),
283:("backend_selection", "Factory with grammar_backend=none and strict thinking disabled returns None."),
288:("backend_selection", "none backend plus strict thinking raises ValueError matching enable-strict-thinking."),
293:("backend_selection", "Unknown backend name is rejected with ValueError."),
298:("backend_selection", "Custom registered factory returns the registered mock object identity."),
305:("backend_selection", "Custom initializer receives tokenizer my_tok, vocab_size 50000 and EOS set {0,2}; server_args receipt is recorded but not asserted."),
322:("backend_selection", "Custom registry backend returns directly despite reasoning_parser=deepseek-r1."),
335:("backend_selection", "Outlines factory forwards tok and whitespace pattern and returns constructor result identity."),
346:("backend_selection", "XGrammar factory forwards tokenizer, vocabulary 32000, EOS [1,2] and any_whitespace=False and returns constructor result."),
359:("backend_selection", "Mock XGrammar unsupported-tokenizer exception yields None and changes grammar_backend to none."),
372:("backend_selection", "Guidance factory forwards tokenizer, any_whitespace=True and whitespace pattern and returns constructor result."),
386:("backend_selection", "Builtin Outlines with deepseek-r1 and think_end_id=42 is wrapped in ReasonerGrammarBackend containing the mock backend."),
405:("backend_selection", "Missing think_end_id prevents wrapping despite configured reasoning parser."),
416:("backend_selection", "Missing reasoning parser prevents wrapping despite think_end_id=42."),
427:("backend_selection", "XGrammar constructor receives model_eos_token_ids=None when factory EOS argument is None."),
},
MGR: {
80:("backend_selection", "Mock factory backend is installed and initial grammar queue length is zero."),
89:("backend_selection", "skip_tokenizer_init=True produces no grammar backend."),
95:("wait_queue", "Fresh manager has zero queue length and no waiting grammar."),
103:("compilation_cache", "Manager clear calls backend reset exactly once."),
131:("wait_queue", "Unconstrained request is not enqueued and returns false."),
138:("request_dispatch", "JSON schema miss enqueues one request, returns true and binds key (json, object schema)."),
150:("request_dispatch", "Regex miss returns true and binds key (regex, [a-z]+)."),
161:("request_dispatch", "EBNF miss returns true and binds key (ebnf, root hello rule)."),
172:("request_dispatch", "Structural tag miss returns true and binds the explicit structures/triggers JSON key."),
186:("wait_queue", "Cache-hit grammar is bound by identity to request; no queue insertion and return false."),
202:("request_abort", "Cached invalid grammar returns false and aborts once with bad schema in reason."),
215:("request_abort", "Constrained request without backend returns false and aborts once with not supported reason."),
229:("request_dispatch", "When both JSON schema and regex are populated, grammar_key is the JSON schema key."),
239:("request_dispatch", "Backend lookup receives JSON schema key and require_reasoning=True exactly once."),
256:("wait_queue", "A schema miss changes has_waiting_grammars from false to true and len(manager) to one."),
267:("reasoning_budget", "Cache-hit ReasonerGrammarObject receives request thinking_budget=7 in place of initial 99."),
285:("reasoning_budget", "Strict unconstrained reasoning request binds returned grammar identity and applies thinking_budget=3 in place of 99."),
311:("request_abort", "Matching request-id abort cancels its Future once and marks request aborted once."),
326:("request_abort", "Nonmatching req-999 is not marked aborted for abort prefix req-123."),
339:("request_abort", "abort_all marks all three queued requests aborted once each."),
364:("request_abort", "Abort prefix req-123 matches queued req-123-suffix and marks it aborted once."),
392:("wait_queue", "Completed Future returns its request and grammar identity, populates cache once and removes queue item."),
414:("request_abort", "Completed InvalidGrammarObject returns one request and aborts it once with compile error reason."),
433:("wait_queue", "Already-finished request with grammar=None is returned and removed from queue."),
445:("request_abort", "Incomplete Future with max_poll_iterations=1 times out, returns one request, aborts with timed out reason and caches InvalidGrammarObject for its key."),
468:("wait_queue", "Incomplete Future below max=100 stays queued, returns no request and increments wait count to one."),
485:("wait_queue", "Mixed complete/incomplete futures return only ready request and retain only pending request by identity."),
513:("wait_queue", "Empty queue produces no ready requests and remains empty."),
520:("request_abort", "Wait count two with maximum three reaches timeout on next poll and aborts once with timed out reason."),
538:("request_abort", "Future RuntimeError becomes invalid grammar on returned request and invokes abort once."),
555:("reasoning_budget", "Completed grammar applies request budget=4 while cached copy retains original budget=99 and original schema key."),
578:("rank_readiness", "Mock two-rank readiness sets {0} and {0,1} return request zero and retain request one; this case has no failed set."),
618:("rank_readiness", "Mock two-rank failure sets {0} and empty propagate timeout abort and remove request from queue."),
658:("request_dispatch", "Strict unconstrained require_reasoning=True binds strict grammar, returns false and calls strict initializer once with True."),
672:("request_dispatch", "Strict unconstrained require_reasoning=False binds strict grammar and calls strict initializer once with False."),
685:("request_dispatch", "Strict initializer returning None leaves request grammar None."),
696:("request_dispatch", "Strict constrained JSON request takes queued normal dispatch and never calls strict initializer."),
709:("request_dispatch", "Disabled strict-thinking path leaves unconstrained grammar None and does not call strict initializer."),
721:("request_abort", "Strict manager Future RuntimeError becomes invalid grammar on one returned request and aborts once."),
}}
UNITS = {
"diagnostics": ("语法编译诊断字段记录", "语法统计实例的默认值、可变字段隔离与编译耗时记录符合各字段声明。", "diagnostic", "diagnostic_artifact"),
"object_state": ("语法对象终止状态", "给定语法对象状态，终止谓词对应该对象声明的状态。", "decode", "request"),
"compile_outcome": ("语法编译分派与错误结果", "按指定语法类型分派编译；不支持、未知类型和失败输入返回声明的无效对象、空结果或异常并保留错误信息。", "admission", "request"),
"compilation_cache": ("语法编译缓存与副本隔离", "语法键确定缓存命中；命中返回声明的请求副本并初始化推理状态，未命中返回编译任务，重置失效旧条目。", "admission", "process"),
"backend_selection": ("约束后端选择与参数绑定", "配置、注册表和 tokenizer 能力确定约束后端及推理包装；初始化参数按声明传递，不支持组合被拒绝或按策略降级。", "preflight", "deployment"),
"request_dispatch": ("请求约束键与初始化路径", "请求约束字段和推理标志确定语法键、普通编译或严格推理初始化路径，所得语法与请求对应。", "admission", "request"),
"wait_queue": ("语法等待队列就绪迁移", "缓存、编译 Future 和请求完成状态确定等待、返回与移出队列的请求身份，未就绪请求保留并累积轮询次数。", "admission", "scheduler"),
"request_abort": ("语法编译失败与取消传播", "无可用后端、编译失败、超时或取消条件按请求身份传播终止状态和原因；超时记录无效编译结果。", "admission", "request"),
"reasoning_budget": ("请求推理预算与缓存隔离", "请求指定的推理预算应用于其语法对象；请求级修改不污染共享编译缓存副本。", "admission", "request"),
"rank_readiness": ("多秩语法就绪与失败同步", "按参与秩就绪集合交集和失败集合并集决定请求的迁移与终止。", "admission", "scheduler"),
}

def main():
    if any(c.get('family') == 'grammar_lifecycle_explicit_source_review' for c in json.loads((HERE/'evidence.json').read_text()).get('reviewed_clusters', [])):
        print('Cumulative review already recorded; no overwrite.'); return
    pending=list(csv.DictReader((HERE/'pending.tsv').open(),delimiter='\t'))
    outpath=HERE/'proposed-crosswalk.tsv'
    old=list(csv.DictReader(outpath.open(),delimiter='\t'))
    key=lambda r:tuple(r[c] for c in ('snapshot','path','test_name','line','unit_id','oracle_source'))
    known={key(r) for r in old}
    audit=[];new=[];selected=set();unit_evidence={}
    for path,cases in CASES.items():
        data=(SOURCE/path).read_bytes();text=data.decode();tree=ast.parse(text)
        nodes={n.lineno:(c.name,n) for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
        for line,(suffix,meaning) in cases.items():
            cls,node=nodes[line];name=cls+'::'+node.name
            checks=sorted([n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith(('assert','assert_'))],key=lambda n:n.lineno)
            assert checks,(path,line)
            unit='constraint.'+suffix
            unit_evidence.setdefault(unit,[]).append({'path':path,'line':checks[0].lineno,'kind':'existing_oracle'})
            body=ast.get_source_segment(text,node)
            conditions=('Pinned SGLang 0.5.14; source-reviewed only, not executed. '+
              ('Base backend thread executor is real; factory constructors and grammar copies are mocked where declared. ' if path==BASE else 'Scheduler/request/backend objects are mocked by _make_scheduler/_make_req/_make_mgr; Futures and ReasonerGrammarObject are real where constructed; all_gather is mocked in rank cases. ')+
              f'Exact case inputs and helper bindings: {path}:{line}-{node.end_lineno}; test-file SHA256={hashlib.sha256(data).hexdigest()}; test-body SHA256={hashlib.sha256(body.encode()).hexdigest()}. No model output or real distributed execution is established.')
            for check in checks:
                excerpt=' '.join(ast.get_source_segment(text,check).split())
                r=dict(snapshot='sglang',path=path,test_name=name,line=str(line),unit_id=unit,property=meaning+' Separate observed predicate: '+excerpt,oracle_source=f'{path}:{check.lineno}',conditions=conditions,rule_id='sglang_grammar_lifecycle_explicit_source_review')
                if key(r) not in known:new.append(r);known.add(key(r))
            selected.add((path,name,str(line)))
            audit.append(dict(path=path,test_name=name,line=line,end_line=node.end_lineno,unit_id=unit,reviewed_semantics=meaning,source_body=body,oracle_lines=[n.lineno for n in checks]))
    with outpath.open('a',newline='') as f:csv.DictWriter(f,fieldnames=COLS,delimiter='\t').writerows(new)
    remaining=[r for r in pending if (r['path'],r['test_name'],r['line']) not in selected]
    with (HERE/'pending.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=pending[0].keys(),delimiter='\t');w.writeheader();w.writerows(remaining)
    units=json.loads((HERE/'units-ready.json').read_text());existing={u['id'] for u in units['units']}
    dim='grammar_compilation_context'
    if dim not in {d['id'] for d in units['condition_dimensions']}:
        units['condition_dimensions'].append(dict(id=dim,type='structured',fields=['constraint_type','schema_text','backend','cache_state','future_state','poll_count','poll_limit','request_id','reasoning_flag','thinking_budget','rank_ready_and_failed_sets'],observed_values=[],constraint='Bind concrete request/helper fixtures and mock boundaries. Source review does not demonstrate actual grammar compilation, GPU inference or rank communication.'))
    for suffix,(name,prop,phase,scope) in UNITS.items():
        uid='constraint.'+suffix
        if uid in existing:continue
        ev=unit_evidence[uid][:3]
        ev.append(dict(path=IMPL+('base_grammar_backend.py' if suffix in ['diagnostics','object_state','compile_outcome','compilation_cache','backend_selection'] else 'grammar_manager.py'),line={'diagnostics':30,'object_state':60,'compile_outcome':183,'compilation_cache':198,'backend_selection':240}.get(suffix,93),kind='source'))
        units['units'].append(dict(id=uid,name=name,kind='obligation',phase=phase,state_scope=scope,conditions=[dim],property=prop,observable=['exact_asserted_predicate','request_or_backend_state'],oracle='分别核对记录中的具体返回值、身份、状态、异常或 mock 调用；各断言独立保留，不将源码审查当作运行通过。',evidence={'sglang':ev},framework_scope=['sglang'],candidate_status='proposed; requires parent integration',semantic_domain='inference_request'))
    (HERE/'units-ready.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n')
    files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in [BASE,MGR,IMPL+'base_grammar_backend.py',IMPL+'grammar_manager.py',IMPL+'reasoner_grammar_backend.py']}
    (HERE/'grammar-lifecycle-review.json').write_text(json.dumps(dict(review_method='Complete source/helper read followed by explicit 74-case manifest; AST only transcribes predicates, no automatic semantic classification. Four assertion-free smoke definitions remain pending.',files_sha256=files,cases=audit),ensure_ascii=False,indent=2)+'\n')
    ev=json.loads((HERE/'evidence.json').read_text())
    if new:ev.setdefault('reviewed_clusters',[]).append(dict(family='grammar_lifecycle_explicit_source_review',new_mapping_rows=len(new),reviewed_entries=len(selected),files_sha256=files,review_method='Complete test/helper and implementation read; explicit per-case semantics and separately recorded predicate rows. Source classification only; no tests executed.',audit='grammar-lifecycle-review.json'))
    ev['total_pending_after_review']=len(remaining)
    (HERE/'evidence.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n')
    print({'new_rows':len(new),'reviewed_entries':len(selected),'removed_pending':len(pending)-len(remaining),'remaining':len(remaining)})
if __name__=='__main__':main()
