"""Prepare source-anchored candidates for round eight; never marks them green."""

import ast
import csv
import collections
import hashlib
import json
import re
from pathlib import Path

BASE=Path(__file__).resolve().parents[3]
HERE=BASE/'semantic_space/test_mapping_pilot'
REPOS=BASE/'../../experiments/third_party/reference_repos'
ROOTS={'vllm':REPOS/'vllm-20261105','sglang':REPOS/'sglang-v0.5.14-complete'}

def rows(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f,delimiter='\t'))

GREEN=set()
for name in ['confirmed-property-crosswalk.tsv','direct-property-crosswalk.tsv','inherited-property-crosswalk.tsv']:
    GREEN.update((r['snapshot'],r['path'],r['test_name'],r['line']) for r in rows(HERE/name))
GREEN.update((r['snapshot'],r['path'],r['test_name'],r['line']) for r in rows(BASE/'semantic_space/oracle-review-queue.tsv') if r['source_review_status']=='mapped' and r['source_reviewed_units'])

def family(s,p):
    if s=='vllm':
        for prefix,name in [('tests/kernels/','kernel'),('tests/models/','model'),('tests/model_executor/','executor'),('tests/quantization/','quant'),('tests/parser/','parser'),('tests/tool_parsers/','tool_parser'),('tests/reasoning/','reasoning'),('tests/renderers/','renderer'),('tests/multimodal/','multimodal'),('tests/v1/kv_connector/unit/','connector'),('tests/v1/kv_offload/','offload'),('tests/v1/spec_decode/','spec'),('tests/v1/core/','core'),('tests/v1/sample/','sample'),('tests/v1/attention/','attention'),('tests/v1/engine/','engine'),('tests/entrypoints/openai/','entrypoints'),('tests/tokenizers_/','tokenizer'),('tests/watermarking/','watermark'),('tests/distributed/','distributed'),('tests/config/','config'),('tests/lora/','lora')]:
            if p.startswith(prefix):return name
    if s=='sglang':
        for prefix,name in [('test/registered/unit/mem_cache/','cache'),('test/registered/unit/function_call/','function_call'),('test/registered/unit/sampling/','sampling'),('test/registered/unit/constrained/','constrained'),('test/registered/unit/entrypoints/','entrypoints'),('test/registered/unit/layers/','layers'),('test/registered/unit/spec/','spec'),('test/registered/unit/model_executor/','executor'),('test/registered/unit/disaggregation/','connector'),('test/registered/unit/parser/','parser'),('test/registered/unit/managers/','managers')]:
            if p.startswith(prefix):return name
    return None

def checks(node):
    result=[]
    class Walk(ast.NodeVisitor):
        def visit_FunctionDef(self,n):
            if n is node:
                for x in n.body:self.visit(x)
        visit_AsyncFunctionDef=visit_FunctionDef
        def visit_Assert(self,n):result.append((n.lineno,ast.unparse(n)))
        def visit_Call(self,n):
            if isinstance(n.func,ast.Attribute) and n.func.attr.startswith('assert'):result.append((n.lineno,ast.unparse(n)))
            self.generic_visit(n)
        def visit_With(self,n):
            for x in n.items:
                y=x.context_expr
                if isinstance(y,ast.Call) and isinstance(y.func,ast.Attribute) and y.func.attr=='raises':result.append((n.lineno,ast.unparse(y)))
            self.generic_visit(n)
    Walk().visit(node)
    return sorted(set(result))

def classify(r,cs):
    s,p,n,o=r['snapshot'],r['path'],r['test_name'].lower(),r['object_candidates']
    f=family(s,p)
    if f is None:return None
    expressions=' '.join(expr for _,expr in cs).lower()
    if not expressions:return None
    if o in {'diagnostics','quality','runtime_support','image_generation','security','data_structure','memory','service'}:return None
    if any(k in p for k in ('/test_metrics.py','/test_stats.py','/test_gpu_ipc_memory.py','/test_gpu_worker.py','/test_adaptive_spec_params.py','/test_backpressure.py','/test_file_mapper.py')) or any(k in n for k in ('metrics','stats','_logs_','_latency_','_timing_','warning','_benchmark_')):return None
    if f=='entrypoints' and 'whisper_adapter' in p:return None
    if f in {'parser','tool_parser','reasoning','function_call'}:
        if any(k in n for k in ('registered','registration','manager_selects','parser_selection','registry')):return 'output.parser_format_selection'
        if any(k in n for k in ('reasoning_token_count','count_reasoning_tokens')):return 'response.usage_accounting'
    if 'pytest.raises' in expressions or 'raises(' in expressions:
        if any(k in n for k in ('invalid','reject','unsupported','error','unavailable','raises','fails')):
            if o=='configuration' or f in {'config','quant','lora'}:return 'deployment.configuration_admissibility'
    if o=='model':
        numeric=bool(re.search(r'assert_close|allclose|assert_equal|assert_array_equal|torch\.equal|np\.allclose',expressions))
        reference=bool(re.search(r'\b(expected|reference|ref_|baseline|target|ground_truth)\b',expressions)) and bool(re.search(r'\b(output|result|actual|out|logits|hidden|scores|y)\b',expressions))
        if not (numeric or reference):return None
        if any(k in n for k in ('routing_rows','partition','rank_mapping')):return 'distributed.partition_binding'
        if any(k in n for k in ('load_weight','weight_load','loaded_weight','parameter_binding')) and f in {'executor','model'}:return 'model.parameter_binding'
        if 'cached_weights' in n:return 'model.parameter_binding'
        if f=='quant':return 'model.execution_variant_relation'
        if f in {'kernel','attention','layers'} or f=='executor' and '/layers/' in p:return 'compute.operator_reference_relation'
        if f=='model':return 'model.reference_relation'
        if f=='executor':return 'model.execution_variant_relation'
        if f=='distributed':return 'distributed.partition_binding'
        return None
    if o=='cache' or o.startswith('cache,'):
        if 'registry' in p or 'factory' in p:return 'cache.eviction_policy_configuration'
        if 'event' in n:return 'cache.event_payload_binding'
        if any(k in n for k in ('config','_default_')):return 'deployment.effective_configuration'
        if 'hash' in n or 'key' in n or 'salt' in n:return 'cache.namespace'
        if any(k in n for k in ('hit','match','prefix')):return 'cache.hit_accounting'
        if any(k in n for k in ('evict','lru','priority')):return 'cache.eviction_outcome'
        if any(k in n for k in ('release','retain','pin','ref','free','cleanup','inflight','pending')):return 'cache.lifetime'
        if any(k in n for k in ('transfer','load','poll','async')):return 'disagg.transfer_gate'
        return 'cache.state_invariants'
    if o=='transfer' or o.startswith('transfer,'):
        if 'remote_prefill_lifecycle' in p:return 'disagg.logical_continuation'
        if any(k in n for k in ('config','_default_')):return 'deployment.effective_configuration'
        if any(k in n for k in ('request_id','req_id','request_binding')):return 'disagg.request_binding'
        if any(k in n for k in ('layout','shard','group','block','mapping')):return 'disagg.kv_mapping'
        if any(k in n for k in ('address','endpoint','socket','_port_')):return 'distributed.remote_endpoint_binding'
        return 'disagg.transfer_gate'
    if o=='output' or o.startswith('output,'):
        if f=='renderer':
            if any(k in p for k in ('test_token_offsets','test_completions')):return None
            return 'input.chat_template_tokenization'
        if f=='reasoning' or 'reasoning' in n:return 'output.channel_event_relation'
        if f in {'parser','tool_parser','function_call'} and any(k in n for k in ('schema_type','coerce_schema','type_conversion')):return 'output.tool_argument_type_binding'
        if f in {'parser','tool_parser','function_call'} and any(k in n for k in ('adjust_request','grammar','tool_choice','schema')):return 'tool.constraint_schema_construction'
        if f in {'tool_parser','function_call'} or 'tool' in n:return 'tool.call_parse_relation'
        if f=='parser':return 'output.incremental_parse_relation'
        if 'stream' in n or 'chunk' in n:return 'response.stream_payload_consistency'
        return 'response.delivery'
    if o=='ingress' or o.startswith('ingress,'):
        if f=='multimodal' and ('/test_cache.py' in p or '/test_gpu_ipc_memory.py' in p):return None
        if f=='multimodal' or any(k in n for k in ('image','audio','video')):return 'input.multimodal_binding'
        if f=='renderer':return 'input.chat_template_tokenization'
        if f=='tokenizer':return 'input.text_tokenization'
        if f=='entrypoints' or f=='parser':return 'request.admission'
        return None
    if o=='sampling' or o.startswith('sampling,'):
        if f=='spec' and ('test_ngram_corpus.py' in p or p.endswith('/test_ngram.py')):return 'spec.draft_candidate_binding'
        if f=='sample' and any(k in n for k in ('head_dtype','projection','logits_shape')):return 'compute.operator_reference_relation'
        if f=='spec':return 'spec.output_relation'
        if f=='constrained':return 'sampling.structured_constraint'
        if f=='watermark':return 'sampling.policy'
        return 'sampling.policy'
    if o=='configuration' or o.startswith('configuration,'):
        if any(k in n for k in ('reject','invalid','error','unsupported','raises')):return 'deployment.configuration_admissibility'
        return 'deployment.effective_configuration'
    if o=='adapter' or o.startswith('adapter,'):return 'model.adapter_binding'
    if o=='distributed' or o.startswith('distributed,'):return 'distributed.partition_binding'
    if o=='scheduling' or o.startswith('scheduling,'):return 'schedule.progress'
    return None

by=collections.defaultdict(list)
for r in rows(HERE/'full-scan-index.tsv'):
    if (r['snapshot'],r['path'],r['test_name'],r['line']) in GREEN or r['entry_kind']!='python_test_definition' or r['oracle_sources'] not in ('direct','mixed') or family(r['snapshot'],r['path']) is None:continue
    by[(r['snapshot'],r['path'])].append(r)
accepted=[];missing=0
for key,rs in by.items():
    src=ROOTS[key[0]]/key[1]
    if not src.is_file():missing+=len(rs);continue
    try:tree=ast.parse(src.read_text(encoding='utf-8'))
    except:missing+=len(rs);continue
    nodes={f'{outer.name}::{n.name}':n for outer in tree.body if isinstance(outer,ast.ClassDef) for n in outer.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
    nodes.update({n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))})
    for r in rs:
        n=nodes.get(r['test_name'])
        if n is None or str(n.lineno)!=r['line']:missing+=1;continue
        cs=checks(n);unit=classify(r,cs)
        if unit:accepted.append((r,unit,cs))

def substantive(checks):
    pattern=r'(?:^assert\b.*(?:==|!=|<=|>=|<|>)|assert_close\(|allclose\(|torch\.equal\(|assert_array_equal\(|assertEqual\(|assertAlmostEqual\(|assertArrayEqual\(|assert_called_once_with\(|assert_called_with\(|assertCountEqual\(|assertIn\(|assertNotIn\(|assertRaisesRegex\(|raises\([^)]*match=)'
    return any(re.search(pattern,expr,re.I) for _,expr in checks)


if __name__=='__main__':
    # These are proposals for audit. None become green through this script.
    excluded_families={'distributed','executor','core','managers'}
    assert len(GREEN)==5330
    candidates=[(r,u,cs) for r,u,cs in accepted if family(r['snapshot'],r['path']) not in excluded_families and r['scan_route']!='reviewed_out_of_request_scope' and substantive(cs)]
    keys=[(r['snapshot'],r['path'],r['test_name'],r['line']) for r,_,_ in candidates]
    assert len(keys)==len(set(keys)) and not (set(keys)&GREEN)
    hashes={key:hashlib.sha256((ROOTS[key[0]]/key[1]).read_bytes()).hexdigest() for key in {(r['snapshot'],r['path']) for r,_,_ in candidates}}
    output=Path(__file__).with_name('candidates.tsv')
    with output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','family','proposed_unit','source_sha256','oracle_lines','checks','status'],delimiter='\t',lineterminator='\n')
        writer.writeheader()
        for r,u,cs in sorted(candidates,key=lambda x:(x[0]['snapshot'],x[0]['path'],int(x[0]['line']))):
            writer.writerow({'snapshot':r['snapshot'],'path':r['path'],'test_name':r['test_name'],'line':r['line'],'family':family(r['snapshot'],r['path']),'proposed_unit':u,'source_sha256':hashes[(r['snapshot'],r['path'])],'oracle_lines':';'.join(str(line) for line,_ in cs),'checks':' || '.join(expr.replace('\n',' ') for _,expr in cs),'status':'candidate_only'})
    summary={'candidates':len(candidates),'files':len(hashes),'missing_source_or_test':missing,'families':dict(collections.Counter(family(r['snapshot'],r['path']) for r,_,_ in candidates)),'units':dict(collections.Counter(u for _,u,_ in candidates)),'green_before':len(GREEN),'source_commits':{'vllm':'76a0ccc209343971c6f4b56f6380a76202428385','sglang':'49e384ce9d304648e9959666ecb8ce8cd98d0deb'},'output':str(output)}
    Path(__file__).with_name('candidate-summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(summary,ensure_ascii=False))
