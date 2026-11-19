# -*- coding: utf-8 -*-
import ast,csv,hashlib,json
from pathlib import Path
O=Path(__file__).parent; R=O.parents[3]; P=R.parents[1]; B=P/'experiments/third_party/reference_repos/vllm-20261105'
rows=list(csv.DictReader((R/'semantic_space/test_mapping_pilot/direct-property-crosswalk.tsv').open(),delimiter='\t'))
rs=[r for r in rows if r['snapshot']=='vllm' and r['unit_id'] in ['output.parser_format_selection','output.structured_marker_boundary']]
fixed=[];support=[];entries=[];paths=set()
for r in rs:
 p=r['path'];s=(B/p).read_text();paths.add(p);tree=ast.parse(s);n=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef) and n.lineno==int(r['line']));body='\n'.join(s.splitlines()[n.lineno-1:n.end_lineno]);old=r['unit_id'];u=old;ln=int(r['line'])
 if p.endswith('test_parser_engine.py') and ln in [882,888]:u=None
 elif p.endswith('test_engine.py') and ln in [797,985,993]:u=None
 elif p.endswith('test_chat_parsing.py') and (old=='output.parser_format_selection' or ln in [1316,1441]):u='output.parser_input_contract'
 elif old=='output.parser_format_selection':
  if p.endswith('test_parser_engine.py') and ln in [1002,1039,1057,1085]:u='deployment.parser_selection'
  elif p.endswith('test_inkling.py'):u='output.structured_marker_boundary'
  else:u='response.reasoning_parser_initial_state'
 assertions=[{'line':a.lineno,'expression':ast.unparse(a)} for a in ast.walk(n) if isinstance(a,ast.Assert) or isinstance(a,ast.Call) and ast.unparse(a.func).split('.')[-1].startswith('assert')]
 ev={'snapshot':'vllm','path':p,'test_name':r['test_name'],'line':r['line'],'old_unit_id':old,'unit_id':u,'test_source':body,'test_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'assertions':assertions,'review_basis':'核查实际构造与断言后拆分初始化状态、解析器装配、输入规范拒绝和标记边界。'};entries.append(ev)
 if u is None:
  support.append({**r,'reason':'只检查局部布尔表达式或终结符配置结构，不执行解析输入路径。'});continue
 rr=dict(r);rr['unit_id']=u;rr['property']='本项实际检查：'+'；'.join(a['expression'] for a in assertions);rr['oracle_source']='; '.join(f'{p}:{a["line"]}' for a in assertions);rr['rule_id']='round15_vllm_parser_semantic_correction';fixed.append(rr)
fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
with (O/'corrections.tsv').open('w') as f:w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(fixed)
srcs={'output.structured_marker_boundary':[('vllm/parser/engine/streaming_parser_engine.py',226),('vllm/parser/engine/streaming_parser_engine.py',361),('vllm/parser/engine/parser_engine.py',733),('vllm/parser/chat_parsing/response_parser.py',185)],'response.reasoning_parser_initial_state':[('vllm/parser/qwen3.py',352),('vllm/parser/deepseek_v4.py',210),('vllm/parser/engine/parser_engine.py',119)],'deployment.parser_selection':[('vllm/parser/parser_manager.py',98),('vllm/parser/parser_manager.py',152)],'output.parser_input_contract':[('vllm/parser/chat_parsing/response_templates.py',129),('vllm/parser/chat_parsing/response_templates.py',147),('vllm/parser/chat_parsing/response_templates.py',219),('vllm/parser/chat_parsing/response_parser.py',137),('vllm/parser/chat_parsing/response_parser.py',211)]}
updates=[]
for uid,anchors in srcs.items():
 paths.update(p for p,l in anchors);updates.append({'unit_id':uid,'add_framework_scope':['vllm'],'add_evidence':{'vllm':[{'path':p,'line':l,'kind':'source'}for p,l in anchors]}})
new={'id':'output.parser_input_contract','name':'解析配置与必需输入校验','kind':'obligation','phase':'output','state_scope':'request','conditions':['output_parse_profile'],'property':'解析器检查模板字段与版本的支持范围、必需的提示前缀和结果字段，按声明规则拒绝不完整或不合法输入。','observable':['exception_type','exception_message','required_fields'],'oracle':'构造缺失前缀、无锚点模板、非法变换和缺失必需输出字段，检查异常类型与提示；具体条件按每项断言记录。','framework_scope':['vllm'],'evidence':updates[-1]['add_evidence']}
(O/'proposed-units.json').write_text(json.dumps([new],ensure_ascii=False,indent=2));(O/'unit-evidence-updates.json').write_text(json.dumps(updates[:-1],ensure_ascii=False,indent=2));(O/'supporting-reclassifications.json').write_text(json.dumps(support,ensure_ascii=False,indent=2));(O/'evidence.json').write_text(json.dumps({'file_reviews':[{'snapshot':'vllm','path':p,'source_sha256':hashlib.sha256((B/p).read_bytes()).hexdigest()}for p in sorted(paths)],'entries':entries},ensure_ascii=False,indent=2));(O/'notes.md').write_text('旧解析映射修正，非新增入口。86 项中 81 项保留性质归类并按实际语义改正单元，5 项改为辅助结构检查。25 项原 format_selection 没有一项应直接解释为按输入自动选格式：9 项是初始化状态、4 项是解析器装配、8 项是输入规范校验、2 项是提示前缀解析、2 项仅检查本地布尔表达式。61 项原 marker_boundary 中 3 项是终结符配置结构、2 项是输入规范校验，56 项保留边界归类。共享单元补充 vLLM 源码锚点，保留 SGLang 现有证据。本轮未运行上游测试。\n');print(len(fixed),len(support))
