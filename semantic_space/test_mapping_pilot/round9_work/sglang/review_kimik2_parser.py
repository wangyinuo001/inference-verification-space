# -*- coding: utf-8 -*-
"""Explicit review of all 45 Kimi parser cases; helpers and implementations read."""
import ast,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[5]/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
PATH='test/registered/function_call/test_kimik2_detector.py'
RULE='sglang_kimik2_explicit_source_review'
COLS='snapshot path test_name line unit_id property oracle_source conditions rule_id'.split()
CASES={
81:'Complete ReadFile call yields one call, exact name/JSON argument string and empty normal text.',
96:'Two complete calls preserve ReadFile then get_weather order, count and Tokyo JSON argument string.',
114:'One parsed call preserves Let me check the file. prefix as normal text.',
128:'Plain response yields zero calls and the same normal text.',
135:'Section-begin token makes has_tool_call true; plain text makes it false; only marker detection is asserted.',
153:'Hyphenated mcp__portal__search-documents name survives nonstreaming parse.',
166:'Four chunks assemble one list-files call whose decoded path is /home.',
199:'Four chunks assemble one ReadFile call with exact JSON argument string for /test.py.',
215:'Sequential streamed calls preserve ReadFile then get_weather order and Paris parameters.',
234:'After complete streamed tool section, detector buffer is empty and current_tool_id is one.',
257:'Streaming normal text plus orphan section-end marker strips that marker and retains normal text.',
266:'Streaming orphan argument-begin marker does not leak into normal_text.',
274:'Direct _strip_special_tokens on leaked markers returns leakedcontent; the detector error fallback itself is not invoked.',
281:'Direct stripping of all five recognized marker strings yields empty text.',
292:'Direct stripping leaves ordinary Hello world text unchanged.',
306:'Closed think block extracts exact reasoning sentence and passes tool section marker in normal_text.',
321:'Unclosed think block splits at tool section marker: reasoning retains sentence without think/tool markers; normal_text retains both tool section and call markers.',
350:'Tool section without think block produces empty reasoning and preserves section marker in normal_text.',
364:'Plain Hello response is entirely normal text with empty reasoning.',
386:'Streaming closed think block retains reasoning sentence and routes tool marker only to normal text.',
401:'Streaming unclosed think block routes subsequent tool section into normal text; reasoning concatenates sentence without tool/think markers.',
432:'Tool marker and call data arriving together while reasoning route to normal text after retaining earlier thinking text.',
443:'Manually enabled reasoning state emits ordinary reasoning, buffers partial <|tool, then routes completed section marker to normal text.',
467:'Two ordinary chunks concatenate into normal output and yield empty reasoning.',
479:'force_reasoning=True treats initial untagged text as reasoning until closing think marker then emits answer as normal text.',
521:'Sequential reasoning and tool parsers preserve reasoning without special markers and emit one ReadFile name plus joined arguments containing the supplied file path.',
571:'Nonstreaming reasoning parser then tool parser preserves clean reasoning and yields one ReadFile with decoded path /src/main.py.',
599:'Closed-think streaming composition preserves Thinking about it... reasoning and emits one get_weather name; London argument value is not asserted.',
630:'Unclosed-think streaming composition preserves clean reasoning and emits two ordered ReadFile names; argument values are not asserted.',
694:'functions.ReadFile:0 ID resolves to name ReadFile and counter zero.',
701:'ReadFile:1 ID without functions prefix resolves to name ReadFile and counter one.',
706:'Bare counter three with sole search tool resolves to search and retains counter three, not a tool-list index.',
714:'Bare counter zero with city argument matches get_weather schema properties and retains counter zero.',
721:'Bare counter five with empty tools gives no name and retains counter five.',
726:'Bare counter two with multiple tools and absent args gives no name and retains counter two.',
731:'Malformed nonnumeric ID some_garbage gives no name and default counter zero.',
738:'Name inference with no tools yields None.',
741:'Name inference with exactly one tool chooses only_one even for unrelated x argument.',
745:'Name inference from city/unit argument keys chooses get_weather.',
751:'Name inference from incomplete JSON with multiple tools yields None.',
755:'Name inference with multiple tools and None or empty argument text yields None.',
759:'Name inference with two schemas lacking properties yields None.',
769:'Complete bare-counter call infers get_weather and preserves exact Tokyo JSON argument string.',
782:'Complete bare-counter call whose unknown_key matches neither schema produces zero calls.',
794:'Streaming bare-counter call with sole search tool assembles one call named search; argument value is not asserted.',
}
REASON={306,321,350,364,386,401,432,443,467,479}
COMPOSE={521,571,599,630}
def main():
 ev=json.loads((HERE/'evidence.json').read_text())
 if any(c.get('family')==RULE for c in ev.get('reviewed_clusters',[])):print('Cumulative review already recorded.');return
 data=(SOURCE/PATH).read_bytes();text=data.decode();tree=ast.parse(text)
 nodes={n.lineno:(c.name,n) for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
 rows=[];aud=[];selected=set()
 for line,meaning in CASES.items():
  cls,n=nodes[line];name=cls+'::'+n.name
  checks=sorted([x for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr.startswith('assert')],key=lambda x:x.lineno);assert checks
  unit='response.reasoning_content_partition' if line in REASON else 'response.reasoning_tool_composition' if line in COMPOSE else 'tool.call_parse_relation'
  body=ast.get_source_segment(text,n)
  helpers=[x for x in ast.walk(n) if isinstance(x,ast.Call) and (isinstance(x.func,ast.Name) and x.func.id in ['_collect_streaming_tool_calls','_make_tool'] or isinstance(x.func,ast.Attribute) and x.func.attr=='_run_streaming')]
  cond=f'Pinned SGLang 0.5.14; local constructed Tool/Function declarations, literal formatted text and explicit chunk boundaries; no server or model execution. Exact fixture/body {PATH}:{line}-{n.end_lineno}; file SHA256={hashlib.sha256(data).hexdigest()}; body SHA256={hashlib.sha256(body.encode()).hexdigest()}. _make_tool at line 18 uses supplied nonempty schema or default path schema; _collect_streaming_tool_calls at line 37 assembles names/arguments by tool_index; reasoning _run_streaming at line 375 concatenates each result channel. These helpers and concrete class setUp inputs were reviewed.'
  prop=meaning+' Separately checked predicates: '+' | '.join(' '.join(ast.get_source_segment(text,x).split()) for x in checks)
  oracle=';'.join(f'{PATH}:{x.lineno}' for x in checks)
  rows.append(dict(snapshot='sglang',path=PATH,test_name=name,line=str(line),unit_id=unit,property=prop,oracle_source=oracle,conditions=cond,rule_id=RULE))
  aud.append(dict(path=PATH,line=line,test_name=name,unit_id=unit,reviewed_semantics=meaning,source_body=body,oracle_lines=[x.lineno for x in checks],helper_call_lines=[x.lineno for x in helpers]))
  selected.add((PATH,name,str(line)))
 for output,is_existing in [('crosswalk.tsv',True),('proposed-crosswalk.tsv',False)]:
  with (HERE/output).open('a',newline='') as f:csv.DictWriter(f,fieldnames=COLS,delimiter='\t').writerows(r for r in rows if (r['unit_id']=='tool.call_parse_relation')==is_existing)
 pending=list(csv.DictReader((HERE/'pending.tsv').open(),delimiter='\t'));remaining=[r for r in pending if (r['path'],r['test_name'],r['line']) not in selected]
 with (HERE/'pending.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=pending[0].keys(),delimiter='\t');w.writeheader();w.writerows(remaining)
 units=json.loads((HERE/'units-ready.json').read_text());dim='reasoning_parser_context'
 if dim not in {d['id'] for d in units['condition_dimensions']}:units['condition_dimensions'].append(dict(id=dim,type='structured',fields=['formatted_input','chunk_boundaries','reasoning_start_marker','reasoning_end_marker','tool_start_marker','force_reasoning','stream_reasoning','declared_tool_schemas'],observed_values=[],constraint='Bind the supplied formatted text and parser flags; reasoning-channel placement and downstream tool parsing are separate observed outputs.'))
 newunits=[dict(id='response.reasoning_content_partition',name='推理文本与普通内容按标记分离',property='按声明的推理起止标记、工具起始标记和强制推理模式，将输入内容分为推理与普通输出；流式不完整标记先缓冲，工具边界保留在普通输出供后续解析。',observable=['reasoning_text','normal_text','partial_marker_buffering'],oracle='对照显式输入检查两通道的精确文本或子串存在/缺失；闭合与未闭合推理块、工具边界、分块标记和 force_reasoning 各自保留条件。',source=[('python/sglang/srt/parser/reasoning_parser.py',62),('python/sglang/srt/parser/reasoning_parser.py',112)],testline=306,oracleline=318),dict(id='response.reasoning_tool_composition',name='推理分离后工具解析保留内容对应',property='推理解析器输出的普通片段输入工具解析器后，推理内容与最终工具调用分别对应原始格式化输出，工具标记不混入已分离的推理内容。',observable=['reasoning_text','downstream_tool_names','downstream_tool_call_count','assembled_tool_arguments'],oracle='用两个真实本地解析器顺序处理相同文本或分块，分别核对推理内容、工具调用数/顺序/名字，以及该测试实际断言的参数字符串或 JSON；未断言的参数不推断通过。',source=[('python/sglang/srt/parser/reasoning_parser.py',112),('python/sglang/srt/function_call/kimik2_detector.py',205)],testline=521,oracleline=559)]
 existing={u['id'] for u in units['units']}
 for u in newunits:
  if u['id'] in existing:continue
  source=u.pop('source');testline=u.pop('testline');oracleline=u.pop('oracleline')
  u.update(kind='obligation',phase='output',state_scope='request',conditions=[dim],evidence={'sglang':[dict(path=p,line=l,kind='source') for p,l in source]+[dict(path=PATH,line=testline,kind='existing_test'),dict(path=PATH,line=oracleline,kind='existing_oracle')]},framework_scope=['sglang'],candidate_status='proposed; requires parent integration',semantic_domain='inference_request')
  units['units'].append(u)
 (HERE/'units-ready.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n')
 paths=[PATH,'python/sglang/srt/function_call/kimik2_detector.py','python/sglang/srt/function_call/base_format_detector.py','python/sglang/srt/function_call/utils.py','python/sglang/srt/parser/reasoning_parser.py']
 hashes={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in paths}
 (HERE/'kimik2-parser-review.json').write_text(json.dumps(dict(review_method='All 45 bodies and local fixtures/helpers read, plus actual detector and inherited base implementations. Explicit source-line manifest; AST only transcribes already-reviewed predicates.',files_sha256=hashes,cases=aud),ensure_ascii=False,indent=2)+'\n')
 ev.setdefault('reviewed_clusters',[]).append(dict(family=RULE,reviewed_entries=len(rows),mapped_existing=sum(r['unit_id']=='tool.call_parse_relation' for r in rows),mapped_proposed=sum(r['unit_id']!='tool.call_parse_relation' for r in rows),files_sha256=hashes,audit='kimik2-parser-review.json'))
 ev['total_pending_after_review']=len(remaining);(HERE/'evidence.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n')
 print(dict(reviewed=len(rows),removed_pending=len(pending)-len(remaining),remaining=len(remaining)))
if __name__=='__main__':main()
