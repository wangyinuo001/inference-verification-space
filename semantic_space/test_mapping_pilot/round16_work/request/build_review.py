"""Reviewed derender response checks; AST preserves each distinct assertion."""
import ast,csv,json,hashlib,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=OUT.parents[1];sys.path.insert(0,str(P))
from build_round9_crosswalk import ROOTS,FIELDS,KEY,key,read
src=ROOTS['vllm'];path='tests/entrypoints/scale_out/derender/test_derender.py';raw=(src/path).read_bytes();text=raw.decode();tree=ast.parse(text)
known={key(r) for r in read(P/'full_delivery/all-entries.tsv') if r['review_status']=='property_reviewed'}
# These line-specific groups were assigned after reading endpoint inputs,
# shared render/server helpers, the assertions and the derender implementation.
groups={
'request.declared_admission_contract':[392,405,418,486,601,620,677,698,864,885,906,927,941,971,1010,1369],
'response.usage_accounting':[189,210,574],
'response.logprob_text_decoding':[228,257,282,708,743,786],
'output.stop_boundary':[1880,1926],
'response.observed_payload_contract':[105,127,145,173,308,329,349,434,453,544,640,818,845,1131,1161,1189,1216,1249,1292,1331,1414,1524,1553,1634,1676,1767,1778,1786,1812,1948]}
assign={line:uid for uid,lines in groups.items() for line in lines}
rows=[];entries=[]
helpers=[{'name':n.name,'line':n.lineno,'source':ast.get_source_segment(text,n)} for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not n.name.startswith('test_')]
for n in tree.body:
 if not isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) or n.lineno not in assign:continue
 r={'snapshot':'vllm','path':path,'test_name':n.name,'line':str(n.lineno)}
 if key(r) in known:continue
 asserts=[a for a in ast.walk(n) if isinstance(a,ast.Assert)];assert asserts
 conditions='真实 tokenizer/OnlineDerenderer；输入为手工构造的生成 token 与元数据，不运行模型生成。HTTP入口由 render server fixture 执行；直接方法测试中的 renderer/model_config 和部分 parser 使用明确替身。具体 parser、模型、输入和 stream 分支见完整函数及fixture。未执行上游测试。'
 if n.lineno in [453,640]:conditions+='检查提示词上下文导致的前导空格及去空格后文本一致性。'
 if n.lineno==257:conditions+='bytes 只检查列表非空及整数类型，不检查实际字节值。'
 if n.lineno==349:conditions+='此项检查指标字段投影和空值，不测量响应性能。'
 if n.lineno==1189:conditions+='此项只排除替换字符，并未完整比较原文。'
 rows.append({**r,'unit_id':assign[n.lineno],'property':'本项断言：'+'；'.join(ast.unparse(a) for a in asserts),'oracle_source':';'.join(f'{path}:{a.lineno}' for a in asserts),'conditions':conditions,'rule_id':'reviewed_derender_source_and_assertions_20261120'})
 entries.append({**r,'source':ast.get_source_segment(text,n),'decorators':[ast.unparse(x) for x in n.decorator_list],'assertions':[{'line':a.lineno,'expression':ast.unparse(a)} for a in asserts],'conditions':conditions})
files=[path,'vllm/entrypoints/scale_out/derender/serving.py','vllm/renderers/online_derenderer.py']
evidence={'file_reviews':[{'snapshot':'vllm','path':f,'source_sha256':hashlib.sha256((src/f).read_bytes()).hexdigest()} for f in files],'entries':entries,'shared_helpers':helpers,'source_review':'读取测试调用/断言、render server和tokenizer fixtures、构造响应helper及生产ServingDerender/OnlineDerenderer。具体输出检查不推广为模型正确性。'}
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,FIELDS,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(len(rows))
