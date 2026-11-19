"""Line-specific source review of streaming response reconstruction."""
import ast,csv,json,hashlib,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=OUT.parents[1];sys.path.insert(0,str(P))
from build_round9_crosswalk import ROOTS,FIELDS,key,read
src=ROOTS['vllm'];path='tests/entrypoints/scale_out/derender/test_derender_stream.py';text=(src/path).read_text();tree=ast.parse(text)
groups={
'response.stream_payload_consistency':[540,547,556,578,591,622,690,760,899,928,951,1023,1343,1450,1909,1946,2293,2364,2594,2651,2732],
'response.stream_state_contract':[564,602,754,789,795,1272,1501,1773,1978,1982,1986,1990,1994,1998,2002],
'response.observed_payload_contract':[658,673,677,807,825,843,863,880,1073,1087,1323,1398,1621,1727,2765],
'response.reasoning_parser_initial_state':[328],
'output.incremental_parse_relation':[458,498],
'output.stop_boundary':[378,1109,1119,1582,1602],
'response.usage_accounting':[1045,1058,2461],
'response.logprob_text_decoding':[1137,1158,1188,1219,1638,1704,1748],
'response.logprob_identity':[1288,1681],
'response.stream_event_lifecycle':[1302,1412,1431,1549],
'request.declared_admission_contract':[1367,2032,2108,2122,2134,2152,2169,2188,2201,2218,2238,2438,2451,2791,2811],
'response.failure_contract':[2059,2073],
'execution.host_executor_dispatch':[986]}
assign={line:uid for uid,lines in groups.items() for line in lines};known={key(r) for r in read(P/'full_delivery/all-entries.tsv') if r['review_status']=='property_reviewed'}
rows=[];entries=[]
def visit(nodes,pre=''):
 for n in nodes:
  if isinstance(n,ast.ClassDef):visit(n.body,pre+n.name+'::')
  elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno in assign:
   r={'snapshot':'vllm','path':path,'test_name':pre+n.name,'line':str(n.lineno)}
   if key(r) in known:continue
   checks=[a for a in ast.walk(n) if isinstance(a,ast.Assert) or isinstance(a,ast.Call) and ((isinstance(a.func,ast.Attribute) and (a.func.attr.startswith('assert_') or a.func.attr=='raises')))]
   assert checks,n.name
   cond='生成token/元数据为构造输入，不运行模型生成。真实tokenizer；renderer与模型配置用替身；解析分支使用_FakeParser或方法替身，Harmony组使用真实HarmonyParser。按fixture区分调用边界。未执行上游测试。'
   if n.lineno>=2293:cond='HTTP render/derender服务；输入为构造的生成token，未运行模型生成。tiny Llama或DeepSeek-R1-Distill-Qwen tokenizer；hermes工具解析与deepseek_r1推理解析按parser_server配置。未执行上游测试。'
   if n.lineno in [1978,1982,1986,1990,1994,1998,2002,2032]:cond='直接构造Pydantic状态或请求，核对字段边界、分块长度一致性及异常；不运行服务或模型。'
   if 2059<=n.lineno<=2238:cond='真实ServingDerender校验逻辑，上游模型配置与OnlineDerenderer用MagicMock/AsyncMock；明确异常替身与未调用断言，不证明真实tokenizer执行。'
   if n.lineno==2594:cond+='triples参数带已知问题xfail，源码归类不代表测试通过。'
   if n.lineno==690:cond+='三次独立状态顺序执行，并非并发测试。'
   if n.lineno==1323:cond+='只核对构造metrics对象透传，不是性能测量。'
   if n.lineno==986:cond+='记录实际线程身份，检测解码运行于executor线程；不测量吞吐。'
   if n.lineno in [602,1272]:cond+='仅对列出的样例检查窗口上界，不据此证明任意输入复杂度。'
   prop='本项检查：'+'；'.join(ast.unparse(a) for a in checks)
   rows.append({**r,'unit_id':assign[n.lineno],'property':prop,'oracle_source':';'.join(f'{path}:{a.lineno}' for a in checks),'conditions':cond,'rule_id':'reviewed_stream_derender_source_20261120'})
   entries.append({**r,'source':ast.get_source_segment(text,n),'decorators':[ast.unparse(x) for x in n.decorator_list],'assertions':[{'line':a.lineno,'expression':ast.unparse(a)} for a in checks],'conditions':cond})
visit(tree.body)
files=[path,'tests/entrypoints/scale_out/derender/utils.py','vllm/renderers/online_derenderer.py','vllm/entrypoints/scale_out/derender/serving.py','vllm/entrypoints/scale_out/token_in_token_out/protocol.py']
evidence={'file_reviews':[{'snapshot':'vllm','path':f,'source_sha256':hashlib.sha256((src/f).read_bytes()).hexdigest()} for f in files],'entries':entries,'shared_helpers':[{'name':n.name,'line':n.lineno,'source':ast.get_source_segment(text,n)} for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and not n.name.startswith('test')],'source_review':'完整读取93个测试定义及共享fixtures/helper；核对生产增量解码、分块重放、state校验、响应构造。排除已确认入口后逐项映射。'}
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,FIELDS,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(len(rows))
