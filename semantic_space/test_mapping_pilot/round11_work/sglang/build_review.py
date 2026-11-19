import ast,csv,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; PROJECT=ROOT.parents[1]
SRC=PROJECT/'experiments/third_party/reference_repos/sglang-v0.5.14-complete';OUT=Path(__file__).parent
BASE='test/registered/debug_utils/comparator/'
IMPL='python/sglang/srt/debug_utils/comparator/'
rows=list(csv.DictReader((ROOT/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'))
EXIST='diagnostic.comparator_report_relation'
META='diagnostic.dimension_metadata_contract'
AXIS='diagnostic.tensor_axis_alignment'
RECORD='diagnostic.record_schema_and_encoding'
DISC='diagnostic.dump_input_discovery'
specs={
'dims_spec/test_dim_parser.py':(META,'诊断维度标记的解析结果及非法标记拒绝'),
'dims_spec/test_tensor_naming.py':(META,'诊断张量维度命名、轴定位及移除名称的结果'),
'dims_spec/test_types.py':(META,'诊断维度标识约定'),
'aligner/test_axis_aligner.py':(AXIS,'依据声明维度生成对齐方案或执行轴变换'),
'tensor_comparator/test_comparator.py':(EXIST,'构造张量的数值统计、差异及比较记录'),
'tensor_comparator/test_types.py':(RECORD,'比较记录字段、判定类别与序列化'),
'test_model_validation.py':(RECORD,'诊断元数据和记录的结构约束及字段一致性'),
'test_output_types.py':(RECORD,'比较记录分类、字段及报告输出'),
'tensor_comparator/test_formatter.py':(RECORD,'比较记录转换为文本或终端显示时的字段、数值与格式'),
'test_utils.py':(EXIST,'张量比较辅助计算与比较退出判定'),
}
source_paths=[IMPL+p for p in ['dims_spec/dim_parser.py','dims_spec/dims_parser.py','dims_spec/modifier_parser.py','dims_spec/comment_parser.py','dims_spec/tensor_naming.py','dims_spec/types.py','aligner/axis_aligner.py','utils.py','aligner/token_aligner/smart/types.py','aligner/unsharder/types.py','tensor_comparator/types.py','tensor_comparator/comparator.py','output_types.py','output_formatter.py','tensor_comparator/formatter.py']]+[BASE+'testing_helpers.py']
cache={};functions={};overridden=set()
for suffix in specs:
 p=BASE+suffix;text=(SRC/p).read_text();tree=ast.parse(text);cache[p]=text
 last_classes={n.name:n.lineno for n in tree.body if isinstance(n,ast.ClassDef)}
 for c in tree.body:
  nodes=[(c.name+'::'+n.name,n) for n in c.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name.startswith('test')] if isinstance(c,ast.ClassDef) else [(c.name,c)] if isinstance(c,(ast.FunctionDef,ast.AsyncFunctionDef)) and c.name.startswith('test') else []
  for name,n in nodes:
   functions[(p,name,n.lineno)]=n
   if isinstance(c,ast.ClassDef) and last_classes[c.name]!=c.lineno:overridden.add((p,name,n.lineno))
results=[];evidence=[];support=[];skipped=[]
for r in rows:
 if r['snapshot']!='sglang' or r['review_status'] in ('property_reviewed','supporting_record_reviewed','outside_current_scope'):continue
 key=(r['path'],r['test_name'],int(r['line']))
 if key not in functions:continue
 n=functions[key];p=r['path'];text=cache[p];unit,category=specs[p[len(BASE):]]
 if key in overridden or r['test_name'].startswith('TestPairMap::'):
  support.append({'snapshot':'sglang','path':p,'test_name':r['test_name'],'line':int(r['line']),'role':'被后续同名类覆盖的源码定义，不是该文件最终绑定的 pytest 类' if key in overridden else '通用 Pair 容器 map 辅助函数，检查值变换及返回新对象，不是推理请求性质'});continue
 if r['test_name'].startswith('TestAutoDescendDir::'):unit=DISC;category='诊断输入目录的发现、唯一性与无数据错误'
 checks=[];calls=[];inputs=[]
 for a in ast.walk(n):
  expr=ast.get_source_segment(text,a)
  if isinstance(a,ast.Assert):checks.append({'line':a.lineno,'kind':'assert','expression':expr})
  if isinstance(a,ast.Call):
   fn=ast.unparse(a.func);calls.append({'line':a.lineno,'target':fn,'expression':expr})
   if fn.startswith('assert_') or fn.split('.')[-1].startswith('assert') or fn.endswith('.raises') or fn.endswith('.assertRaises') or fn.endswith('.assertRaisesRegex'):
    checks.append({'line':a.lineno,'kind':'raises' if fn.endswith('raises') or 'assertRaises' in fn else 'assertion_call','expression':expr})
  if isinstance(a,(ast.Assign,ast.AnnAssign)):
   inputs.append({'line':a.lineno,'expression':expr})
 checks.sort(key=lambda a:a['line'])
 if not checks:
  if p.endswith('test_model_validation.py') and r['test_name'] in ('TestCheckEqualLengths::test_all_equal','TestCheckEqualLengths::test_empty_lists'):
   c=calls[-1];checks=[{'line':c['line'],'kind':'implicit_no_exception','expression':'无异常完成 '+c['expression']}]
  else:skipped.append({'path':p,'test_name':r['test_name'],'reason':'当前提取未定位显式断言，保留未确认'});continue
 property_text='类别：'+category+'；本项实际检查：'+'；'.join(' '.join(a['expression'].split()) for a in checks)
 result={k:r[k] for k in ['snapshot','path','test_name','line']}
 result.update(unit_id=unit,property=property_text,oracle_source='; '.join(f'{p}:{a["line"]}' for a in checks),conditions='逐项阅读测试体并核对被测辅助实现；输入及参数见 evidence 中的 input_statements 和 calls。属于诊断支持工具，使用构造张量、维度注释或记录；未执行上游测试，不表示请求运行或模型计算已验证。',rule_id='round11_sglang_comparator_source_review_v1')
 results.append(result)
 evidence.append({'snapshot':'sglang','path':p,'test_name':r['test_name'],'line':int(r['line']),'unit_id':unit,'test_body_sha256':hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest(),'assertions':checks,'input_statements':inputs,'calls':calls,'review_basis':'逐项测试体源码阅读；按实际检查表达式登记，实施文件审查见file_reviews。'})
fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
for fname,items in [('crosswalk.tsv',results),('existing-units-crosswalk.tsv',[x for x in results if x['unit_id']==EXIST]),('proposed-units-crosswalk.tsv',[x for x in results if x['unit_id']!=EXIST])]:
 with (OUT/fname).open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(items)
file_reviews=[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()} for p in sorted(set(cache)|set(source_paths))]
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':file_reviews,'entries':evidence,'supporting_records':support,'unconfirmed':skipped,'count':len(results),'scope':'诊断比较器源码性质审查。测试使用构造输入；没有新增执行结果。'},ensure_ascii=False,indent=2)+'\n')
proposals=[]
for uid,name,prop,obs,oracle,paths in [
(META,'诊断维度元数据','诊断工具对维度名称、融合关系、分片轴及修饰符进行解析和命名，保留声明对应关系并拒绝格式或约束冲突。',['parsed_dimensions','axis_indices','dimension_names','validation_error'],'对显式维度字符串及构造张量检查解析结果、命名和异常。',['dims_spec/dim_parser.py','dims_spec/dims_parser.py','dims_spec/tensor_naming.py','dims_spec/types.py']),
(AXIS,'诊断张量轴对齐','比较两侧张量时，根据维度语义建立轴置换、单例压缩和融合计划；声明冲突时返回诊断，不生成有效对齐计划。',['axis_plan','aligned_shape','aligned_values','alignment_error'],'构造维度注释和张量，检查变换模式与对应形状或元素；具体保证随测试断言。',['aligner/axis_aligner.py']),
(RECORD,'诊断记录约束与表达','诊断记录满足字段与计数约束，序列化和文字报告保留所声明的类别、字段、数值及诊断信息。',['record_category','serialized_record','summary_counts','text_report'],'通过构造记录、非法字段和预期文本逐项检查结构约束、往返结果或格式。',['output_types.py','output_formatter.py','tensor_comparator/formatter.py','aligner/token_aligner/smart/types.py','aligner/unsharder/types.py']),
(DISC,'诊断转储输入发现','给定目录，优先使用已有转储文件；否则定位唯一含转储的子目录，无数据或存在多个候选时拒绝歧义。',['selected_directory','discovery_error'],'构造含转储文件的目录层级并检查选择路径和异常。',['utils.py'])]:
 proposals.append({'id':uid,'name':name,'kind':'obligation','phase':'diagnostic','state_scope':'diagnostic_artifact','conditions':[],'property':prop,'observable':obs,'oracle':oracle,'evidence':{'sglang':[{'path':IMPL+p,'line':next((n.lineno for n in ast.parse((SRC/(IMPL+p)).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))),1),'kind':'source','source_sha256':hashlib.sha256((SRC/(IMPL+p)).read_bytes()).hexdigest()} for p in paths]},'framework_scope':['sglang'],'semantic_domain':'diagnostic_support','proposal_note':'支持工具性质，不归入推理请求路径；条件需按元数据/记录输入进一步细化。'})
(OUT/'proposed-units.json').write_text(json.dumps(proposals,ensure_ascii=False,indent=2)+'\n')
assert len({(x['path'],x['test_name'],x['line']) for x in results})==len(results)
assert all(re.fullmatch(r'[^;]+:\d+(; [^;]+:\d+)*',x['oracle_source']) for x in results)
print('rows',len(results),'existing',sum(x['unit_id']==EXIST for x in results),'support',len(support),'unconfirmed',skipped)
print({u:sum(x['unit_id']==u for x in results) for u in sorted({x['unit_id'] for x in results})})
