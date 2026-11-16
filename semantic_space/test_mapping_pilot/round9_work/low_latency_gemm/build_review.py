"""Reproduce reviewed contracts for two complete GEMM test modules.

Explicit index sets are decisions after reading every test and shared helper;
they are not a classifier for unseen files. No GPU execution is performed.
"""
import ast, csv, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PILOT=HERE.parents[1]
ROOT=PILOT.parents[3]/'experiments/third_party/reference_repos/vllm-20261105'
F=('snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id')
index=list(csv.DictReader((PILOT/'full-scan-index.tsv').open(),delimiter='\t'))
groups={
 'test_ll_fp32w_gemm.py': {
  'reference':{0,1,2,3,4,5,6,8,9,13,15}, 'bounds':{7,10},
  'relation':{11,12,14}, 'admissibility':set(range(16,23)),
  'dispatch':set(range(23,30))},
 'test_ll_bf16_gemm.py': {
  'reference':set(range(1,12))|{13,14,18,20,34,35},'bounds':{12,15},
  'relation':{16,17,19}, 'admissibility':set(range(27,34)),
  'dispatch':{0}|set(range(21,27))},
}
units={
 'reference':('compute.operator_reference_relation','将实际 GEMM 结果与本例明确的矩阵乘或零值参考比较'),
 'bounds':('compute.output_constraints','检查输出类型，或有限性及绝对值上界，具体以该入口断言为准'),
 'relation':('compute.execution_relation','比较重复执行、图重放或声明执行变体之间的输出关系'),
 'admissibility':('compute.input_admissibility','给定不支持的算子输入类型、设备、维度或布局时，入口按断言拒绝'),
 'dispatch':('compute.kernel_dispatch_binding','在给定格式、批量及后端可用条件下检查允许标志或实际调用实现'),
}
out=[];contracts=[];reviews=[]
for filename, decisions in groups.items():
 rel='tests/kernels/'+filename;raw=(ROOT/rel).read_bytes();src=raw.decode();tree=ast.parse(src)
 tests=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
 helper={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
 assigned={i:g for g,indices in decisions.items() for i in indices}
 assert len(assigned)==sum(map(len,decisions.values()))==len(tests)
 for i,node in enumerate(tests):
  row=next(r for r in index if r['snapshot']=='vllm' and r['path']==rel and int(r['line'])==node.lineno)
  checks=[];calls=[]
  for n in ast.walk(node):
   if isinstance(n,ast.Assert):checks.append((n.lineno,ast.unparse(n)))
   if isinstance(n,ast.Call):
    fn=ast.unparse(n.func)
    if fn in {'_assert_close','torch.testing.assert_close','pytest.raises'}:
     checks.append((n.lineno,ast.unparse(n)));calls.append(fn)
  assert checks,node.name
  helper_checks=[]
  if '_assert_close' in calls:
   helper_checks=[(n.lineno,ast.unparse(n)) for n in ast.walk(helper['_assert_close']) if isinstance(n,ast.Assert)]
  uid,prop=units[assigned[i]]
  condition='CUDA、SM90+ 与 cuteDSL 可用性由模块标记及自动 fixture 检查；参数、形状、精度、随机种子与输入值见 '+rel+':'+str(node.lineno)+'。'
  condition+='参数装饰：'+'; '.join(ast.unparse(n) for n in node.decorator_list)+'。'
  if helper_checks:condition+='共享 _assert_close 检查 CUDA、有限性和展平输出的余弦相似度，默认阈值 0.99；调用显式覆盖时使用该值。打印的最大绝对误差不是通过阈值。'
  if assigned[i]=='dispatch':condition+='记录替身内核、能力标志及 TP=1 fixture；此项只支持实际断言的分派或参数传递结论。'
  if assigned[i]=='bounds':condition+='此项没有与完整矩阵乘参考进行数值对照。'
  if assigned[i]=='relation':condition+='比较的两侧及相等/相似阈值见调用表达式；相同输出本身不建立相对正确参考的一致性。'
  all_checks=sorted(set(checks+helper_checks))
  out.append({**{k:row[k] for k in F[:4]},'unit_id':uid,
   'property':prop+'；实际检查：'+'; '.join(x for _,x in checks),
   'oracle_source':';'.join(f'{rel}:{line}' for line,_ in all_checks),
   'conditions':condition,'rule_id':'round9_reviewed_low_latency_gemm_'+assigned[i]})
  contracts.append({'snapshot':'vllm','path':rel,'test_name':row['test_name'],'line':node.lineno,
    'source_sha256':hashlib.sha256(raw).hexdigest(),'unit_id':uid,'checks':all_checks,
    'input_and_call_source':ast.get_source_segment(src,node),'conditions':condition})
 reviews.append({'snapshot':'vllm','path':rel,'source_sha256':hashlib.sha256(raw).hexdigest(),
   'review':'Read every test, _ref/_assert_close/_gemm helpers, platform fixtures and GateLinear mock setup; distinguish reference, bounded output, execution relation, rejection and dispatch.'})
with (HERE/'crosswalk.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,F,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(out)
(HERE/'reviewed-contracts.json').write_text(json.dumps(contracts,ensure_ascii=False,indent=2)+'\n')
(HERE/'evidence.json').write_text(json.dumps({'file_reviews':reviews},ensure_ascii=False,indent=2)+'\n')
print({'reviewed_contracts':len(out),'files':len(reviews)})
