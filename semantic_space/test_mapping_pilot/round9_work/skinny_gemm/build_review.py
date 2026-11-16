"""Explicit per-definition decisions after reading the complete 1,524-line module.
AST transcribes evidence; it does not assign semantic classes to unseen tests.
"""
import ast,csv,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parents[1];R=P.parents[3]/'experiments/third_party/reference_repos/vllm-20261105'
path='tests/kernels/test_bf16_skinny_gemm.py';raw=(R/path).read_bytes();src=raw.decode();tree=ast.parse(src)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
assert len(functions)==50
reference={7,29,30,31,32,36,37,38,39,40,41,42,43,44,45,46,47,49}
plans={0,1,3,4,5,11,19,20,22,23,25,33}
admissibility={2,13}
dispatch=set(range(50))-reference-plans-admissibility
units={'reference':'compute.operator_reference_relation','plans':'compute.kernel_plan_consistency','admissibility':'compute.input_admissibility','dispatch':'compute.kernel_dispatch_binding'}
F=('snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id');out=[];contracts=[]
for i,n in enumerate(functions):
 group=next(g for g,indices in [('reference',reference),('plans',plans),('admissibility',admissibility),('dispatch',dispatch)] if i in indices)
 checks=sorted([(x.lineno,ast.unparse(x)) for x in ast.walk(n) if isinstance(x,ast.Assert) or isinstance(x,ast.Call) and ast.unparse(x.func)=='torch.testing.assert_close'])
 assert checks,(i,n.name)
 helpers=[x for x in tree.body if isinstance(x,ast.FunctionDef) and not x.name.startswith('test_') and any(isinstance(c,ast.Call) and ast.unparse(c.func)==x.name for c in ast.walk(n))]
 # Preserve local helper gates and the GLM device-plan assertion in each contract.
 helperchecks=[(h.name,x.lineno,ast.unparse(x)) for h in helpers for x in ast.walk(h) if isinstance(x,ast.Assert)]
 prop={'reference':'实际输出按本入口的线性/矩阵乘及可选残差参考进行数值比较','plans':'计划表的形状键、配置约束、编译实例或分派对应满足本入口的具体断言','admissibility':'核对单行和多行切片的真实stride与紧密行主序条件','dispatch':'受控形状、平台、格式及配置下核对选中的实现、返回身份或回退结果'}[group]
 prop+='；'+ '；'.join(t for _,t in checks)
 conditions='参数与受控条件：'+','.join(ast.unparse(d) for d in n.decorator_list)+'；源码内的shape、dtype、stride、平台模拟、配置表及reference按本入口绑定。'
 if group=='reference':conditions+=' GPU门禁保留；cosine阈值或assert_close容差以逐项断言为准，图重放只在该入口显式使用时成立。'
 else:conditions+=' 计划与模拟分派检查不提供真实kernel输出数值结论。'
 row=dict(zip(F,('vllm',path,n.name,str(n.lineno),units[group],prop,';'.join(f'{path}:{l}' for l,_ in checks)+''.join(f';{path}:{l}' for _,l,_ in helperchecks),conditions,'source_review_skinny_gemm_complete_module')));out.append(row)
 contracts.append({**row,'ordinal':i,'source_body':ast.get_source_segment(src,n),'assertions':checks,'helper_checks':helperchecks,'helper_sources':{h.name:ast.get_source_segment(src,h) for h in helpers},'source_sha256':hashlib.sha256(raw).hexdigest()})
with (H/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,F,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(out)
(H/'reviewed-contracts.json').write_text(json.dumps(contracts,ensure_ascii=False,indent=2)+'\n')
(H/'evidence.json').write_text(json.dumps({'file_reviews':[{'snapshot':'vllm','path':path,'source_sha256':hashlib.sha256(raw).hexdigest(),'reviewed_definitions':50,'executed':False}]},indent=2)+'\n')
print({'reviewed':len(out),'executed':False})
