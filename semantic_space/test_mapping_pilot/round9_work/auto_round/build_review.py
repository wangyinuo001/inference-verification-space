"""Reproduce the explicit function-by-function AutoRound source review.

The sets below are review decisions, not name-based classification rules.
The manifest retains the actual assertions and decorators of every decision.
No test execution or numerical equivalence is inferred from this source audit.
"""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
PROJECT = PILOT.parents[3]
REL = 'tests/quantization/test_auto_round.py'
ROOT = PROJECT / 'experiments/third_party/reference_repos/vllm-20261105'
raw = (ROOT / REL).read_bytes()
text = raw.decode()
tree = ast.parse(text)
functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith('test_')]
assert len(functions) == 93

groups = [
    ('deployment.effective_configuration', '层配置解析所得位宽、分组、对称性、量化开关及融合匹配字段符合给定配置', {1,8,9,10,11,12,15,17,18,20,26,45,88,89,90,92}),
    ('deployment.configuration_admissibility', '给定不支持的配置、后端能力或分片形状触发测试指定异常及错误文本', {13,14,16,19,37,41,47,48,52,54,66,67,68,70,71,72,73,74,75,91}),
    ('compute.kernel_dispatch_binding', '指定层类型、量化格式、可用后端及批量条件选择预期实现，或返回不适用结果', {2,3,4,5,6,7,21,22,23,24,25,27,28,29,30,34,35,36,38,39,44,46,50,51,53,55,56,57,58,59,60,61,62,63,64,65,69,80,85,87}),
    ('model.parameter_storage_layout', '创建或处理后的权重与缩放参数满足断言中的形状、数据类型、打包方式和别名关系', {31,32,33,49,76,77}),
    ('compute.kernel_call_binding', '包装层向替身内核传递约定参数并返回规定形状及类型的结果', {40,42,43,81,82,83,84}),
    ('model.parameter_representation_relation', '权重预处理前后保留断言指定的数值或位表示，同时建立执行后端需要的参数表示', {78,79}),
    ('quality.sample_semantic_expectation', '真实权重生成的回答包含预先指定的样本答案 Paris', {86}),
]
assigned = {}
for unit, prop, indices in groups:
    for i in indices:
        assert i not in assigned
        assigned[i] = (unit, prop)
assert set(assigned) == set(range(1, 93))

special = {
    31: '同时检查不满足块大小的权重创建被拒绝；未执行真实 MoE 数值对照。',
    32: '同时检查后处理后数据指针及替身矩阵乘调用参数；此项没有真实内核数值判定。',
    33: '检查打包参数别名与路由表的透传；计算后端使用替身。',
    78: '本例填充值为 0.125，在两种精度中均可精确表示；同时保留 BF16 原缩放量。',
    79: '比较打包权重与处理前克隆张量逐元素相等，零点为固定值 8。',
    80: '阈值以下调用回退路径；另一内核替身会主动使测试失败。',
    81: '阈值处检查量化参数、GEMM 参数对象及类型、结果类型；数值内核是替身。',
    84: '仅检查高维输入的前导形状和输出类型恢复。',
    85: '用展平后的 token 总数选择路径；检查输出形状，未比较计算值。',
    86: '需要真实 XPU 内核和模型权重；样本条件是回答中包含 Paris。',
    87: '仅检查收集到的 W4A8 层集合非空及其分片对齐，没有断言所有 Linear 均在该集合中。',
}
index = list(csv.DictReader((PILOT / 'full-scan-index.tsv').open(), delimiter='\t'))
by_location = {(r['test_name'].split('::')[-1], int(r['line'])):r for r in index if r['snapshot']=='vllm' and r['path']==REL}
rows = []
contracts = []
for i in range(1, 93):
    node = functions[i]
    entry = by_location[(node.name, node.lineno)]
    checks = []
    for n in ast.walk(node):
        if isinstance(n, ast.Assert):
            checks.append((n.lineno, ast.unparse(n.test)))
        elif isinstance(n, ast.Call):
            call = ast.unparse(n.func)
            if call in {'pytest.raises', 'torch.testing.assert_close'} or call.endswith(('.assert_called_once', '.assert_called_once_with', '.assert_not_called')):
                checks.append((n.lineno, ast.unparse(n)))
    checks = sorted(set(checks))
    assert checks, node.name
    decorators = [ast.unparse(n) for n in node.decorator_list]
    body = ast.get_source_segment(text, node)
    unit, prop = assigned[i]
    sites = ';'.join(f'{REL}:{line}' for line, _ in checks)
    condition = f'输入构造、fixture 与替身范围见 {REL}:{node.lineno}；'
    condition += '装饰条件：' + ('; '.join(decorators) or '无函数级参数装饰') + '。'
    if 'monkeypatch' in body or '_stub_' in body or 'Fake' in body:
        condition += '包含平台或计算后端替身，结论限于该受控调用条件。'
    if i in special:
        condition += special[i]
    # Each contract retains every actual check; family text is only its common heading.
    property_text = prop + '；具体判定：' + '; '.join(expr for _, expr in checks)
    rows.append({**{k:entry[k] for k in ('snapshot','path','test_name','line')},
                 'unit_id':unit, 'property':property_text, 'oracle_source':sites,
                 'conditions':condition, 'rule_id':'round9_individually_reviewed_autoround_contract'})
    contracts.append({'test_name':entry['test_name'], 'line':node.lineno,
                      'end_line':node.end_lineno, 'unit_id':unit,
                      'checks':[{'line':line,'expression':expr} for line,expr in checks],
                      'decorators':decorators,'source_sha256':hashlib.sha256(raw).hexdigest(),
                      'function_sha256':hashlib.sha256(body.encode()).hexdigest(),
                      'conditions':condition})
fields = ('snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id')
with (HERE / 'crosswalk.tsv').open('w', newline='') as f:
    w = csv.DictWriter(f, fields, delimiter='\t', lineterminator='\n');w.writeheader();w.writerows(rows)
(HERE/'reviewed-contracts.json').write_text(json.dumps(contracts,ensure_ascii=False,indent=2)+'\n')
(HERE/'evidence.json').write_text(json.dumps({'file_reviews':[{
    'snapshot':'vllm','path':REL,'source_sha256':hashlib.sha256(raw).hexdigest(),
    'review':'Read all 93 function bodies, configuration constructors, dispatch and weight fixtures, quant/GEMM stubs and selected product implementations. Existing model smoke entry remains in the earlier crosswalk; 92 remaining source contracts are recorded here.',
    'product_sources':['vllm/model_executor/layers/quantization/inc/inc.py:322',
                       'vllm/model_executor/layers/quantization/inc/config_parser.py:64',
                       'vllm/model_executor/layers/quantization/inc/schemes/factory.py:11',
                       'vllm/model_executor/layers/quantization/inc/schemes/inc_w4a8_linear.py:42']
}]},ensure_ascii=False,indent=2)+'\n')
print({'reviewed':len(rows),'existing_smoke_entry_unchanged':1})
