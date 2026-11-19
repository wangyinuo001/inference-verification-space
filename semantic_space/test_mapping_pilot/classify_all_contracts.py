"""Classify every indexed record by its observed checking relation.

These framework-independent relation classes supplement, rather than replace,
the reviewed system properties. No rule infers isolation, progress, lifetime
safety or a runtime path from a filename or an equality assertion.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_contract_batches import HERE, KEY, call_name
from resolve_inherited_methods import ROOTS

LABELS = {
    'numeric_tolerance': '数值误差', 'equality': '值相等', 'inequality': '值不同',
    'order_bound': '范围或大小关系', 'membership': '成员或包含关系',
    'identity': '对象身份或空值', 'type': '类型约束', 'shape': '形状或维度',
    'dtype': '数据精度类型', 'cardinality': '数量或长度',
    'predicate': '条件成立', 'exception': '预期异常', 'warning': '预期警告',
    'mock_interaction': '模拟调用参数或次数', 'helper_check': '辅助函数中的判定',
    'reviewed_property': '已有具体性质归类', 'unresolved_check': '需定位检查内容',
    'supporting_record': '辅助代码', 'incomplete_test': '尚无有效检查',
    'outside_scope': '当前范围外', 'non_python_review': '其他语言或脚本检查',
}

METHODS = {
    'assertEqual': 'equality', 'assertEquals': 'equality', 'assertListEqual': 'equality',
    'assertTupleEqual': 'equality', 'assertDictEqual': 'equality', 'assertSetEqual': 'equality',
    'assertSequenceEqual': 'equality', 'assertMultiLineEqual': 'equality',
    'assertNotEqual': 'inequality', 'assertNotEquals': 'inequality',
    'assertAlmostEqual': 'numeric_tolerance', 'assertAlmostEquals': 'numeric_tolerance',
    'assertNotAlmostEqual': 'numeric_tolerance',
    'assertGreater': 'order_bound', 'assertGreaterEqual': 'order_bound',
    'assertLess': 'order_bound', 'assertLessEqual': 'order_bound',
    'assertIn': 'membership', 'assertNotIn': 'membership',
    'assertIs': 'identity', 'assertIsNot': 'identity', 'assertIsNone': 'identity',
    'assertIsNotNone': 'identity', 'assertIsInstance': 'type', 'assertNotIsInstance': 'type',
    'assertTrue': 'predicate', 'assertFalse': 'predicate',
    'assertRaises': 'exception', 'assertRaisesRegex': 'exception',
    'assertWarns': 'warning', 'assertWarnsRegex': 'warning',
}


def read(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def key(row):
    return tuple(row[k] for k in KEY)


def observed_facets(expr):
    """Look at the checked operands, not the error message or call name."""
    result = set()
    for n in ast.walk(expr):
        if isinstance(n, ast.Attribute):
            if n.attr in {'shape', 'ndim'}: result.add('shape')
            if n.attr == 'dtype': result.add('dtype')
        if isinstance(n, ast.Call):
            name = call_name(n.func)
            if name in {'len', 'builtins.len'}: result.add('cardinality')
            if name in {'isinstance', 'issubclass', 'type'}: result.add('type')
    return result


def relation_kinds(check):
    """Classify the assertion's stated relation, not its semantic guarantee."""
    try:
        node = ast.parse(check['expression']).body[0]
    except (SyntaxError, ValueError):
        return {'helper_check'}
    expr = node.test if isinstance(node, ast.Assert) else node.value if isinstance(node, ast.Expr) else node
    result = set()
    if isinstance(node, ast.Assert):
        for n in ast.walk(expr):
            if isinstance(n, ast.Compare):
                for op in n.ops:
                    if isinstance(op, ast.Eq): result.add('equality')
                    elif isinstance(op, ast.NotEq): result.add('inequality')
                    elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)): result.add('order_bound')
                    elif isinstance(op, (ast.In, ast.NotIn)): result.add('membership')
                    elif isinstance(op, (ast.Is, ast.IsNot)): result.add('identity')
            if isinstance(n, ast.Call) and call_name(n.func) in {
                'torch.allclose', 'np.allclose', 'numpy.allclose', 'math.isclose', 'pytest.approx'}:
                result.add('numeric_tolerance')
        result.update(observed_facets(expr))
        return result or {'predicate'}
    if isinstance(expr, ast.Call):
        name = call_name(expr.func)
        method = name.rsplit('.', 1)[-1]
        if check['kind'] == 'expected_exception': result.add('exception')
        elif check['kind'] == 'expected_warning': result.add('warning')
        elif name.startswith('self.') and method in METHODS: result.add(METHODS[method])
        elif name in {'torch.testing.assert_close', 'np.testing.assert_allclose',
                      'numpy.testing.assert_allclose', 'np.testing.assert_almost_equal',
                      'numpy.testing.assert_almost_equal', 'np.testing.assert_array_almost_equal'}:
            result.add('numeric_tolerance')
        elif name in {'np.testing.assert_array_equal', 'numpy.testing.assert_array_equal',
                      'np.testing.assert_equal', 'numpy.testing.assert_equal'}:
            result.add('equality')
        elif method in {'assert_called', 'assert_called_once', 'assert_called_with',
                        'assert_called_once_with', 'assert_not_called', 'assert_has_calls',
                        'assert_any_call', 'assert_awaited_once_with', 'assert_awaited'}:
            result.add('mock_interaction')
        else: result.add('helper_check')
        # Message/regex arguments must not become observations of output shape.
        if result & {'exception', 'warning', 'mock_interaction', 'helper_check'}:
            return result
        count = 1 if method in {'assertTrue', 'assertFalse', 'assertIsNone', 'assertIsNotNone'} else 2
        for operand in expr.args[:count]: result.update(observed_facets(operand))
    return result or {'helper_check'}


def classify(card, reviewed, placement):
    atoms = []
    evidence = card.get('evidence') or {}
    for check in evidence.get('checks', []):
        atoms.append({**{k: check[k] for k in ('source', 'expression')},
                      'relations': sorted(relation_kinds(check)), 'origin': 'test_body'})
    for helper in evidence.get('local_helpers', []):
        for check in helper['body']['checks']:
            atoms.append({**{k: check[k] for k in ('source', 'expression')},
                          'relations': sorted(relation_kinds(check)), 'origin': 'local_helper',
                          'helper_call': helper['call_site']})
    relations = sorted({r for atom in atoms for r in atom['relations']})
    status = reviewed['review_status']
    if status == 'property_reviewed': disposition = 'reviewed_property'
    elif status == 'supporting_record_reviewed': disposition = 'supporting_record'
    elif status == 'incomplete_test_reviewed': disposition = 'incomplete_test'
    elif status == 'outside_current_scope': disposition = 'outside_scope'
    elif relations and set(relations) != {'helper_check'}: disposition = 'source_relation_classified'
    elif relations: disposition = 'helper_check'
    elif card['extraction_status'] == 'non_python': disposition = 'non_python_review'
    else: disposition = 'unresolved_check'
    return {
        **{k: v for k, v in zip(KEY, card['entry'])},
        'disposition': disposition,
        'relation_ids': ';'.join(relations),
        'relation_labels': '；'.join(LABELS[r] for r in relations),
        'reviewed_units': reviewed['unit_ids'], 'reviewed_property': reviewed['property'],
        'object_candidates': reviewed['object_candidates'],
        'stage': placement.get('stage', ''), 'stage_basis': placement.get('stage_source', ''),
        'product_target': placement.get('product_target', ''),
        'definition_path': card['definition_path'], 'definition_line': card['definition_line'],
        'source_sha256': reviewed['source_sha256'],
        'definition_source_sha256': card['source_sha256'],
        'checks': atoms,
        'conditions': {'parameters': evidence.get('parameters', ''),
                       'decorators': evidence.get('decorators', []),
                       'guards': evidence.get('guards', []),
                       'class_context': evidence.get('class_context')},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cards', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'full_classification')
    args = parser.parse_args()
    reviewed = {key(r): r for r in read(HERE/'full_delivery/all-entries.tsv')}
    placed = {key(r): r for r in read(HERE/'stage-placement-all.tsv')}
    assert reviewed.keys() == placed.keys()
    rows, details, seen = [], [], set()
    out = args.output
    out.mkdir(exist_ok=True, parents=True)
    # Large exact expressions remain local; the page carries compact summaries.
    local = args.cards.parent/'classified-contracts.jsonl'
    with args.cards.open() as cards, local.open('w') as full:
        for line in cards:
            card = json.loads(line); k = tuple(card['entry'])
            assert k in reviewed and k not in seen
            seen.add(k)
            result = classify(card, reviewed[k], placed[k])
            full.write(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
            row = {name: value for name, value in result.items() if name not in {'checks', 'conditions'}}
            row['check_count'] = len(result['checks'])
            row['check_sources'] = ';'.join(dict.fromkeys(c['source'] for c in result['checks']))
            rows.append(row)
            details.append({**{name: row[name] for name in KEY}, 'disposition': row['disposition'],
                            'relations': row['relation_ids'],
                            'examples': [{'source': c['source'], 'expression': c['expression'][:450]}
                                         for c in result['checks'][:3]]})
    assert seen == reviewed.keys()
    with (out/'all-entries.tsv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    (out/'page-details.json').write_text(json.dumps(details, ensure_ascii=False, separators=(',', ':'))+'\n')
    summary = {
        'entries': len(rows), 'dispositions': dict(Counter(r['disposition'] for r in rows)),
        'entries_with_relation': sum(bool(r['relation_ids']) for r in rows),
        'relation_counts': dict(Counter(k for r in rows for k in r['relation_ids'].split(';') if k)),
        'checked_system_property_entries': sum(r['disposition']=='reviewed_property' for r in rows),
        'new_system_property_confirmations': 0,
        'records_with_multiple_relations': sum(';' in r['relation_ids'] for r in rows),
        'index_sha256': hashlib.sha256((HERE/'full_delivery/all-entries.tsv').read_bytes()).hexdigest(),
        'labels': LABELS,
        'meaning': '检查关系来自源码表达式；系统性质采用已有审查结果。检查方式分类不自动构成系统性质归类。',
    }
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'labels','relation_counts'}}, ensure_ascii=False))


if __name__ == '__main__': main()
