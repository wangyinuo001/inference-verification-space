"""Build source-backed review batches, without promoting semantic review status.

Borrow vstack's separation of promise, assumptions and evidence. The seven
pillars are NOT assigned from keywords: a comparison alone does not establish
isolation, integrity, or equivalence of whole executions. Batch identity is a
reading aid; argument values and conditions remain in each individual record.
No upstream code is imported or executed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from resolve_inherited_methods import ROOTS

HERE = Path(__file__).resolve().parent
KEY = ('snapshot', 'path', 'test_name', 'line')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = call_name(node.value)
        return f'{base}.{node.attr}' if base else ''
    return ''


def own_nodes(node):
    """Exclude uncalled nested functions/classes and lambda bodies."""
    yield node
    children = node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else ast.iter_child_nodes(node)
    for child in children:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield from own_nodes(child)


def check_kind(node):
    if isinstance(node, ast.Assert):
        return 'predicate'
    if not isinstance(node, ast.Call):
        return ''
    name = call_name(node.func)
    tail = name.rsplit('.', 1)[-1]
    if name in {'pytest.raises', 'self.assertRaises', 'self.assertRaisesRegex'}:
        return 'expected_exception'
    if name in {'pytest.warns', 'self.assertWarns', 'self.assertWarnsRegex'}:
        return 'expected_warning'
    if name.startswith(('torch.testing.assert', 'np.testing.assert', 'numpy.testing.assert')):
        return 'numerical_comparison'
    if name.startswith('self.assert'):
        return 'assertion_method'
    if tail.startswith('assert_'):
        return 'assertion_helper_call'
    return ''


def ast_shape(node):
    """Keep identifiers/operators; values are stored separately in each card."""
    class Shape(ast.NodeTransformer):
        def visit_Constant(self, n):
            return ast.Constant(value=f'<{type(n.value).__name__}>')
    return ast.dump(Shape().visit(ast.parse(ast.unparse(node))), include_attributes=False)


def extract_function(node, path, helpers=None, depth=0, seen=()):
    helpers = helpers or {}
    checks, calls, guards, local_helpers = [], [], [], []
    for n in own_nodes(node):
        if n is node:
            continue
        kind = check_kind(n)
        if kind:
            checks.append({'source': f'{path}:{n.lineno}', 'kind': kind,
                           'expression': ast.unparse(n), 'structure': ast_shape(n)})
        if isinstance(n, ast.Call):
            name = call_name(n.func)
            calls.append({'source': f'{path}:{n.lineno}', 'name': name or '<dynamic>',
                          'expression': ast.unparse(n)})
            if name in helpers and name not in seen and depth < 1:
                helper = helpers[name]
                local_helpers.append({'call_site': f'{path}:{n.lineno}', 'name': name,
                                      'body': extract_function(helper, path, helpers, depth+1, (*seen, name))})
        if isinstance(n, (ast.If, ast.While)):
            guards.append({'source': f'{path}:{n.lineno}', 'kind': type(n).__name__,
                           'expression': ast.unparse(n.test)})
        if isinstance(n, (ast.With, ast.AsyncWith)):
            guards.extend({'source': f'{path}:{n.lineno}', 'kind': 'context',
                           'expression': ast.unparse(item.context_expr)} for item in n.items)
    return {'parameters': ast.unparse(node.args),
            'decorators': [ast.unparse(n) for n in node.decorator_list],
            'checks': checks, 'calls': calls, 'guards': guards,
            'local_helpers': local_helpers}


def parse_source(path):
    raw = path.read_bytes()
    result = {'hash': hashlib.sha256(raw).hexdigest(), 'functions': {}, 'parents': {},
              'helpers': {}, 'imports': [], 'status': 'non_python'}
    if path.suffix != '.py':
        return result
    try:
        tree = ast.parse(raw.decode('utf-8'))
    except (SyntaxError, UnicodeError) as exc:
        result['status'] = f'parse_error:{type(exc).__name__}'
        return result
    result['status'] = 'parsed'
    result['functions'] = {n.lineno: n for n in ast.walk(tree)
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for n in cls.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result['parents'][n.lineno] = cls
    # Follow only uniquely defined module helpers whose name is not rebound in
    # the module. self helpers and fixtures need contextual resolution.
    bindings = Counter()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[n.name] += 1
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for target in targets:
                for part in ast.walk(target):
                    if isinstance(part, ast.Name): bindings[part.id] += 1
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            result['imports'].append(ast.unparse(n))
            for alias in n.names: bindings[alias.asname or alias.name.split('.')[0]] += 1
    result['helpers'] = {n.name: n for n in tree.body
                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and bindings[n.name] == 1}
    return result


def make_card(row, source):
    node = source['functions'].get(int(row['definition_line'] or 0))
    card = {'entry': [row[k] for k in KEY], 'source_sha256': source['hash'],
            'definition_path': row['definition_path'], 'definition_line': row['definition_line'],
            'existing_review_status': row['review_status'], 'reviewed_units': row['unit_ids'],
            'reviewed_property': row['property'], 'object_candidates': row['object_candidates'],
            'extraction_status': source['status'], 'evidence': None}
    if not node or node.name != row['test_name'].split('::')[-1]:
        card['extraction_status'] = 'definition_needs_resolution' if source['status'] == 'parsed' else source['status']
        card['batch_id'] = digest(json.dumps(card['entry']))[:20]
        return card
    # A parameter/local assignment may shadow a module-level helper.
    local = {n.id for n in own_nodes(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    local.update(n.arg for n in ast.walk(node.args) if isinstance(n, ast.arg))
    helpers = {k: v for k, v in source['helpers'].items() if k not in local and k != node.name}
    evidence = extract_function(node, row['definition_path'], helpers, seen=(node.name,))
    cls = source['parents'].get(node.lineno)
    evidence['class_context'] = {'name': cls.name, 'bases': [ast.unparse(b) for b in cls.bases],
                                  'source_line': cls.lineno} if cls else None
    evidence['imports'] = source['imports']
    card['evidence'] = evidence
    card['extraction_status'] = 'direct_checks' if evidence['checks'] else (
        'local_helper_checks' if any(h['body']['checks'] for h in evidence['local_helpers']) else 'needs_helper_or_observation')
    # Same-file/context grouping prevents cross-framework or fixture identity
    # assumptions. Each member retains its own decorators, values and guards.
    signature = [row['snapshot'], row['definition_path'], cls.name if cls else '',
                 sorted({c['name'] for c in evidence['calls']}),
                 [c['structure'] for c in evidence['checks']],
                 [h['name'] for h in evidence['local_helpers']]]
    if not evidence['checks'] and not evidence['local_helpers']:
        signature.append(card['entry'])
    card['batch_id'] = digest(json.dumps(signature, ensure_ascii=False))[:20]
    return card


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=Path, default=HERE/'full_delivery/all-entries.tsv')
    parser.add_argument('--output', type=Path,
                        default=ROOTS['vllm'].parents[2]/'results/verification-contract-batches')
    args = parser.parse_args()
    started = time.monotonic()
    rows = list(csv.DictReader(args.index.open(), delimiter='\t'))
    assert len(rows) == len({tuple(r[k] for k in KEY) for r in rows})
    cache, batches, stats = {}, defaultdict(list), Counter()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    with (args.output/'entries.jsonl').open('w') as out:
        for row in rows:
            k = row['snapshot'], row['definition_path'] or row['path']
            if k not in cache:
                path = ROOTS[k[0]] / k[1]
                try:
                    cache[k] = parse_source(path)
                except OSError:
                    cache[k] = {'status': 'source_missing', 'hash': '', 'functions': {}}
            card = make_card(row, cache[k])
            stats[card['extraction_status']] += 1
            checks = card['evidence']['checks'] if card['evidence'] else []
            batches[card['batch_id']].append({
                'entry': card['entry'], 'review_status': row['review_status'],
                'units': row['unit_ids'], 'check_count': len(checks),
                'extraction_status': card['extraction_status']})
            out.write(json.dumps(card, ensure_ascii=False, separators=(',', ':'))+'\n')
    for batch, members in batches.items():
        reviewed = [x for x in members if x['review_status'] == 'property_reviewed']
        pending = [x for x in members if x['review_status'] not in {
            'property_reviewed', 'supporting_record_reviewed', 'outside_current_scope', 'incomplete_test_reviewed'}]
        manifest.append({'batch_id': batch, 'size': len(members), 'pending': len(pending),
                         'reviewed_examples': reviewed, 'members': members})
    manifest.sort(key=lambda b: (-b['pending'], -len(b['reviewed_examples']), b['batch_id']))
    (args.output/'batches.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    summary = {'entries': len(rows), 'source_files_read': len(cache), 'status': dict(stats),
               'batches': len(manifest), 'pending_entries': sum(b['pending'] for b in manifest),
               'pending_batches': sum(b['pending'] > 0 for b in manifest),
               'pending_in_batches_with_reviewed_example': sum(b['pending'] for b in manifest if b['reviewed_examples']),
               'existing_property_reviewed': sum(r['review_status']=='property_reviewed' for r in rows),
               'new_property_reviewed': 0,
               'elapsed_seconds': round(time.monotonic()-started, 2),
               'input_sha256': hashlib.sha256(args.index.read_bytes()).hexdigest(),
               'meaning': '按实际源码提取检查表达式并组织审查批次；批次相同不表示测试等价，保留原有归类状态。'}
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
