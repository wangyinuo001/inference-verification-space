"""Pinned-source audit scaffold for unresolved vLLM runtime test entries.

This module locates source and direct checks. It deliberately does not infer a
verification property from a file path, test name, or the existence of assert.
Curated file-group decisions live in REVIEWED below and must cite direct checks.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
PROJECT = PILOT.parents[3]
ROOT = PROJECT / 'experiments/third_party/reference_repos/vllm-20261105'
ALL = PILOT / 'full_delivery/all-entries.tsv'
FIELDS = ('snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id')

# Source-reviewed cohorts. Add only after checking complete bodies and shared
# helpers; mappings must be narrow enough for every matching entry.
def read_decisions():
    path = HERE / 'reviewed-decisions.json'
    if not path.exists():
        return {}
    return {(row['path'], row['test_name']): row
            for row in json.loads(path.read_text())['decisions']}

def read_rows():
    rows=list(csv.DictReader(ALL.open(),delimiter='\t'))
    # Keep this agent's reviewed decisions in the cumulative crosswalk even
    # after the main ledger incorporates an earlier batch. The main merger
    # rebuilds from agent outputs, so dropping accepted rows would lose them.
    own_decisions=set(read_decisions())
    return [x for x in rows if x['snapshot']=='vllm'
            and (x['path'].startswith('tests/v1/') or x['path'].startswith('tests/distributed/'))
            and (x['review_status'] not in ('property_reviewed','outside_current_scope')
                 or (x['path'],x['test_name']) in own_decisions)]

def direct_checks(fn: ast.AST, lines: list[str]):
    out=[]
    for n in ast.walk(fn):
        if isinstance(n,ast.Assert): out.append({'line':n.lineno,'source':lines[n.lineno-1].strip(),'expression':ast.unparse(n)[:500], 'kind':'assert'})
        elif isinstance(n,ast.With):
            for item in n.items:
                if isinstance(item.context_expr,ast.Call) and ast.unparse(item.context_expr.func).endswith('raises'):
                    out.append({'line':n.lineno,'source':lines[n.lineno-1].strip(),'expression':ast.unparse(item.context_expr)[:500],'kind':'expected_exception'})
        elif isinstance(n,ast.Call):
            expr=ast.unparse(n)
            mock_check = isinstance(n.func, ast.Attribute) and n.func.attr in {
                'assert_called', 'assert_called_once', 'assert_called_with',
                'assert_called_once_with', 'assert_not_called', 'assert_has_calls',
                'assert_any_call', 'assert_awaited', 'assert_awaited_once_with'}
            if mock_check or expr.startswith(('torch.testing.assert_close','np.testing.assert','numpy.testing.assert','self.assert','mock.assert','pytest.approx')):
                out.append({'line':n.lineno,'source':lines[n.lineno-1].strip(),'expression':expr[:500],'kind':'direct_oracle_call'})
    seen=set();result=[]
    for x in sorted(out,key=lambda v:(v['line'],v['kind'])):
        key=(x['line'],x['expression'])
        if key not in seen:result.append(x);seen.add(key)
    return result

def main():
    rows=read_rows(); bypath=defaultdict(list)
    reviewed=read_decisions()
    for x in rows:bypath[x['path']].append(x)
    evidence={'source_root':str(ROOT),'scope':'vLLM tests/v1 and tests/distributed unresolved entries',
              'method':'AST locates complete definition boundaries, direct checks and source hashes. Property mapping requires additional file-group semantic review.',
              'files':{},'entries':[]}
    mapped=[];pending=[]
    for path,group in sorted(bypath.items()):
        p=ROOT/path
        if not p.exists():
            for x in group:pending.append({**{k:x[k] for k in ('snapshot','path','test_name','line')},'reason':'pinned source file missing','detail':''})
            continue
        data=p.read_bytes();sha=hashlib.sha256(data).hexdigest();src=data.decode();lines=src.splitlines()
        try:tree=ast.parse(src)
        except SyntaxError:
            for x in group:pending.append({**{k:x[k] for k in ('snapshot','path','test_name','line')},'reason':'source AST parse failed','detail':''})
            continue
        funcs={n.lineno:n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        fileentry={'snapshot':'vllm','path':path,'source_sha256':sha,'entry_count':len(group),'entries':[]}
        evidence['files'][path]=fileentry
        for x in group:
            base={k:x[k] for k in ('snapshot','path','test_name','line')}
            fn=funcs.get(int(x['definition_line'])) if x['definition_line'] else None
            if fn is None:
                pending.append({**base,'reason':'test definition not located in pinned source AST','detail':x['source_status']})
                fileentry['entries'].append({**base,'source_status':'unlocated','review_status':'pending'})
                continue
            checks=direct_checks(fn,lines)
            decorators=[]
            for d in fn.decorator_list:
                seg=ast.get_source_segment(src,d)
                if seg and ('parametrize' in seg or 'skipif' in seg or 'requires_' in seg):decorators.append(' '.join(seg.split())[:260])
            info={**base,'definition_line':fn.lineno,'end_line':fn.end_lineno,
                  'source_sha256':sha,'direct_checks':checks,'parameter_signature':ast.unparse(fn.args),
                  'decorators':decorators,'source_status':'definition_read'}
            fileentry['entries'].append(info)
            key=(path,x['test_name'])
            if key in reviewed:
                decision=reviewed[key]
                unit,prop,conditions=(decision[k] for k in ('unit_id','property','conditions'))
                cited=decision['oracle_lines']
                assert all(any(c['line']==line for c in checks) for line in cited), key
                citations=[f'{path}:{line}' for line in cited]
                helper_evidence=[]
                for helper in decision.get('helper_oracles', []):
                    hp=ROOT/helper['path']; hb=hp.read_bytes(); hs=hb.decode()
                    ht=ast.parse(hs); hl=hs.splitlines()
                    matches=[n for n in ast.walk(ht) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==helper['function']]
                    assert len(matches)==1, (key,helper)
                    hf=matches[0]; hc=direct_checks(hf,hl)
                    assert helper['oracle_lines'] and all(any(c['line']==line for c in hc) for line in helper['oracle_lines']), (key,helper)
                    call=next((n for n in ast.walk(fn) if isinstance(n,ast.Call) and n.lineno==helper['call_line']),None)
                    assert call is not None and ast.unparse(call)==helper['call_expression'], (key,helper)
                    assert helper['binding'], (key,helper)
                    helper_evidence.append({**helper,'source_sha256':hashlib.sha256(hb).hexdigest(),
                                            'definition_line':hf.lineno,'end_line':hf.end_lineno,
                                            'checks':[c for c in hc if c['line'] in helper['oracle_lines']]})
                    citations.extend(f'{helper["path"]}:{line}' for line in helper['oracle_lines'])
                assert citations, key
                cites=';'.join(citations)
                mapped.append({**base,'unit_id':unit,'property':prop,'oracle_source':cites,
                               'conditions':conditions,'rule_id':decision['rule_id']})
                info['review_status']='source_reviewed';info['unit_id']=unit
                info['asserted_property']=prop;info['conditions']=conditions
                info['oracle_source']=citations
                if helper_evidence:info['helper_oracles']=helper_evidence
                if decision.get('helper_transport_evidence'):
                    for ref in decision['helper_transport_evidence']:
                        rb=(ROOT/ref['path']).read_bytes()
                        assert hashlib.sha256(rb).hexdigest()==ref['source_sha256'], (key,ref)
                        assert rb.decode().splitlines()[ref['line']-1].strip()==ref['source'], (key,ref)
                    info['helper_transport_evidence']=decision['helper_transport_evidence']
            else:
                reason=('No direct assertion in test body; helper/fixture oracle needs review' if not checks
                        else 'Direct checks located; target, input/state and asserted property need semantic review')
                pending.append({**base,'reason':reason,'detail':'; '.join(f'{c["line"]}:{c["kind"]}' for c in checks[:10])})
                info['review_status']='pending'
    assert len(rows)==len(mapped)+len(pending)
    with (HERE/'crosswalk.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,FIELDS,delimiter='\t');w.writeheader();w.writerows(mapped)
    with (HERE/'pending.tsv').open('w',newline='') as f:
        fields=('snapshot','path','test_name','line','reason','detail');w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(pending)
    evidence['counts']={'input':len(rows),'mapped':len(mapped),'pending':len(pending),'files':len(evidence['files'])}
    (HERE/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    if not (HERE/'needs-space.json').exists():
        (HERE/'needs-space.json').write_text(json.dumps({'proposals':[],'note':'No new unit is asserted until both vLLM and SGLang source evidence is checked.'},ensure_ascii=False,indent=2)+'\n')
    print(evidence['counts'])

if __name__=='__main__':main()
