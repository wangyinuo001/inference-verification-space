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
    boundaries_path=HERE/'pending-boundaries.json'
    boundaries={(x['path'],x['test_name'].split('::')[-1]):x['reason'] for x in json.loads(boundaries_path.read_text())['entries']} if boundaries_path.exists() else {}
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
                if decision.get('source_sha256'):
                    assert decision['source_sha256']==sha, key
                info['input_and_call_source']=decision.get('input_and_call_source', [])
                for call_ref in info['input_and_call_source']:
                    assert lines[call_ref['line']-1].strip()==call_ref['source'], (key,call_ref)
                info['reviewed_definition_source']='\n'.join(lines[fn.lineno-1:fn.end_lineno])
                info['full_decorator_sources']=[{'line':d.lineno,'source':ast.get_source_segment(src,d)} for d in fn.decorator_list]
                # Retain reachable input constructors/fixtures as source context;
                # these are not automatically promoted to verification oracles.
                symbols={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
                for node in tree.body:
                    if isinstance(node,(ast.Assign,ast.AnnAssign)):
                        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
                        for target in targets:
                            for name in ast.walk(target):
                                if isinstance(name,ast.Name):symbols[name.id]=node
                wanted={n.id for n in ast.walk(fn) if isinstance(n,ast.Name)} | {a.arg for a in fn.args.args}
                wanted.update(n.name for n in tree.body if isinstance(n,ast.FunctionDef) and any(isinstance(d,ast.Call) and any(k.arg=='autouse' and isinstance(k.value,ast.Constant) and k.value.value is True for k in d.keywords) for d in n.decorator_list))
                context=[]; seen={fn.name}
                while wanted-seen:
                    name=(wanted-seen).pop();seen.add(name)
                    node=symbols.get(name)
                    if node is None:continue
                    context.append({'path':path,'function_or_class':name,'line':node.lineno,'end_line':node.end_lineno,'source_sha256':sha,'source':'\n'.join(lines[node.lineno-1:node.end_lineno])})
                    wanted.update(n.id for n in ast.walk(node) if isinstance(n,ast.Name))
                if 'make_moriio_writer' in wanted:
                    context_path='tests/v1/kv_connector/unit/utils.py';cb=(ROOT/context_path).read_bytes();cl=cb.decode().splitlines()
                    cn=next(n for n in ast.parse(cb.decode()).body if isinstance(n,ast.FunctionDef) and n.name=='make_moriio_writer')
                    context.append({'path':context_path,'function_or_class':cn.name,'line':cn.lineno,'end_line':cn.end_lineno,'source_sha256':hashlib.sha256(cb).hexdigest(),'source':'\n'.join(cl[cn.lineno-1:cn.end_lineno])})
                if path.startswith('tests/v1/kv_connector/unit/'):
                    for fixture_path in ['tests/v1/kv_connector/conftest.py','tests/v1/kv_connector/unit/conftest.py']:
                        fb=(ROOT/fixture_path).read_bytes()
                        context.append({'path':fixture_path,'source_sha256':hashlib.sha256(fb).hexdigest(),'source':fb.decode(),'role':'Reviewed inherited fixture context; CPU backend block-size declaration is mocked to MultipleOf(16)'})
                info['module_test_guards']=[{'line':n.lineno,'source':ast.get_source_segment(src,n)} for n in tree.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pytestmark' for t in n.targets)) or (isinstance(n,ast.If) and any(isinstance(v,ast.Call) and ast.unparse(v.func)=='pytest.skip' for v in ast.walk(n)))]
                info['input_helper_context']=context
                info['reviewed_extra_source_files']=[]
                for ref in decision.get('reviewed_extra_source_files',[]):
                    rb=(ROOT/ref['path']).read_bytes();assert hashlib.sha256(rb).hexdigest()==ref['source_sha256'], (key,ref)
                    info['reviewed_extra_source_files'].append({**ref,'source':rb.decode()})
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
                    call=next((n for n in ast.walk(fn) if isinstance(n,ast.Call) and n.lineno==helper['call_line'] and ast.unparse(n)==helper['call_expression']),None)
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
                if decision.get('helper_call_graph'):
                    for edge in decision['helper_call_graph']:
                        eb=(ROOT/edge['path']).read_bytes()
                        assert hashlib.sha256(eb).hexdigest()==edge['source_sha256'], (key,edge)
                        et=ast.parse(eb.decode())
                        caller=next(n for n in ast.walk(et) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno==edge['caller_definition_line'])
                        ec=next(n for n in ast.walk(caller) if isinstance(n,ast.Call) and n.lineno==edge['call_line'] and ast.unparse(n)==edge['call_expression'])
                        assert any(isinstance(n,ast.Name) and n.id==edge['callee'] for n in ast.walk(ec)), (key,edge)
                    info['helper_call_graph']=decision['helper_call_graph']
            else:
                reason=('No direct assertion in test body; helper/fixture oracle needs review' if not checks
                        else 'Direct checks located; target, input/state and asserted property need semantic review')
                if (path,x['test_name'].split('::')[-1]) in boundaries:
                    reason=boundaries[(path,x['test_name'].split('::')[-1])]
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
