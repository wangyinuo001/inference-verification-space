"""Validate and publish source-reviewed entry contracts from the full-scope review.

Extraction manifests and proposed units alone never promote an entry. A record
must have a reviewed crosswalk, a registered unit, pinned source hashes and
resolvable definition/check sites. Unregistered unit proposals remain separate.
"""
from __future__ import annotations
import ast, csv, hashlib, json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
BATCH=HERE/'round9_work'
PROJECT=HERE.parents[3]
ROOTS={'vllm':PROJECT/'experiments/third_party/reference_repos/vllm-20261105',
       'sglang':PROJECT/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'}
KEY=('snapshot','path','test_name','line')
FIELDS=KEY+('unit_id','property','oracle_source','conditions','rule_id')
def read(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f,delimiter='\t'))
def key(r):return tuple(str(r[x]) for x in KEY)
def hashes(name):
    if name=='rust':
        return {(r['snapshot'],r['path']):r['source_sha256'] for r in json.loads((BATCH/name/'functions.json').read_text())}
    d=json.loads((BATCH/name/'evidence.json').read_text())
    if 'file_reviews' in d:
        return {(x['snapshot'],x['path']):x['source_sha256'] for x in d['file_reviews']}
    result={}
    for path,x in d['files'].items():
        result[(x.get('snapshot','sglang' if name=='sglang' else 'vllm'),path)]=x.get('source_sha256',x.get('sha256'))
    return result

def main(batch=None, label="round9", names=None, prior=()):
    global BATCH
    BATCH = batch or HERE/"round9_work"
    index={key(r):r for r in read(HERE/'full-scan-index.tsv')}
    origins={key(r):r for r in read(HERE/'inherited-method-sources.tsv')}
    existing=set()
    for name in ['confirmed-property-crosswalk.tsv','direct-property-crosswalk.tsv','inherited-property-crosswalk.tsv','reviewed-supplement-crosswalk.tsv', *prior]:
        existing.update(key(r) for r in read(HERE/name))
    existing.update(key(r) for r in read(HERE.parent/'oracle-review-queue.tsv') if r['source_review_status']=='mapped' and r['source_reviewed_units'])
    units={u['id'] for u in json.loads((HERE.parent/'space_seed.json').read_text())['units']}
    cache={};out={};pending=[];receipts=[]
    def source(snapshot,path):
        k=(snapshot,path)
        if k not in cache:
            raw=(ROOTS[snapshot]/path).read_bytes();text=raw.decode('utf-8')
            nodes={}
            if path.endswith('.py'):
                tree=ast.parse(text)
                nodes={n.lineno:n for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
            cache[k]=(hashlib.sha256(raw).hexdigest(),text.splitlines(),nodes)
        return cache[k]
    for name in (names if names is not None else ['sglang','vllm_other','vllm_runtime','rust','auto_round','low_latency_gemm','skinny_gemm']):
        if not (BATCH/name/'crosswalk.tsv').exists():continue
        expected=hashes(name)
        for filename in ['crosswalk.tsv','proposed-crosswalk.tsv']:
            path=BATCH/name/filename
            if not path.exists():continue
            for r in read(path):
                k=key(r)
                assert tuple(r)==FIELDS,(name,tuple(r))
                assert k in index,k
                assert all(r[x].strip() for x in FIELDS),k
                if k in existing:continue
                if r['unit_id'] not in units or 'candidate' in r['rule_id']:
                    pending.append({**r,'review_file':str(path.relative_to(HERE)),'reason':'已核查具体检查内容，验证空间中的新单元尚未接入'})
                    continue
                digest,lines,nodes=source(r['snapshot'],r['path'])
                assert digest==expected.get((r['snapshot'],r['path'])),('source hash',k)
                origin=origins.get(k)
                if index[k]['entry_kind']=='python_inherited_test_candidate':
                    assert origin and origin['status']=='resolved_unique',k
                    _,_,origin_nodes=source(r['snapshot'],origin['source_path'])
                    node=origin_nodes.get(int(origin['source_line']))
                    assert node and node.name==r['test_name'].split('::')[-1],k
                elif r['path'].endswith('.py'):
                    node=nodes.get(int(r['line']))
                    assert node and node.name==r['test_name'].split('::')[-1],k
                else:
                    assert 0<int(r['line'])<=len(lines),k
                check_hashes={}
                for site in r['oracle_source'].split(';'):
                    check_path,line=site.strip().rsplit(':',1)
                    check_digest,check_lines,_=source(r['snapshot'],check_path)
                    assert 0<int(line)<=len(check_lines) and check_lines[int(line)-1].strip(),(k,site)
                    check_hashes[check_path]=check_digest
                assert k not in out or out[k]==r,('conflicting review',k)
                out[k]=r
                receipts.append({'key':k,'review_file':str(path.relative_to(HERE)),'source_sha256':digest,'oracle_source_sha256':check_hashes})
    rows=sorted(out.values(),key=lambda r:(r['snapshot'],r['path'],int(r['line']),r['test_name']))
    with (HERE/f'reviewed-{label}-crosswalk.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,FIELDS,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    (HERE/f'reviewed-{label}-pending.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2)+'\n')
    (HERE/f'reviewed-{label}-receipts.json').write_text(json.dumps(receipts,ensure_ascii=False,indent=2)+'\n')
    summary={'full_scope':len(index),'new_reviewed_mappings':len(rows),'proposed_unit_mappings_pending':len(pending),'by_framework':dict(Counter(r['snapshot'] for r in rows)),'complete':False}
    (HERE/f'reviewed-{label}-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False));return summary
if __name__=='__main__':main()
