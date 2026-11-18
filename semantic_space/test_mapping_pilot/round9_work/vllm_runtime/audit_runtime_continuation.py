"""Revalidate cumulative static-source artifacts; does not execute test sources."""
import csv,json,hashlib,collections,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent
import build_runtime_review as b
R=b.ROOT;G=R.parent/'sglang-v0.5.14-complete'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=G,text=True).strip()=='49e384ce9d304648e9959666ecb8ce8cd98d0deb'
rows=list(csv.DictReader((H/'crosswalk.tsv').open(),delimiter='\t'));props=list(csv.DictReader((H/'proposed-crosswalk.tsv').open(),delimiter='\t'))
keys=[(r['snapshot'],r['path'],r['test_name']) for r in rows+props];assert len(keys)==len(set(keys))
seed=json.load(open(H.parents[2]/'space_seed.json'));units={u['id'] for u in seed['units']} | {u['id'] for u in json.load((H/'units-ready.json').open())['units']};assert all(r['unit_id'] in units for r in rows+props)
refs=set()
for row in rows+props:
 for ref in row['oracle_source'].split(';'):
  path,line=ref.rsplit(':',1);line=int(line);assert (R/path).read_text().splitlines()[line-1].strip(),ref;refs.add((path,line))
e=json.load((H/'evidence.json').open());fs=e['files'];assert len(rows)==e['counts']['mapped']
for path,info in fs.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==info['source_sha256'],path
ledger=json.load((H/'reviewed-decisions.json').open())['decisions'];assert len(ledger)==len(rows)
for d in ledger:
 if d.get('source_sha256'):assert hashlib.sha256((R/d['path']).read_bytes()).hexdigest()==d['source_sha256']
sg=json.load((H/'seed-evidence-replacements.json').open());sgrefs=[]
for u in sg['units']:
 for ref in u['evidence']['sglang']:
  rb=(G/ref['path']).read_bytes();assert hashlib.sha256(rb).hexdigest()==ref['source_sha256'];assert rb.decode().splitlines()[ref['line']-1].strip()==ref['source'];sgrefs.append(ref)
old=json.load((H/'continuation-audit.json').open())
new=[d for d in ledger if d['rule_id']=='runtime_source_review_continuation_20261116']
old.update(crosswalk_entries=len(rows),proposed_entries=len(props),total_unique_entries=len(keys),new_crosswalk_entries=len(rows)-81,new_proposed_entries=len(props)-179,new_total_entries=len(keys)-260,previous_records_retained=260,helper_bound_new_entries=sum(bool(d.get('helper_oracles')) for d in new),new_by_file=dict(collections.Counter(d['path'] for d in new)),oracle_references_checked=len(refs),source_files_rehashed=len(fs),pending_count=e['counts']['pending'],sglang_snapshot_commit=sg['snapshot_commit'],sglang_corrected_unit_count=len(sg['units']),sglang_references_revalidated=len(sgrefs),vllm_snapshot_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip())
if (H/'storage-layout-proposals.json').exists():
 sp=json.load((H/'storage-layout-proposals.json').open())
 for entry in sp['entries']:
  assert hashlib.sha256((R/entry['path']).read_bytes()).hexdigest()==entry['source_sha256']
  for helper in entry['helper_oracles']:
   assert hashlib.sha256((R/helper['path']).read_bytes()).hexdigest()==helper['source_sha256']
   for check in helper['checks']:assert (R/helper['path']).read_text().splitlines()[check['line']-1].strip()==check['source']
 for ref in sp['unit']['evidence']['vllm']:
  rb=(R/ref['path']).read_bytes();assert hashlib.sha256(rb).hexdigest()==ref['source_sha256'];assert rb.decode().splitlines()[ref['line']-1].strip()==ref['source']
 old['storage_layout_proposal_entries']=len(sp['entries'])
 if sp['unit']['id'] not in old['new_units']:old['new_units'].append(sp['unit']['id'])
(H/'continuation-audit.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
print({k:old[k] for k in ['crosswalk_entries','proposed_entries','total_unique_entries','new_total_entries','oracle_references_checked','source_files_rehashed','sglang_references_revalidated']})
