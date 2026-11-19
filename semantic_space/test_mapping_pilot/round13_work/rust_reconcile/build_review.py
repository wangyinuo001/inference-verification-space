"""Attach previously source-reviewed Rust contracts to explicit property units.

The old review's per-entry decisions and source hashes are prerequisites. This
is not classification by test names, nor a claim that each assertion proves
all clauses in a family. Exact assertions and conditional source bodies remain
attached to each mapping.
"""
import csv,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PILOT=HERE.parents[1]
sys.path.insert(0,str(PILOT))
from build_round9_crosswalk import ROOTS,KEY,FIELDS,key,read
OLD=PILOT/'round9_work/rust'
contracts=json.loads((OLD/'reviewed-contracts.json').read_text())
functions=json.loads((OLD/'functions.json').read_text())
families={r['id']:r for r in json.loads((OLD/'reviewed-families.json').read_text())}
families['response.forwarded_headers'].update(name='代理请求头传递', property='代理向下游传递允许的请求头，并按具体断言保留或过滤相应字段。')
existing=set()
for name in ('confirmed-property-crosswalk.tsv','direct-property-crosswalk.tsv','inherited-property-crosswalk.tsv','reviewed-supplement-crosswalk.tsv','reviewed-round9-crosswalk.tsv','reviewed-round10-crosswalk.tsv','reviewed-round11-crosswalk.tsv','reviewed-round12-crosswalk.tsv'):
 existing.update(key(r) for r in read(PILOT/name))
existing.update(key(r) for r in read(PILOT.parent/'oracle-review-queue.tsv') if r['source_review_status']=='mapped' and r['source_reviewed_units'])
# Product anchors were checked against the pinned source. External protocol
# dependencies are explicitly represented by the gateway's re-export boundary.
VCLIENT=('rust/src/engine-core-client/src/client/imp.rs',134)
VRESP=('rust/src/server/src/routes/openai/responses.rs',18)
VREQ=('rust/src/chat/src/request.rs',684)
SAPP=('sgl-model-gateway/src/server.rs',536)
SPROTO=('sgl-model-gateway/src/lib.rs',8)
META={
 'worker.registration_configuration':('preflight','shared',['deployment_configuration','parallel_topology','remote_endpoint_profile'],{'vllm':[('rust/src/engine-core-client/src/client.rs',120),('rust/src/engine-core-client/src/client.rs',345)],'sglang':[('experimental/sgl-router/src/workers/manager.rs',206),('sgl-model-gateway/src/core/worker.rs',703)]}),
 'schedule.coordinated_progress':('scheduling','shared',['parallel_topology','request_phase','concurrent_request_profile'],{'vllm':[('rust/src/engine-core-client/src/client.rs',725),('rust/src/engine-core-client/src/coordinator/handle.rs',107)]}),
 'cache.block_identity_encoding':('cache','shared',['tokenization_profile','cache_state','request_batch_profile'],{'vllm':[('rust/src/engine-core-client/src/client.rs',729)],'sglang':[('experimental/sgl-router/src/policies/kv_events/hash.rs',57)]}),
 'cache.distributed_index_state':('cache','shared',['cache_event_profile','concurrent_request_profile','transfer_topology'],{'sglang':[('experimental/sgl-router/src/policies/kv_events/tree.rs',291),('experimental/sgl-router/src/policies/kv_events/tree.rs',650)]}),
 'service.shutdown_completion':('service','shared',['deployment_configuration','concurrent_request_profile','stream_fragment_profile'],{'sglang':[('experimental/sgl-router/src/main.rs',188)]}),
 'response.history_store':('output','shared',['request_api_kind','storage_backend_profile','semantic_response_profile'],{'sglang':[('sgl-model-gateway/src/server.rs',296),('sgl-model-gateway/src/server.rs',388)]}),
 'request.access_control':('ingress','shared',['access_policy','request_api_kind','deployment_configuration'],{'sglang':[('sgl-model-gateway/src/lib.rs',2),('sgl-model-gateway/src/server.rs',1038)]}),
}
# Retain framework-neutral names while recording which implementations are mapped.
rows=[];files={};reviewed=[]
for r,fn in zip(contracts,functions):
 assert key(r)==key(fn)
 if r['test_name']=='python_msgpack_fixtures_match_rust_encoding_and_record_nixl_metrics':continue
 if r['family'] not in META or r['decision']!='property_identified' or not r['assertions'] or key(r) in existing:continue
 root=ROOTS[r['snapshot']];raw=(root/r['path']).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==r['source_sha256']
 lines=raw.decode().splitlines()
 assert fn['source'].splitlines()[0].strip() in lines[int(r['line'])-1]
 assertions=r['assertions']
 for a in assertions:assert lines[a['line']-1].strip()
 prop='本项检查：'+'；'.join(' '.join(a['expression'].split()) for a in assertions)
 if r['review_note']: prop+='。源码核对：'+r['review_note']
 conditions={'source_body':r['condition_source'],'attributes':r['attributes'],
   'calls':r['calls'],'conditional_checks':'断言的分支、循环和异步上下文按所附完整函数体限定。',
   'upstream_review':r['source_record'],'review_basis':r['review_basis'],
   'execution':'本轮复核源码及既有审查记录，未执行上游测试。'}
 rows.append({**{k:r[k] for k in KEY},'unit_id':r['family'],'property':prop,
  'oracle_source':';'.join(r['path']+':'+str(a['line']) for a in assertions),
  'conditions':json.dumps(conditions,ensure_ascii=False,separators=(',',':')),
  'rule_id':'round13_reconciled_rust_source_contract'})
 files[(r['snapshot'],r['path'])]={'snapshot':r['snapshot'],'path':r['path'],'source_sha256':r['source_sha256']}
 reviewed.append({**r,'full_source':fn['source']})
units=[]
for uid in sorted({r['unit_id'] for r in rows}):
 phase,scope,conditions,anchors=META[uid]
 frameworks=sorted({r['snapshot'] for r in rows if r['unit_id']==uid})
 evidence={}
 for framework in frameworks:
  refs=[]
  for path,line in anchors[framework]:
   raw=(ROOTS[framework]/path).read_bytes();source=raw.decode().splitlines()[line-1];assert source.strip(),(path,line)
   refs.append({'path':path,'line':line,'kind':'source','source_sha256':hashlib.sha256(raw).hexdigest(),'source':source})
  example=next(r for r in rows if r['unit_id']==uid and r['snapshot']==framework)
  refs.append({'path':example['path'],'line':int(example['line']),'kind':'existing_test'})
  evidence[framework]=refs
 units.append({**families[uid],'kind':'obligation','phase':phase,'state_scope':scope,'conditions':conditions,
  'observable':['被断言的返回字段、状态或消息','断言目标及参考值'],
  'oracle':'按各入口记录的操作序列、输入参数、模拟边界及具体断言判定；存在性、相等、允许状态集合与时限分别保留。',
  'framework_scope':frameworks,'evidence':evidence,
  'mapping_status':'使用已逐项审查的 Rust 合约；外部协议依赖映射到声明的公开入口，不推断其内部实现。'})
with (HERE/'crosswalk.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,FIELDS,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
(HERE/'evidence.json').write_text(json.dumps({'file_reviews':list(files.values()),'entries':reviewed},ensure_ascii=False,indent=2)+'\n')
(HERE/'proposed-units.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'entries':len(rows),'units':len(units),'files':len(files)}))
