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
for name in ('confirmed-property-crosswalk.tsv','direct-property-crosswalk.tsv','inherited-property-crosswalk.tsv','reviewed-supplement-crosswalk.tsv','reviewed-round9-crosswalk.tsv','reviewed-round10-crosswalk.tsv'):
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
 'request.declared_admission_contract':('ingress','request',['request_api_kind','input_error_profile','deployment_configuration'],{'vllm':[VREQ,('rust/src/server/src/routes/openai/responses/validate.rs',18)],'sglang':[SAPP,SPROTO]}),
 'request.field_binding':('ingress','request',['request_api_kind','parameter_source_rule','request_batch_profile'],{'vllm':[('rust/src/llm/src/request.rs',1),VRESP],'sglang':[SPROTO,SAPP]}),
 'response.observed_payload_contract':('output','request',['semantic_response_profile','output_mode','stream_fragment_profile'],{'vllm':[VRESP,('rust/src/engine-core-client/src/client/stream.rs',31)],'sglang':[SAPP,('sgl-model-gateway/src/routers/openai/responses/streaming.rs',58)]}),
 'distributed.control_delivery':('transfer','shared',['message_transport_profile','parallel_topology'],{'vllm':[VCLIENT,('rust/src/engine-core-client/src/client/imp.rs',377)]}),
 'service.lifecycle_health':('service','shared',['deployment_configuration','concurrent_request_profile'],{'vllm':[('rust/src/engine-core-client/src/client/imp.rs',237)],'sglang':[SAPP,('sgl-model-gateway/src/core/worker.rs',703)]}),
 'service.failure_admission_recovery':('service','shared',['concurrent_request_profile','remote_endpoint_profile','deployment_configuration'],{'sglang':[('experimental/sgl-router/src/health/circuit_breaker.rs',124),('sgl-model-gateway/src/core/circuit_breaker.rs',148),SAPP]}),
 'service.inflight_accounting':('service','shared',['concurrent_request_profile','shared_resource_lifecycle_profile'],{'sglang':[('sgl-model-gateway/src/observability/inflight_tracker.rs',44),('sgl-model-gateway/src/core/worker.rs',1115)]}),
 'service.measurement_record':('service','shared',['workload_profile','remote_endpoint_profile','deployment_configuration'],{'vllm':[('rust/src/llm/src/request_metrics.rs',25)],'sglang':[('sgl-model-gateway/src/core/metrics_aggregator.rs',15),('experimental/sgl-router/src/server/routes/chat.rs',86)]}),
 'service.endpoint_observed_fields':('service','request',['request_api_kind','remote_endpoint_profile','deployment_configuration'],{'sglang':[SAPP]}),
 'service.configured_request_success':('service','request',['request_api_kind','routing_profile','deployment_configuration'],{'sglang':[SAPP]}),
 'abort.request_id_binding':('abort','request',['request_api_kind','concurrent_request_profile'],{'vllm':[('rust/src/engine-core-client/src/client/imp.rs',427),('rust/src/llm/src/lib.rs',330)]}),
 'response.batch_metadata_binding':('output','request',['output_mode','request_batch_profile','stream_fragment_profile'],{'vllm':[('rust/src/engine-core-client/src/client/stream.rs',31),('rust/src/llm/src/output.rs',1)]}),
 'response.failure_contract':('output','request',['remote_endpoint_profile','deployment_configuration'],{'sglang':[('experimental/sgl-router/src/server/routes/chat.rs',96)]}),
 'response.forwarded_headers':('ingress','request',['request_api_kind','remote_endpoint_profile'],{'sglang':[('experimental/sgl-router/src/server/header_utils.rs',10)]}),
 'service.extension_registration':('service','shared',['deployment_configuration','request_api_kind'],{'sglang':[('sgl-model-gateway/src/wasm/route.rs',74),('sgl-model-gateway/src/wasm/route.rs',159)]}),
 'embedding.routing_text':('ingress','request',['input_form','routing_profile'],{'sglang':[SPROTO]}),
}
# Retain framework-neutral names while recording which implementations are mapped.
rows=[];files={};reviewed=[]
for r,fn in zip(contracts,functions):
 assert key(r)==key(fn)
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
  'rule_id':'round11_reconciled_rust_source_contract'})
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
