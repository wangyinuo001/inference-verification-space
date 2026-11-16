# -*- coding: utf-8 -*-
"""Complete 52-case manual source review, including local and comparison helpers.
AST transcribes explicit reviewed cases; it does not discover semantic mappings.
"""
import ast,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[5]/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
PATH='test/registered/unit/utils/test_weight_checker.py'
IMPL='python/sglang/srt/utils/weight_checker.py'
COLS='snapshot path test_name line unit_id property oracle_source conditions rule_id'.split()
CASES={
135:('probe_generation','For float32/float16/bfloat16 inputs, generated tensor preserves dtype/shape/device and has positive absolute sum.'),
144:('probe_generation','1024-element bool probe preserves tensor metadata and includes both true and false values.'),
153:('probe_generation','int8/int32/int64 probes preserve dtype/shape, fall within dtype bounds, and have more than one distinct value.'),
164:('probe_generation','1024-element float32 probe values lie in [0,1).'),
170:('probe_generation','Generating a probe leaves the input tensor of sixteen fives unchanged.'),
186:('comparison_projection','Nonquantized weight and bias retain names/order/values and compare=True.'),
195:('comparison_projection','Weight without matching inverse scale remains a raw compare=True entry.'),
202:('comparison_projection','cos_sin_cache-name tensor is retained with compare=False; ordinary layer weight remains compare=True.'),
217:('comparison_projection','rotary inv_freq-name tensor is retained with compare=False.'),
224:('comparison_projection','gate _weight_fp32-name tensor is retained with compare=False.'),
231:('comparison_projection','Skip pattern inside weird.cos_sin_cache.foo.bar, not only a suffix, yields compare=False.'),
241:('comparison_projection','FP8 weight with int32 packed UE8M0 scale projects to BF16 dequant compare=True followed by original weight/scale compare=False; reference uses the same block_quant_dequant helper with original FP32 scale.'),
259:('comparison_projection','FP8 weight with FP32 scale projects to BF16 dequant then raw weight/scale marked false; reference shares the block_quant_dequant implementation, so it does not independently validate that helper.'),
275:('comparison_projection','Quantized pair dequant entry precedes raw weight and scale, followed by unquantized bias; flags and values follow the explicit expected stream.'),
297:('comparison_projection','Inverse scale without matching weight is kept as ordinary compare=True data.'),
314:('comparison_verdict','Two equal named tensors marked compare=True pass _check_tensors strict metadata and exact-value error gate.'),
320:('comparison_verdict','Ones versus zeros marked compare=True raises an exception whose text contains name=a and max_abs_err.'),
329:('comparison_verdict','Ones versus zeros marked compare=False is informational and does not trigger _check_tensors error gate.'),
335:('comparison_verdict','Different tensor names a/b raise AssertionError even with equal values.'),
341:('comparison_verdict','Different compare flags for same tensor raise AssertionError.'),
347:('comparison_verdict','Unequal stream lengths two/one raise ValueError through strict zip.'),
377:('snapshot_isolation','Snapshot contains both parameters and all four named buffers, including later comparison-skip buffers.'),
390:('snapshot_isolation','CUDA-model snapshot lives on CPU and stays unchanged after live w is overwritten with 99.'),
403:('probe_reset','Reset changes normal parameter w values while preserving its storage pointer.'),
411:('probe_reset','Reset preserves rotary_emb_cos_sin_cache tensor values.'),
416:('probe_reset','Reset preserves rotary_emb_freqs_cis tensor values.'),
421:('probe_reset','Reset preserves gate_proj_weight_fp32_cache tensor values.'),
429:('comparison_verdict','Comparing before taking a snapshot raises AssertionError.'),
433:('comparison_verdict','Snapshot followed by unchanged model passes exact-value comparison gate.'),
437:('comparison_verdict','Snapshot followed by reset fails comparison and names normal parameter w or b in the error.'),
445:('comparison_verdict','Overwriting only skipped rotary cosine/sine cache with 99 does not trigger comparison failure.'),
452:('comparison_verdict','Snapshot, reset, then manual restoration of all parameters and buffers passes comparison; actual loop restores skipped buffers too.'),
468:('action_dispatch','snapshot/reset_tensors/compare/checksum actions call their corresponding mocked methods exactly once.'),
486:('action_dispatch','Snapshot action then compare action return None.'),
490:('checksum_record','Checksum action returns a dictionary with checksums, per_gpu_checksum and parallelism_info.'),
497:('action_dispatch','Unknown nonsense_action raises an exception containing Unsupported.'),
510:('comparison_projection','Name predicate matches model.rotary_emb.cos_sin_cache.'),
515:('comparison_projection','Name predicate matches model.rotary_emb.inv_freq.'),
518:('comparison_projection','Name predicate matches model.rotary_emb.freqs_cis.'),
521:('comparison_projection','Name predicate matches model.layers.0.mlp.gate._weight_fp32.'),
526:('comparison_projection','Name predicate excludes ordinary mlp.weight and embed_tokens.weight.'),
538:('checksum_record','Tensor hash agrees between arange(64,float32) and its CUDA clone.'),
542:('checksum_record','Tensor hash differs between the chosen all-zero and all-one CUDA tensors; no general collision freedom claim.'),
547:('checksum_record','Hash of zero CUDA tensor has length sixteen and is parseable base sixteen.'),
553:('checksum_record','Hashing arange CUDA tensor leaves its values unchanged.'),
584:('checksum_record','Checksum dictionary has exactly checksums, per_gpu_checksum and parallelism_info keys.'),
590:('checksum_record','Checksum names include w,b,running_mean and exclude the three declared nonpersistent-pattern buffers.'),
602:('checksum_record','Every per-tensor checksum has sixteen characters and is parseable base sixteen.'),
608:('checksum_record','Checksum parallelism metadata matches fake runner TP=2/4, DP=1/2, PP=0/1 and contains rank/size keys; rank/size values are not asserted.'),
620:('checksum_record','Two checksum records for unchanged model are equal.'),
625:('checksum_record','Overwriting w with 99 changes that parameter checksum.'),
632:('checksum_record','Checksum record validates as ChecksumInfo with ParallelismInfo nested model.'),
}
UNITS={
'probe_generation':('诊断随机探针保持张量属性','为权重诊断生成的探针保持输入张量的类型、形状与设备，遵守声明的取值域且不改写输入。',209),
'comparison_projection':('权重诊断比较表示与过滤','按名称与量化格式生成有序的待比较表示，保留逻辑名字、比较标志与数值；声明排除项仍可用于诊断但不决定相等失败。',229),
'comparison_verdict':('权重诊断比较判定','比较流的名字、标志和长度必须一致；需要比较的张量按声明相等谓词决定失败并提供对象定位，排除项差异不触发该失败。',164),
'snapshot_isolation':('权重诊断快照隔离','诊断快照包含指定参数与缓冲区，保留采集时值并与后续模型状态变更隔离。',70),
'probe_reset':('权重诊断扰动保持存储与排除项','诊断扰动按声明原地改写正常参数，保持存储身份并保留排除缓冲区的值。',79),
'action_dispatch':('权重诊断动作分派','诊断接口将声明动作映射到对应处理器并返回其结果，未知动作给出明确错误。',57),
'checksum_record':('权重诊断校验记录对应内容与拓扑','诊断校验记录包含被纳入的张量散列与执行拓扑元数据；固定输入散列稳定、具体内容变化可观察且不改写输入。',103),
}

def main():
 if any(c.get('family') == 'weight_checker_explicit_source_review' for c in json.loads((HERE/'evidence.json').read_text()).get('reviewed_clusters', [])):
  print('Cumulative review already recorded; no overwrite.'); return
 data=(SOURCE/PATH).read_bytes();text=data.decode();tree=ast.parse(text)
 nodes={n.lineno:(c.name,n) for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
 helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_assert_triples_close')
 def assertions(node):return sorted([x for x in ast.walk(node) if isinstance(x,ast.Assert) or isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr.startswith('assert')],key=lambda n:n.lineno)
 output=HERE/'proposed-crosswalk.tsv';old=list(csv.DictReader(output.open(),delimiter='\t'))
 key=lambda r:tuple(r[c] for c in ('path','test_name','line','unit_id','oracle_source'))
 seen={key(r) for r in old};new=[];audits=[];selected=set();examples={}
 for ln,(suffix,meaning) in CASES.items():
  cls,n=nodes[ln];name=cls+'::'+n.name;uid='diagnostic.weight_'+suffix
  checks=[(PATH,x.lineno,' '.join(ast.get_source_segment(text,x).split())) for x in assertions(n)]
  if ln in (186,195,202,217,224,231,241,259,275,297):
   checks += [(PATH,x.lineno,'Helper predicate bound to this test actual/expected streams: '+' '.join(ast.get_source_segment(text,x).split())) for x in assertions(helper)]
  if ln in (314,329,433,445,452):
   checks += [(IMPL,179,'Comparison helper requires matching tensor names.'),(IMPL,180,'Comparison helper requires matching should_compare flags.'),(IMPL,206,'Comparison helper raises iff any compare=True tensor differs under exact torch.all(expect == actual); this positive case requires no raise.')]
  if ln in (547,602):
   checks += [(PATH,x.lineno,' '.join(ast.get_source_segment(text,x).split())+' must complete without ValueError.') for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='int']
  assert checks
  body=ast.get_source_segment(text,n)
  cond=f'Pinned SGLang 0.5.14; source classification only, no execution. Exact case {PATH}:{ln}-{n.end_lineno}; source SHA256={hashlib.sha256(data).hexdigest()}; body SHA256={hashlib.sha256(body.encode()).hexdigest()}. Local helpers _assert_triples_close:52, _build_fp8_quant_pair:69, _TinyModel:90, _FakeModelRunner:106; inherited setup _WeightCheckerTestBase:362 and _ChecksumTestBase:563 use seed 0 and CUDA TinyModel, fake runner metadata. Local random_like/name cases use CPU. FP8 cases require real CUDA/DeepGEMM and share dequant reference helper. No model inference or real distributed execution is demonstrated.'
  for op,ol,expr in checks:
   assert (SOURCE/op).read_text().splitlines()[ol-1].strip()
   row=dict(snapshot='sglang',path=PATH,test_name=name,line=str(ln),unit_id=uid,property=meaning+' Separate observed predicate: '+expr,oracle_source=f'{op}:{ol}',conditions=cond,rule_id='sglang_weight_checker_explicit_source_review')
   if key(row) not in seen:new.append(row);seen.add(key(row))
  selected.add((PATH,name,str(ln)));examples.setdefault(uid,[]).append(dict(path=checks[0][0],line=checks[0][1],kind='existing_oracle'))
  audits.append(dict(path=PATH,line=ln,test_name=name,unit_id=uid,reviewed_semantics=meaning,source_body=body,oracles=[f'{p}:{l}' for p,l,_ in checks]))
 with output.open('a',newline='') as f:csv.DictWriter(f,fieldnames=COLS,delimiter='\t').writerows(new)
 pending=list(csv.DictReader((HERE/'pending.tsv').open(),delimiter='\t'));remaining=[r for r in pending if (r['path'],r['test_name'],r['line']) not in selected]
 with (HERE/'pending.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=pending[0].keys(),delimiter='\t');w.writeheader();w.writerows(remaining)
 units=json.loads((HERE/'units-ready.json').read_text());existing={u['id'] for u in units['units']};dim='weight_diagnostic_context'
 if dim not in {d['id'] for d in units['condition_dimensions']}:units['condition_dimensions'].append(dict(id=dim,type='structured',fields=['tensor_names','tensor_values','tensor_dtype','tensor_device','snapshot_state','comparison_flags','quantization_scales','parallel_rank_metadata'],observed_values=[],constraint='Diagnostic checker behavior under explicit fixtures; a matching hash is not a proof of collision freedom or end-to-end model correctness.'))
 for suffix,(name,prop,line) in UNITS.items():
  uid='diagnostic.weight_'+suffix
  if uid not in existing:units['units'].append(dict(id=uid,name=name,kind='obligation',phase='diagnostic',state_scope='diagnostic_artifact',conditions=[dim],property=prop,observable=['declared_tensor_relation','diagnostic_record'],oracle='核对逐项断言或实际比较 helper 的异常门限；按记录保留名称、标志、值、身份或元数据的具体性质。',evidence={'sglang':[dict(path=IMPL,line=line,kind='source')]+examples[uid][:2]},framework_scope=['sglang'],candidate_status='proposed; requires parent integration',semantic_domain='diagnostic_support'))
 (HERE/'units-ready.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n')
 files={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in [PATH,IMPL,'python/sglang/srt/managers/mm_utils.py','python/sglang/srt/layers/quantization/fp8_utils.py','python/sglang/test/test_utils.py']}
 (HERE/'weight-checker-review.json').write_text(json.dumps(dict(review_method='Complete test and WeightChecker implementation read; exact helper/fixture bindings reviewed. 52 explicit cases; each predicate retained separately. No CUDA tests run.',files_sha256=files,cases=audits),ensure_ascii=False,indent=2)+'\n')
 ev=json.loads((HERE/'evidence.json').read_text())
 if new:ev.setdefault('reviewed_clusters',[]).append(dict(family='weight_checker_explicit_source_review',new_mapping_rows=len(new),reviewed_entries=len(selected),files_sha256=files,audit='weight-checker-review.json',review_method='Complete source test/helper read, exact per-case manual semantic manifest; source-only, no test execution.'))
 ev['total_pending_after_review']=len(remaining);(HERE/'evidence.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n')
 print(dict(new_rows=len(new),reviewed_entries=len(selected),removed_pending=len(pending)-len(remaining),remaining=len(remaining)))
if __name__=='__main__':main()
