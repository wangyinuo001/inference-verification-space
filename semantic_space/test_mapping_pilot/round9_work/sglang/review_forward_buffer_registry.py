# -*- coding: utf-8 -*-
"""39 explicit source-reviewed registry cases; no runtime execution."""
import ast,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parents[5]/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
PATH='test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py'
IMPL='python/sglang/srt/model_executor/cuda_graph_buffer_registry.py'
RULE='sglang_forward_buffer_explicit_source_review'
COLS='snapshot path test_name line unit_id property oracle_source conditions rule_id'.split()
CASES={
70:('registration','GraphSlot axis garbage raises ValueError.'),
79:('registration','slice_for before buffer allocation raises RuntimeError.'),
91:('registration','Token-axis input_ids slot allocates sixteen int64 zeros.'),
105:('registration','FILL_SENTINEL seq_lens allocation contains eight int32 sevens.'),
121:('registration','Registering duplicate name dup raises ValueError.'),
141:('registration','Disabled slot has no buffer and is excluded from has_slot and slot_names.'),
156:('registration','Explicit CPU slot override produces CPU buffer initialized with sentinel eleven.'),
236:('fill','No-padding fill copies six declared tensor fields into their matching registry buffers.'),
265:('fill','Raw two requests/four tokens copy head; zero-policy tails are zero, seq_lens tails are sentinel five, and host seq_lens stays on CPU.'),
325:('fill','KEEP_PAD copies raw position head [0,1] while six padded elements retain poison 77.'),
363:('view','Extract produces new padded batch with slot-backed input_ids pointer/length, request-index length, retained forward_mode/spec_info and unchanged template batch size.'),
407:('fill','Missing encoder_lens source skips copy and sentinel-zero buffer remains eight zeros.'),
440:('view','Present input_ids becomes slot-backed view; absent plain mrope_positions stays None.'),
471:('view','Computed global_num_tokens slot is exposed and shares slot pointer despite original batch field None; hook value two is not directly asserted.'),
501:('computed_fields','post_fill multiplies copied head [1,2,3,4] into [10,20,30,40] and observes raw/padded token counts 4/8.'),
543:('fill','Custom second-axis slice copies mrope 3x4 head and extract returns shape 3x8.'),
585:('fill','Nested ngram column-starts source copies its own three-element head [1,2,3], retaining remaining poison 99.'),
616:('fill','Nested source_fn returning None leaves entire buffer at seven.'),
638:('fill','Pipeline side-input tensor from FillContext copies four head values [5,6,7,8].'),
672:('view','Dotted structured slot is skipped by extraction, allowing new batch_size eight without invalid dotted dataclass argument.'),
719:('storage_sharing','share_pool=False registries allocate distinct addresses for otherwise identical ids slots.'),
728:('storage_sharing','share_pool=True identical sixteen-token ids slots share one data_ptr.'),
739:('storage_sharing','Different token sizes sixteen/thirty-two preserve shapes and use different addresses.'),
753:('storage_sharing','After resetting shared pool, equal sizes alias and different sizes remain distinct in either registration order.'),
789:('fill','Decode factory registers declared core slots but not mamba; input head copied and tail retained, cache/request tails zero, sequence tails sentinel five, host dtype/device correct, mrope head copied and extracted dimensions padded.'),
874:('storage_sharing','Decode registry adopts source input_ids, seq_lens, host seq_lens and mrope storage by pointer identity.'),
906:('computed_fields','With attention TP size two/rank zero, gathered local nonpadded count clamps global100 into padded8/TP2 local count four.'),
949:('view','Disabling computed global token registration carries original batch tensors 37 and 41 by identity; default factory registers both fields.'),
1004:('fill','Decode nested ngram slots adopt column storage and copy explicit column starts [7,8,9] and request lengths [1,1,2] into heads.'),
1053:('fill','Decode pipeline hidden-state slot adopts source pointer; side input copies three rows of ones and leaves remaining zero rows.'),
1100:('fill','Decode canary fields register and copy request-id/bootstrap heads; bootstrap tail retains original -1.'),
1159:('fill','Prefill core slots adopt source storage; input_ids head becomes [1,2,3], in-bucket tail3:8 is zero, beyond-bucket tail retains poison seven.'),
1195:('fill','Multimodal prefill registers mrope and embeds; reset-only embeds ignore input nines, keep head fives and zero padded tail; mrope copies ones into head and zeroes second-axis tail.'),
1236:('fill','Enabled prefill mamba index slot copies two batch-axis indices [3,4] despite token-axis counts three/eight.'),
1267:('registration','Source-free prefill allocates expected core/mrope/embed/mamba shapes; subsequent fill copies [1,2,3] head and zeroes padded tail.'),
1304:('view','Multimodal prefill without input_embeds registration retains mrope slot and carries the original input embedding object into extracted batch.'),
1345:('fill','FILL_ONCE initializes four nines; later raw head copies [1,2] while manually poisoned padded tail remains 99.'),
1375:('computed_fields','Non-gathered num_token_non_padded slot preserves copied batch value seven.'),
1403:('computed_fields','Gathered global_num_tokens ignores stale batch value999 and writes padded token count four.'),
}
UNITS={
'registration':('前向缓冲区注册与配置有效性','前向缓冲区按名称、轴、形状、类型和启用状态建立；非法轴、重复名称或未分配的读取被拒绝，分配值遵守初始化策略。',['slot_names','slot_enabled','buffer_shape','buffer_dtype','buffer_device','initial_buffer_values','raised_exception'],'逐项核对注册后的名字集合、启用状态、shape/dtype/device 与初始化值；指定非法轴、重复名和分配前访问触发相应异常。',['GraphSlot.__post_init__','GraphSlot.slice_for','CudaGraphBufferRegistry.register_slot']),
'fill':('前向输入复制与补齐区域关系','前向缓冲区活动区域绑定本次批次或侧输入；补齐区域按零、哨兵、保留或只初始化策略处理，桶外存储保持声明状态。',['active_region_values','padded_region_values','beyond_bucket_values','source_tensor_values','batch_axis','token_axis','registered_slot_names'],'比较显式源张量与目标活动区；用预先污染的尾部区分 ZERO、FILL_SENTINEL、KEEP_PAD/FILL_ONCE，检查桶外值及二维切片；缺失源不复制，嵌套/侧输入按源长度绑定。',['CudaGraphBufferRegistry.fill_from','GraphSlot.reset_padding','build_decode_registry','build_prefill_registry']),
'view':('前向批次视图绑定与字段保留','抽取的前向批次使用指定补齐尺寸的缓冲区视图，同时保留未注册或本次缺失字段；计算字段按策略暴露，原模板保持不变。',['extracted_batch_size','extracted_shapes','buffer_data_ptr','carried_field_identity','optional_field_presence','template_state'],'核对抽取对象与槽位 data_ptr、补齐长度及形状；非槽字段和禁用计算字段保持原对象或值，缺失普通字段保持 None，计算槽即使模板 None 仍暴露；模板身份/尺寸不变。',['CudaGraphBufferRegistry.extract_buffer','GraphSlot.slice_for']),
'storage_sharing':('前向缓冲区存储共享符合键与来源','前向缓冲区按声明的共享键复用同一存储或保持独立；采用已有来源时绑定其存储，注册顺序不改变同键或异键的共享关系。',['slot_name','buffer_numel','buffer_dtype','buffer_device','buffer_shape','data_ptr','source_data_ptr','registration_order'],'用 data_ptr 比较关闭共享、同键共享、不同大小隔离及前后注册顺序；来源采用场景必须与指定源张量指针一致。',['CudaGraphBufferRegistry.register_slot']),
'computed_fields':('前向缓冲区派生字段与复制顺序','前向输入复制完成后执行派生字段更新；结果对应本次原始/补齐计数、参与分片和复制开关，而非陈旧输入值。',['post_fill_output_values','raw_num_tokens','padded_num_tokens','attn_tp_rank','attn_tp_size','local_nonpadded_count','global_token_count'],'用复制后乘十验证 hook 顺序；核对 hook 上下文4/8、未聚合复制值7、聚合全局计数4，以及TP2/rank0把全局100夹紧至本地4的结果。',['CudaGraphBufferRegistry.fill_from','build_decode_registry']),
}
def main():
 ev=json.loads((HERE/'evidence.json').read_text())
 if any(c.get('family')==RULE for c in ev.get('reviewed_clusters',[])):print('Already recorded.');return
 data=(SOURCE/PATH).read_bytes();text=data.decode();tree=ast.parse(text)
 nodes={n.lineno:(c.name,n) for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
 impltree=ast.parse((SOURCE/IMPL).read_text());implnodes={}
 for n in impltree.body:
  if isinstance(n,ast.FunctionDef):implnodes[n.name]=n.lineno
  if isinstance(n,ast.ClassDef):
   for f in n.body:
    if isinstance(f,ast.FunctionDef):implnodes[n.name+'.'+f.name]=f.lineno
 rows=[];audit=[];examples={};selected=set()
 for line,(suffix,meaning) in CASES.items():
  cls,n=nodes[line];name=cls+'::'+n.name;uid='model.forward_buffer_'+suffix
  checks=sorted([x for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr.startswith('assert')],key=lambda x:x.lineno);assert checks
  body=ast.get_source_segment(text,n)
  row=dict(snapshot='sglang',path=PATH,test_name=name,line=str(line),unit_id=uid,property=meaning+' Separately checked predicates: '+' | '.join(' '.join(ast.get_source_segment(text,x).split()) for x in checks),oracle_source=';'.join(f'{PATH}:{x.lineno}' for x in checks),conditions=f'Pinned SGLang0.5.14 source-only review. Explicit CPU tensors, _MiniForwardBatch dataclass and SimpleNamespace stand-ins; _make_registry uses CPU and stated max sizes. Actual case and class fixtures {PATH}:{line}-{n.end_lineno}; file SHA256={hashlib.sha256(data).hexdigest()}; body SHA256={hashlib.sha256(body.encode()).hexdigest()}. Pool tests clear process-wide pool; gathered count case overrides attn_tp_size=2/rank=0 without communication. CPU fill uses copy_, so GPU grouped-foreach-copy and CUDA capture/replay are not exercised by these definitions.',rule_id=RULE)
  rows.append(row);selected.add((PATH,name,str(line)));examples.setdefault(uid,[]).append(dict(path=PATH,line=checks[0].lineno,kind='existing_oracle'))
  audit.append(dict(path=PATH,line=line,test_name=name,unit_id=uid,reviewed_semantics=meaning,source_body=body,oracle_lines=[x.lineno for x in checks]))
 with (HERE/'proposed-crosswalk.tsv').open('a',newline='') as f:csv.DictWriter(f,fieldnames=COLS,delimiter='\t').writerows(rows)
 pending=list(csv.DictReader((HERE/'pending.tsv').open(),delimiter='\t'));remaining=[r for r in pending if (r['path'],r['test_name'],r['line']) not in selected]
 with (HERE/'pending.tsv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=pending[0].keys(),delimiter='\t');w.writeheader();w.writerows(remaining)
 units=json.loads((HERE/'units-ready.json').read_text());dim='forward_buffer_context'
 if dim not in {d['id'] for d in units['condition_dimensions']}:units['condition_dimensions'].append(dict(id=dim,type='structured',fields=['slot_specification','raw_batch_size','padded_batch_size','raw_token_count','padded_token_count','padding_policy','source_presence','source_values','share_pool','parallel_rank_metadata'],observed_values=[],constraint='Buffer registration, copying, padding and alias relations are checked under concrete shapes and CPU fixtures; graph replay and cross-stream ordering require separate evidence.'))
 known={u['id'] for u in units['units']}
 for suffix,(name,prop,obs,oracle,methods) in UNITS.items():
  uid='model.forward_buffer_'+suffix
  if uid in known:continue
  evidence=[dict(path=IMPL,line=implnodes[m],kind='source') for m in methods]+examples[uid][:2]
  if suffix=='storage_sharing':evidence.append(dict(path='python/sglang/srt/model_executor/input_buffers.py',line=16,kind='source'))
  if suffix=='computed_fields':evidence.append(dict(path='python/sglang/srt/model_executor/forward_batch_info.py',line=200,kind='source'))
  for e in evidence:assert (SOURCE/e['path']).read_text().splitlines()[e['line']-1].strip()
  units['units'].append(dict(id=uid,name=name,kind='obligation',phase='forward',state_scope='process',conditions=[dim],property=prop,observable=obs,oracle=oracle,evidence={'sglang':evidence},framework_scope=['sglang'],candidate_status='proposed; requires parent integration',semantic_domain='inference_request'))
 (HERE/'units-ready.json').write_text(json.dumps(units,ensure_ascii=False,indent=2)+'\n')
 paths=[PATH,IMPL,'python/sglang/srt/model_executor/input_buffers.py','python/sglang/srt/model_executor/forward_batch_info.py'];hashes={p:hashlib.sha256((SOURCE/p).read_bytes()).hexdigest() for p in paths}
 for r in rows:
  for ref in r['oracle_source'].split(';'):
   p,l=ref.rsplit(':',1);assert (SOURCE/p).read_text().splitlines()[int(l)-1].strip()
 assert len(rows)==len({(r['path'],r['test_name'],r['line']) for r in rows})==39
 (HERE/'forward-buffer-review.json').write_text(json.dumps(dict(review_method='Complete test file and called implementation/helpers reviewed; explicit per-case manifest. CPU fixture bypasses grouped GPU foreach-copy; no runtime run.',files_sha256=hashes,unique_entry_keys=True,oracle_lines_nonempty=True,cases=audit),ensure_ascii=False,indent=2)+'\n')
 ev.setdefault('reviewed_clusters',[]).append(dict(family=RULE,reviewed_entries=len(rows),mapped_proposed=len(rows),predicate_records=sum(len(a['oracle_lines']) for a in audit),files_sha256=hashes,audit='forward-buffer-review.json'))
 ev['total_pending_after_review']=len(remaining);(HERE/'evidence.json').write_text(json.dumps(ev,ensure_ascii=False,indent=2)+'\n')
 print(dict(reviewed=len(rows),removed_pending=len(pending)-len(remaining),remaining=len(remaining)))
if __name__=='__main__':main()
