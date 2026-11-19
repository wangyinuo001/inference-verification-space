from pathlib import Path
exec(compile(Path(__file__).with_name('review_specs.py').read_text(),str(Path(__file__).with_name('review_specs.py')),'exec'))
(OUT/'proposed-units.json').write_text(json.dumps(PROPOSED,ensure_ascii=False,indent=2))
def external_body(anchor):
 path,line=anchor.rsplit(':',1);line=int(line);txt=(SRC/path).read_text();tree=ast.parse(txt)
 options=[n for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.lineno<=line<=n.end_lineno]
 if not options:return txt.splitlines()[line-1]
 return ast.get_source_segment(txt,min(options,key=lambda n:n.end_lineno-n.lineno))
def defs(tree):
 out={}
 for x in tree.body:
  if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)):out[x.name]=x
  if isinstance(x,ast.ClassDef):
   for y in x.body:
    if isinstance(y,ast.FunctionDef):out[x.name+'::'+y.name]=y
 return out
idx=list(csv.DictReader((P/'full_delivery/all-entries.tsv').open(),delimiter='\t'))
skip=set()
for rd in ['round12_work','round13_work','round14_work','round15_work']:
 for f in (P/rd).rglob('crosswalk.tsv'):
  if f.parent==OUT:continue
  skip.update((r['snapshot'],r['path'],r['test_name'],r['line']) for r in csv.DictReader(f.open(),delimiter='\t'))
rows=[];ev=[];reviews=[]
for path in sorted({p for p,n in S}):
 txt=(SRC/path).read_text();tree=ast.parse(txt);dd=defs(tree)
 reviews.append({'snapshot':'vllm','path':path,'source_sha256':hashlib.sha256((SRC/path).read_bytes()).hexdigest(),'review_scope':'仅本批显式列出的测试入口及其调用helper，非整文件无条件确认'})
 for r in idx:
  key=(r['snapshot'],r['path'],r['test_name'],r['line'])
  if r['snapshot']!='vllm' or r['path']!=path or r['review_status']=='property_reviewed' or key in skip or (path,r['test_name']) not in S:continue
  fn=dd[r['test_name']];unit,desc,impl=S[path,r['test_name']]
  helpers=[];seen=set()
  def scan(f):
   for a in ast.walk(f):
    if isinstance(a,ast.Name) and a.id in dd and a.id not in seen and not a.id.startswith('test_'):
     seen.add(a.id);helpers.append(dd[a.id]);scan(dd[a.id])
    if isinstance(a,ast.Attribute) and isinstance(a.value,ast.Name) and a.value.id=='self' and '::' in r['test_name']:
     member=r['test_name'].rsplit('::',1)[0]+'::'+a.attr
     if member in dd and member not in seen:
      seen.add(member);helpers.append(dd[member]);scan(dd[member])
  scan(fn)
  for node in ast.walk(fn):
   if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='self' and '::' in r['test_name']:
    member=r['test_name'].rsplit('::',1)[0]+'::'+node.attr
    if member in dd and member not in seen:
     seen.add(member);helpers.append(dd[member]);scan(dd[member])
  # Include actual fixture setup and supporting fake/recording classes.
  def is_fixture(f):
   return any('fixture' in ast.unparse(d) for d in f.decorator_list)
  fixtures=[f for f in dd.values() if is_fixture(f)]
  needed={a.arg for a in fn.args.args+fn.args.kwonlyargs}
  for decorator in fn.decorator_list:
   if isinstance(decorator,ast.Call) and ast.unparse(decorator.func).endswith('usefixtures'):
    needed.update(a.value for a in decorator.args if isinstance(a,ast.Constant) and isinstance(a.value,str))
  for f in fixtures:
   if any('autouse=True' in ast.unparse(d) for d in f.decorator_list):needed.add(f.name)
  added=True
  while added:
   added=False
   for f in fixtures:
    if f.name in needed and f.name not in seen:
     seen.add(f.name);helpers.append(f);scan(f)
     needed.update(a.arg for a in f.args.args+f.args.kwonlyargs);added=True
  owner=next((c for c in tree.body if isinstance(c,ast.ClassDef) and r['test_name'].startswith(c.name+'::')),None)
  if owner is not None:
   for setup in owner.body:
    if isinstance(setup,ast.FunctionDef) and setup.name in ['setup_method','teardown_method','setup_class','teardown_class']:
     helpers.append(setup);scan(setup)
  checks=[]
  for f in [fn]+helpers:
   for a in ast.walk(f):
    if isinstance(a,ast.Assert) or (isinstance(a,ast.Call) and (ast.unparse(a.func).split('.')[-1].startswith('assert_') or ast.unparse(a.func).endswith('.raises'))):checks.append({'source':f'{path}:{a.lineno}','expression':ast.unparse(a)})
  assert checks
  cond={'test_range':f'{path}:{fn.lineno}-{fn.end_lineno}','module_conditions':[ast.unparse(x) for x in tree.body if isinstance(x,(ast.Assign,ast.AnnAssign))],'decorators':[ast.unparse(x) for x in fn.decorator_list],'class_conditions':[] if owner is None else [ast.unparse(x) for x in owner.body if isinstance(x,(ast.Assign,ast.AnnAssign))],'class_decorators':[] if owner is None else [ast.unparse(x) for x in owner.decorator_list],'module_guards':[ast.unparse(x) for x in tree.body if isinstance(x,ast.If)],'supporting_classes':[{'source':f'{path}:{c.lineno}','body':ast.get_source_segment(txt,c)} for c in tree.body if isinstance(c,ast.ClassDef) and not c.name.startswith('Test')],'input_and_call_source':ast.get_source_segment(txt,fn),'helpers':[{'source':f'{path}:{h.lineno}','body':ast.get_source_segment(txt,h)} for h in helpers],'implementation_anchors':impl,'external_helper_anchors':[a for a in impl if a.startswith('tests/')],'external_helper_sources':[{'source':a,'body':external_body(a)} for a in impl if a.startswith('tests/')],'validation':'仅静态源码核对，未导入或执行上游，未使用GPU'}
  row={k:r[k] for k in ['snapshot','path','test_name','line']};row.update(unit_id=unit,property=desc+' 实际断言：'+'；'.join(x['expression'] for x in checks),oracle_source=';'.join(dict.fromkeys(x['source'] for x in checks)),conditions=json.dumps(cond,ensure_ascii=False),rule_id='round15_compute_source_'+Path(path).stem)
  rows.append(row);ev.append({**{k:row[k] for k in ['snapshot','path','test_name','line','unit_id']},'review':desc,'assertions':checks,'conditions':cond})
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id'],delimiter='\t');w.writeheader();w.writerows(rows)
impl_paths=sorted({x.rsplit(':',1)[0] for _,_,im in S.values() for x in im})
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':reviews,'entry_reviews':ev,'supporting_source_reviews':[{'snapshot':'vllm','path':p,'source_sha256':hashlib.sha256((SRC/p).read_bytes()).hexdigest()} for p in impl_paths],'confirmed_rows':len(rows),'deferred':DEFERRED},ensure_ascii=False,indent=2))
(OUT/'notes.md').write_text(f'确认{len(rows)}条。逐项阅读输入、调用和断言。每条保留完整源码参数及helper，数值参考、同核变形、分派、mock边界分开描述。未执行上游或GPU。新增性质详见proposed-units.json；其余使用现有性质。\n')
print('confirmed',len(rows))
