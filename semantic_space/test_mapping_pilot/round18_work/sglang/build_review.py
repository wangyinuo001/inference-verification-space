# -*- coding: utf-8 -*-
import ast,csv,json,hashlib,re,collections
from pathlib import Path
O=Path(__file__).parent;R=O.parents[3];P=R.parents[1];B=P/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'
T='sgl-model-gateway/bindings/python/tests/'
D='sgl-model-gateway/bindings/python/src/'
FILES=[T+x for x in ['test_validation.py','test_router_config.py','test_startup_sequence.py','test_pyo3_binding.py']]
PROD=[D+'sglang_router/'+x for x in ['router.py','router_args.py','launch_router.py','launch_server.py']]+[D+'lib.rs']
cache={p:(B/p).read_text()for p in FILES+PROD};trees={p:ast.parse(cache[p])for p in FILES};defs={p:{n.lineno:n for n in ast.walk(t)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}for p,t in trees.items()}
def seg(p,n):return '\n'.join(cache[p].splitlines()[n.lineno-1:n.end_lineno])
def checks(p,n,guards=(),root=True):
 if not root and isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):return []
 out=[]
 if isinstance(n,ast.If):
  for c in n.body:out+=checks(p,c,guards+(ast.unparse(n.test),),False)
  for c in n.orelse:out+=checks(p,c,guards+('not ('+ast.unparse(n.test)+')',),False)
  return out
 if isinstance(n,ast.Assert)or isinstance(n,ast.Call)and (ast.unparse(n.func).split('.')[-1].startswith('assert')or ast.unparse(n.func)in ['pytest.raises','pytest.fail','self.fail']):out.append({'path':p,'line':n.lineno,'expression':ast.unparse(n),'guards':list(guards)})
 for c in ast.iter_child_nodes(n):out+=checks(p,c,guards,False)
 return out
def anchor(path,line):return {'path':path,'line':line,'kind':'source'}
units=[{'id':'deployment.configuration_value_contract','name':'部署配置值的表示与约束','property':'配置值按接口声明转换为对应字段或枚举，并对不合法的取值、结构或组合给出约定异常。具体规则与验证所在边界分别保留。','observable':['effective_value','converted_enum','exception_type','exception_message'],'oracle':'逐项检查字段、枚举及异常；不由字段读取推断服务运行行为。','evidence':{'sglang':[anchor(PROD[0],22),anchor(PROD[0],39),anchor(PROD[1],1110),anchor(PROD[4],243)]}}, {'id':'deployment.service_constructor_contract','name':'服务构造与启动调用约定','property':'部署入口将生效参数传入服务构造器，并按启动流程调用实例；构造或启动异常按接口约定返回或传播；清理阶段按进程存活状态发出终止或强制结束调用。','observable':['constructor_arguments','constructed_type','start_call','propagated_error','diagnostic_message','cleanup_signal_calls'],'oracle':'区分真实构造器与测试替身，分别检查参数、构造结果类型、调用次数或异常，不由构造成功推断服务已可用。','evidence':{'sglang':[anchor(PROD[2],21),anchor(PROD[0],230),anchor(PROD[0],336),anchor(PROD[4],766),anchor(PROD[3],86),anchor(PROD[3],129)]}}]
for u in units:u.update(kind='obligation',phase='preflight',state_scope='deployment',conditions=['deployment_configuration','configuration_resolution_profile'],framework_scope=['sglang'])
rows=[];ev=[];support=[]
for r in csv.DictReader((R/'semantic_space/test_mapping_pilot/full_delivery/all-entries.tsv').open(),delimiter='\t'):
 if r['snapshot']!='sglang'or r['definition_path']not in FILES or r['review_status']in ['property_reviewed','supporting_record_reviewed','outside_current_scope']:continue
 p=r['definition_path'];n=defs[p][int(r['definition_line'])];name=n.name;body=seg(p,n);owner=next((c.name for c in trees[p].body if isinstance(c,ast.ClassDef)and n in c.body),'');aa=checks(p,n)
 base={k:r[k]for k in ['snapshot','path','test_name','line','definition_path','definition_line','entry_kind']}
 note='按字段保留、转换和显式检查解释；不按测试名字中的validation推断范围检查。'
 reason=None
 if owner=='TestSetupLogger':reason='测试文件内部复制的setup_logger，未调用产品实现；属于测试自带辅助逻辑。'
 elif 'pytest.skip(' in body and not aa:reason='函数无条件pytest.skip，无实际验证断言。'
 elif name=='test_validation_error_handling':reason='嵌套在test_launch_server_process_and_cleanup中，未调用且不是独立pytest收集方法；预期异常与当前允许空URL实现不符。'
 if reason:
  support.append({**base,'reason':reason,'test_source':body,'source_sha256':hashlib.sha256((B/p).read_bytes()).hexdigest()});continue
 u='deployment.effective_configuration';anchors=[anchor(PROD[1],38)]
 if p==FILES[2]:
  u='deployment.service_constructor_contract';anchors=[anchor(PROD[2],21),anchor(PROD[1],1021)]
  note='Router.from_args/start为MagicMock，断言仅覆盖入口调用、捕获参数或错误传播；不表示实际服务启动。'
  if owner=='TestPolicyFromStr':u='deployment.configuration_value_contract';anchors=[anchor(PROD[0],22)];note='字符串到枚举及未知枚举KeyError。'
  if name=='test_router_defaults_and_start':anchors=[anchor(PROD[0],230),anchor(PROD[0],336)];note='Rust _Router被FakeRouter替换，检查转换参数和start转发。'
  if name=='test_find_available_ports_and_wait_health':u='service.lifecycle_health';anchors=[anchor(PROD[3],100),anchor(PROD[3],116)];note='可用端口/随机步长/HTTP与时间均替身；先请求异常再200，检查端口列表与健康等待成功。'
  if name=='test_launch_server_process_and_cleanup':u='deployment.service_constructor_contract';anchors=[anchor(PROD[3],86),anchor(PROD[3],129)];note='假进程检查port、base_gpu_id=dp_id*tp_size、dp_size=1及SIGTERM/SIGKILL调用，不检查真实进程退出。'
 elif p==FILES[1]:
  if 'launch_router(args)'in body:u='deployment.service_constructor_contract';anchors=[anchor(PROD[2],21)];note='只检查mock from_args调用一次，不检查策略实际执行。'
  if name in ['test_policy_enum_conversion','test_invalid_policy_enum_conversion']:u='deployment.configuration_value_contract';anchors=[anchor(PROD[0],22)]
  if name=='test_config_serialization':note='仅hasattr，未进行序列化/反序列化。'
  if name=='test_config_immutability':note='断言实际确认可修改，不是不可变性。'
 elif p==FILES[0]:
  if owner=='TestLaunchValidation':u='deployment.service_constructor_contract';anchors=[anchor(PROD[2],21)];note='mock from_args被调用一次；空URL配置允许进入构造，不断言远端连接。'
  if name in ['test_invalid_ports','test_malformed_urls']:note='断言原样保留非法值，当前dataclass不进行该项范围/格式校验；不归为拒绝性质。'
 elif p==FILES[3]:
  anchors=[anchor(PROD[0],230),anchor(PROD[1],959)]
  if owner=='TestEnumConversions':u='deployment.configuration_value_contract';anchors=[anchor(PROD[0],22),anchor(PROD[0],39),anchor(PROD[0],54),anchor(PROD[0],76)]
  if owner.startswith('TestPy'):
   anchors=[anchor(PROD[4],{'TestPyOracleConfig':243,'TestPyPostgresConfig':337,'TestPyRedisConfig':304,'TestPyApiKeyEntry':72,'TestPyJwtConfig':116,'TestPyControlPlaneAuthConfig':175}[owner])]
   if 'pytest.raises' in body:u='deployment.configuration_value_contract'
  if owner=='TestBuildControlPlaneAuthConfig':anchors=[anchor(PROD[0],83)];note='检查是否创建认证配置以及具体嵌套字段和warning；不验证API权限认证结果。'
  if owner in ['TestParseControlPlaneApiKeys','TestParseJwtRoleMapping']:u='deployment.configuration_value_contract';anchors=[anchor(PROD[1],1110 if owner=='TestParseControlPlaneApiKeys'else 1134)]
  if owner=='TestRouterFromArgsKitchenSink'or name=='test_jwt_role_claim_end_to_end':u='deployment.service_constructor_contract';anchors=[anchor(PROD[0],230),anchor(PROD[4],766)];note='真实PyO3构造器，未调用start或远端worker；类型检查不等于所有参数值已比对。kitchen_sink只有HTTP连接池和WASM五个字段另有显式值断言。'
 if not aa:
  assert name in ['test_all_policies_construct','test_redis_persistent_retention'],(p,name)
  aa=[{'path':p,'line':x.lineno,'expression':'完成构造且无异常: '+ast.unparse(x),'guards':[]}for x in ast.walk(n)if isinstance(x,ast.Call)and ast.unparse(x.func)=='Router.from_args']
  note+=' 本项以调用无异常为判定，不断言保留值。'
 rr={k:r[k]for k in ['snapshot','path','test_name','line']};rr.update(unit_id=u,property='本项实际检查：'+'；'.join(' '.join(a['expression'].split())for a in aa),oracle_source='; '.join(f'{p}:{a["line"]}'for a in aa),conditions='输入值、配置、替身与装饰器见逐项evidence。'+note,rule_id='round18_sglang_gateway_configuration_source_review');rows.append(rr)
 ev.append({**base,'unit_id':u,'test_source':body,'test_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'assertions':aa,'production_anchors':anchors,'notes':note,'decorators':[ast.unparse(x)for x in n.decorator_list],'local_helpers':[{'line':x.lineno,'source':seg(p,x)}for x in trees[p].body if isinstance(x,ast.FunctionDef)and not x.name.startswith('test')],'imports':[seg(p,x)for x in trees[p].body if isinstance(x,(ast.Import,ast.ImportFrom))]})
fields=['snapshot','path','test_name','line','unit_id','property','oracle_source','conditions','rule_id']
with (O/'crosswalk.tsv').open('w')as f:w=csv.DictWriter(f,fields,delimiter='\t');w.writeheader();w.writerows(rows)
(O/'proposed-units.json').write_text(json.dumps(units,ensure_ascii=False,indent=2));(O/'supporting-records.json').write_text(json.dumps(support,ensure_ascii=False,indent=2))
(O/'evidence.json').write_text(json.dumps({'file_reviews':[{'snapshot':'sglang','path':p,'source_sha256':hashlib.sha256((B/p).read_bytes()).hexdigest()}for p in FILES+PROD],'entries':ev,'supporting_records':support},ensure_ascii=False,indent=2))
known={u['id']for u in json.load((R/'semantic_space/space_seed.json').open())['units']}|{u['id']for u in units}
assert len(rows)==len({tuple(r[k]for k in ['snapshot','path','test_name','line'])for r in rows})
for r in rows:
 assert r['unit_id']in known
 for loc in r['oracle_source'].split('; '):
  assert re.fullmatch(r'.+:\d+',loc);p,l=loc.rsplit(':',1);assert 0<int(l)<=len(cache[p].splitlines())
for e in ev:
 for a in e['production_anchors']:assert 0<a['line']<=len(cache[a['path']].splitlines())
v={'status':'passed','count':len(rows),'supporting_records':len(support),'file_hashes':len(FILES+PROD),'new_units':len(units),'unit_counts':dict(collections.Counter(r['unit_id']for r in rows)),'upstream_tests_run':False}
(O/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));(O/'notes.md').write_text('本批核查4个gateway Python配置/启动/绑定测试文件。读取产品dataclass/CLI、字符串转换、认证配置转换、启动入口、PyO3构造器及进程辅助函数。每条保留实际断言和替身边界。名称validation/immutability/serialization不能替代实际断言。4个本地复制logger测试、3个skip、嵌套未调用测试单列。Rust对象构造未启动服务；mock构造不证明Rust参数或路由策略运行。\n');print(v)
