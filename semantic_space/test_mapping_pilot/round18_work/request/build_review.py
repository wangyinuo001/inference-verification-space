"""Reviewed multimodal configuration and embedding decoding contracts."""
import ast,csv,json,hashlib,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;P=OUT.parents[1];sys.path.insert(0,str(P))
from build_round9_crosswalk import ROOTS,FIELDS,key,read
src=ROOTS['vllm'];known={key(r) for r in read(P/'full_delivery/all-entries.tsv') if r['review_status']=='property_reviewed'}
files=['tests/config/test_multimodal_config.py','tests/renderers/test_sparse_tensor_validation.py']
rows=[];entries=[];helpers=[]
for path in files:
 text=(src/path).read_text();tree=ast.parse(text)
 def visit(nodes,pre=''):
  for n in nodes:
   if isinstance(n,ast.ClassDef):visit(n.body,pre+n.name+'::')
   elif isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
    if not n.name.startswith('test'):
     helpers.append({'path':path,'name':pre+n.name,'line':n.lineno,'source':ast.get_source_segment(text,n)});continue
    r={'snapshot':'vllm','path':path,'test_name':pre+n.name,'line':str(n.lineno)}
    if key(r) in known:continue
    if path==files[1] and n.lineno in [458,465]:continue # arithmetic constant/fixture size alone, no target behavior
    checks=[a for a in ast.walk(n) if isinstance(a,ast.Assert) or isinstance(a,ast.Call) and isinstance(a.func,ast.Attribute) and (a.func.attr.startswith('assert_') or a.func.attr=='raises')]
    if path==files[0]:
     uid='deployment.effective_configuration'
     if n.lineno in [48,53,92,487,501,517,522,672]:uid='deployment.configuration_admissibility'
     if n.lineno in [58,66,74,97]:uid='deployment.computation_config_identity'
     cond='直接构造配置与解析输入；HF模型配置可能需下载，未运行模型。设备、可选库及VllmConfig/ModelConfig在角色解析用例中由明确替身控制；只检查最终配置/异常，不证明GPU执行、内存隔离或性能。嵌套kwargs检查保留覆盖优先级及输入不被改写条件。'
     if n.lineno in [517,522]:
      checks=[a for a in ast.walk(n) if isinstance(a,ast.Call) and isinstance(a.func,ast.Name) and a.func.id=='_validate_mm_processor_device']
      cond+='该入口仅以校验函数正常返回作为接受条件，没有额外结果断言。'
    else:
     uid='input.embedding_decode_contract'
     cond='构造稠密/稀疏张量并以torch.save或numpy编码，经真实读取函数检查类型、形状、值或异常。prompt配置使用OPT-125M的float32/768维；不运行模型或HTTP服务。每项仅确认实际断言，名字中的集成/安全不扩大验证结论。'
     if n.lineno>=448:cond+='大小限制环境为4096字节，实际超限测试张量为1024×1024；4TiB常量只用于未纳入本类的算术/夹具检查，不分配该大小。'
    assert checks,(path,n.name)
    prop='本项检查：'+'；'.join(ast.unparse(a) for a in checks)
    rows.append({**r,'unit_id':uid,'property':prop,'oracle_source':';'.join(f'{path}:{a.lineno}' for a in checks),'conditions':cond,'rule_id':'reviewed_multimodal_config_and_embedding_decode_20261120'})
    entries.append({**r,'source':ast.get_source_segment(text,n),'decorators':[ast.unparse(a) for a in n.decorator_list],'assertions':[{'line':a.lineno,'expression':ast.unparse(a)} for a in checks],'conditions':cond})
 visit(tree.body)
products=['vllm/config/multimodal.py','vllm/config/vllm.py','vllm/config/model.py','vllm/utils/sparse_utils.py','vllm/renderers/embed_utils.py','vllm/multimodal/media/image.py','vllm/multimodal/media/audio.py','vllm/multimodal/media/video.py']
with (OUT/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,FIELDS,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
(OUT/'evidence.json').write_text(json.dumps({'file_reviews':[{'snapshot':'vllm','path':f,'source_sha256':hashlib.sha256((src/f).read_bytes()).hexdigest()} for f in files+products],'entries':entries,'shared_helpers':helpers,'source_review':'逐项阅读两文件测试、共享配置构造和张量编码helper；核对配置合并/设备解析/哈希以及生产张量读取、稀疏索引和解码大小边界。'},ensure_ascii=False,indent=2)+'\n')
print(len(rows))
