import json,csv,hashlib,ast,re
from pathlib import Path
P=Path(__file__).resolve().parents[3]
roots={'vllm':P.parent.parent/'experiments/third_party/reference_repos/vllm-20261105','sglang':P.parent.parent/'experiments/third_party/reference_repos/sglang-v0.5.14-complete'}
rows=list(csv.DictReader(open(P/'semantic_space/test_mapping_pilot/full-scan-index.tsv'),delimiter='\t'))
rows=[r for r in rows if r['entry_kind'] not in ['python_test_definition','python_inherited_test_candidate','rust_test_definition']]
manual={
'docker/entrypoints/test_vllm_nonroot_entrypoint.sh':('shell_test','以 shell stub 检查非 root 启动入口的环境变量、用户身份及转交参数。'),
'rust/src/chat/src/renderer/test_utils.rs':('shared_helper','聊天渲染公共样本与比较辅助过程。'),
'rust/src/chat/renderer/test_utils.rs':('shared_helper','聊天渲染公共样本与比较辅助过程。'),
 'tests/distributed/test_ca_buffer_sharing.py':('manual_execution_test','跨进程共享缓冲区写入与主机读回字节对照。'),
 'tests/distributed/test_node_count.py':('manual_execution_test','torchrun 下节点计数与 NUM_NODES 环境声明对照。'),
 'tests/distributed/test_same_node.py':('manual_execution_test','分布式同节点判定与 VLLM_TEST_SAME_HOST 声明对照。'),
 'tests/entrypoints/scale_out/ec_integration/test_scale_out_ec_e2e.py':('manual_execution_test','基线与分离执行文本对照，并核对多模态特征/传输元数据和缺失字段拒绝。'),
 'tests/spec_decode/test_custom_proposer.py':('manual_execution_test','通过 proposer 写出的标记文件确认自定义草稿类被调用；不证明草稿或最终输出等价。'),
 'tests/v1/ec_connector/integration/test_epd_correctness.py':('manual_execution_test','指定图片/提示的结果检查，以及编码/P/D 分离与基线输出逐项对照。'),
 'tests/v1/kv_connector/nixl_integration/test_disagg_accuracy.py':('manual_execution_test','Prefill/Decode 分离的各提示输出与保存的基线对应且相等。'),
 'tools/pre_commit/generate_nightly_torch_test.py':('test_configuration_generator','生成 nightly Torch 测试的依赖清单。'),
 'tools/pre_commit/test_tethering_allowlist.txt':('test_configuration','测试与改动关联检查使用的允许清单。'),
 'test/manual/kv_transfer/test_mooncake_transfer_engine.py':('external_test_launcher','启动外部 Mooncake benchmark 并检查退出状态；实际内容与性能判断在外部可执行程序中。'),
 'test/manual/lang_frontend/test_jump_forward.py':('manual_demonstration','运行 regex/JSON/角色生成并打印结果；源码没有判定输出符合格式的断言。'),
 'test/manual/layers/attention/dsa/test_get_k_scale_triton_kernel.py':('manual_execution_test','Triton 提取的 K 和 scale 张量与 PyTorch 路径精确对照，同时包含性能测量。'),
 'test/manual/layers/moe/test_moe_runners_1gpu.py':('generated_test_methods','通过 CONFIGS 和 setattr 生成 test 方法；各配置启动模型，run_eval 结果 score >= 0.48。'),
 'test/manual/layers/moe/test_moe_runners_4gpu.py':('generated_test_methods','通过 CONFIGS 和 setattr 生成多卡 test 方法；每组配置运行评测并要求 score >= 0.48。'),
 'test/manual/quant/test_cutlass_moe.py':('manual_execution_test','默认测量性能；启用 --check 才比较 Cutlass 与 Triton MoE 输出，计算差异并要求 < 1e-4。')}
helper_notes={
'python/sglang/multimodal_gen/test/server/test_server_utils.py':'图像/视频生成服务启动、性能测量及结果检查辅助过程。',
'python/sglang/multimodal_gen/test/test_utils.py':'图像/视频生成样本准备、服务器调用和 CLIP/SSIM 等比较辅助过程。',
'python/sglang/test/ascend/test_ascend_utils.py':'昇腾测试的启动、负载与性能辅助过程。',
'python/sglang/test/scripted_runtime/test_case.py':'脚本执行测试的基类及比较入口。',
'python/sglang/test/test_deepep_utils.py':'分布式专家通信和量化数据的测试辅助过程。',
'python/sglang/test/test_marlin_utils.py':'Marlin 权重量化和参考数据准备。',
'vllm/model_executor/layers/quantization/utils/marlin_utils_test.py':'Marlin 量化测试用参考权重与张量转换辅助过程。',
'python/sglang/test/test_utils.py':'模型服务启动、请求发送、评测与清理的共享测试辅助过程。',
'sgl-kernel/python/sgl_kernel/test_utils.py':'算子测试输入准备及数值比较辅助过程。',
'experimental/sgl-router/tests/component/policies/zmq_helpers.rs':'缓存事件 wire 消息与 ZMQ 发布辅助过程。'}
out=[];unresolved=[]
for r in rows:
 p=roots[r['snapshot']]/r['path'];raw=p.read_bytes();s=raw.decode(errors='replace');path=r['path']; role=note=''
 if not raw:role,note='empty_file','空文件，无测试执行或断言。'
 elif path in manual:role,note=manual[path]
 elif path in helper_notes:role,note='shared_helper',helper_notes[path]
 elif p.suffix in ['.json','.yaml','.txt']:
  role,note='fixture_or_configuration','保存输入样本、参考输出或测试配置；应由引用它的测试读取后执行检查。'
  if p.suffix=='.json': json.loads(s)
 elif p.suffix=='.rs':
  stripped=re.sub(r'/\*.*?\*/','',s,flags=re.S);stripped=re.sub(r'//[^\n]*','',stripped)
  # Only a file made solely of module exports is classified structurally.
  leftovers=re.sub(r'#\[[^\]]*\]|(?:pub(?:\([^)]*\))?\s+)?mod\s+\w+\s*;','',stripped).strip()
  leftovers=re.sub(r'pub\s+use\s+[\w:*]+\s*;', '', leftovers).strip()
  if not leftovers:role,note='module_index','声明并组织 Rust 测试模块；具体测试定义位于子模块。'
  elif any(x in path for x in ['common/','test_utils.rs']):role,note='shared_helper','构造 mock 服务、测试请求、证书或服务器环境的辅助文件；其导出的函数供其他入口调用。'
 if not role:
  role,note='file_purpose_pending','已定位文件，需要继续检查其定义与执行方式。';unresolved.append(path)
 out.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'role':role,'purpose':note,'source_sha256':hashlib.sha256(raw).hexdigest(),'source_lines':len(s.splitlines()),'counted_as_completed_test_mapping':False})
D=P/'semantic_space/test_mapping_pilot/round9_work'
(D/'file-record-roles.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
from collections import Counter
print(dict(Counter(x['role'] for x in out)));print('unresolved',unresolved)
