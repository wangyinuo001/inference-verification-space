import json,csv,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[4]
D=P/'semantic_space/test_mapping_pilot/round9_work/rust'
rows=json.loads((D/'functions.json').read_text()); assigned={};families={}
def group(indices,uid,title,contract,decision='property_identified',note=''):
    if isinstance(indices,str):
        expanded=[]
        for s in indices.split(','):
            a=s.split('-'); expanded.extend(range(int(a[0]),int(a[-1])+1))
        indices=expanded
    families[uid]={'id':uid,'name':title,'property':contract}
    for i in indices: assigned[i]={'family':uid,'family_name':title,'reviewed_property':contract,'decision':decision,'review_note':note}
# Explicit source-review decisions. The indices refer to the immutable extraction manifest,
# not to name-based inference; each record retains source hash and exact assertion sites.
group('0-2','support.cli_option_compatibility','压测工具命令行选项兼容','比较 Rust/Python 压测命令行选项、别名、默认值及明确允许的差异。','support_property_reviewed')
group('3,5,8,9,12,74,76,79,100,101,109-114,181,183,184,187-191,248,249,251,260,263,266,275,298,305,306,308,316,317,333,490-492,524','response.observed_payload_contract','响应中的被检查字段与内容','在测试声明的请求和模拟上游输出下，比较响应状态、被断言的载荷字段、终止状态或文本。响应状态检查与内容、时序检查分别保存。')
group('4,10,23','output.finalization','输出与停止状态一致','按照显式 token/文本片段与结束原因，检查字符边界、终止 token 或 completion 区间。')
group('6,7,17-19,97,99,102,103,104,121,192-195,197-199,250,255,257,264,267,268,270,271,274,281-285,290,291,311,312,431-435,445-448,455,457,459,534-547,549-552,564,565,567,571,574-576,585-607,612,613','request.declared_admission_contract','请求接受或拒绝符合声明条件','对给定请求格式、字段范围、资源名称或参数组合，检查校验结果、错误信息或允许的 HTTP 状态集合；具体允许集合按入口记录。')
group('11,115-120,122,261,307,452,453','response.stream_event_lifecycle','响应事件与内容项生命周期','检查事件的顺序、内容项身份、结束与错误事件，以及测试实际比较的响应持久化或检索字段。')
group('13,14,15','tool.call_parse_relation','输出解析的内容与工具调用对应','根据给定输出标记与解析选项，检查正文、推理文本、工具名称和参数的对应。')
group('16','response.logprob_identity','返回概率项与 token 对应','检查生成与提示 token 的身份、候选、概率字段及流式汇总。')
group('20','input.prompt_text_rendering','聊天提示词与工具选择对应','检查渲染请求中工具声明与工具选择的传递。')
group('21','support.external_grammar_replay','外部语法回放命令结果','此 Rust 入口调用外部 Python 回放程序并检查退出状态；其具体语法性质需展开外部脚本后确定。','needs_helper_review')
group('22,24-26,91-94,427','support.helper_definition','测试辅助函数','该源码定义提供公共测试过程、配置、分词或请求辅助操作，不能单独按命名视为已收集的独立测试。','helper_identity_reviewed')
group('27,29,34,35,66,67-73,136-138,167-176,465,488,489','worker.registration_configuration','工作节点身份与注册配置','核对节点角色、rank、地址、配置一致性及注册/移除后的状态；各入口保存实际检查的字段和异常条件。')
group('28,36-39,43-46,50,52,61-65','distributed.control_delivery','跨进程请求、控制命令与结果交付','检查控制消息、请求结果和错误的交付，以及多节点响应的汇总或一致性；保留接收端与调用上下文。')
group('30-33','schedule.coordinated_progress','多节点协调状态与执行波次','检查协调消息、波次与 rank/前端状态的对应及广播行为。')
group('40,41,80-83','abort.request_id_binding','取消命令与请求身份绑定','检查丢弃流或按外部标识取消时，发出的 abort 请求与内部请求标识集合对应。')
group('42,47-49,55-60,243-247','service.lifecycle_health','服务生命周期与健康状态','在初始化、等待、睡眠、唤醒、暂停或运行错误后检查服务可用性、超时与健康状态。')
group('51,150','cache.block_identity_encoding','缓存块身份编码','根据给定 token 与元数据计算的块标识与明确参考值一致。')
group('53,54,148,155-162,165,229-241,407-409,424-426,428-430,440,441','schedule.work_distribution','工作节点选择与请求对应','按候选节点、负载、键或策略，检查实际被选择的节点、分布或策略状态；仅成功状态的入口不证明分配公平性。')
group('75,77','response.batch_metadata_binding','生成批次与返回元数据对应','检查 token、概率、路由信息、权重版本及时间戳的合并或维度约束。')
group('78,107,108,177-180,217-228,277,278,300,301,304,410-415,532,533,548,553,554,559-563,566,568,569,572,573,577,578,584,608-611,614-621','request.field_binding','请求字段规范化与传递','检查请求标识、采样参数、工具、前缀分离元数据或任务字段在序列化、规范化及转发中保持声明的对应关系。')
group('84-87,185,186,351-357','service.measurement_record','服务指标与追踪记录','核对指标计数、聚合字段、跟踪标识或导出事件与输入记录和观察到的操作对应；不据此判定模型性能达标。')
group('88-90,123-135,149,212,292-297,314,315,318-329,464,466,468-487,530,531','support.fixture_or_example_contract','测试工具、样本或局部示例检查','检查测试专用分词器、mock 服务、样本、局部构造的请求/配置或第三方工具行为；记录被测对象，不能直接提升为真实推理执行的保障。','support_property_reviewed')
group('151-154','cache.distributed_index_state','跨节点缓存索引与节点状态','检查缓存事件同步、节点增删及并发更新后的索引查询与成员关系。')
group('139-147,164,200-205,213,358-374,385,389-406,459','service.failure_admission_recovery','服务失败后的接纳、恢复与重试','在给定失败、取消或恢复操作序列中检查断路状态、探测数量、失败计数、请求状态或实际重试结果；未断言的阈值和次数不作为已检查性质。')
group('182,196,211,242','response.failure_contract','失败响应与超时返回','检查不可达节点、连接中断或超时的响应分类、可见字段及测试设置的时间界限。')
group('206-210,330-338','service.inflight_accounting','在途请求占用与释放','检查请求完成、失败、断连和负载守卫退出后，在途计数与资源持有状态按生命周期更新。')
group('214,215','service.shutdown_completion','服务关闭与在途请求收尾','检查关闭请求后在途响应完成情况或退出时间界限。')
group('216','response.forwarded_headers','代理响应头遵守传递策略','检查允许的端到端字段保留、逐跳字段移除及 Host/长度等字段的处理。')
group('252-254,256,262,272,273,280,286-289,449-451,454,456,458,460','service.endpoint_observed_fields','管理与服务接口的可观察字段','对实际调用的接口检查状态与响应对象中明确断言的字段；缓存清理接口成功不代表缓存内容已逐项核验。')
group('258,416-423,436-438,442-444,519-524','service.configured_request_success','配置下请求可完成','在声明的路由或安全配置下检查请求完成状态；这些入口未必检查策略选择、身份一致或加密通道。')
group('259,269,276,439','support.incomplete_test_body','仅搭建环境或未完成的测试','源码只搭建环境、保留待实现检查，或使用恒真的断言；未建立其名称声称的检查。','incomplete_test_reviewed')
group('279,570,579-583','rerank.result_contract','重排任务输出与索引关系','检查重排结果排序、top-k、原文保留和用量字段与声明的分数及输入对应。')
group('299,309,310,313','response.history_store','会话与响应记录的存取隔离','检查会话或响应的创建、查询、删除、元数据与共享条目在不同标识下的可见性。')
group('302','response.usage_accounting','响应使用量字段对应','检查用量字段在格式转换中保留明确的数值和计量字段。')
group('303','request.effective_sampling_parameters','默认请求参数生效','检查未显式设置的推理参数按声明的默认策略规范化。')
group('339-350','tool.execution_binding','外部工具注册与调用结果','检查工具连接、目录、schema、执行和失败时的返回与目标工具对应。')
group('375-384,386-388','abort.delivery_persistence','断连或取消后的响应与持久化','按 store 选项和执行阶段检查上游停止、输出停止、继续收集或保存结果的行为；保存相关时间界限与缓冲允许量。')
group('461-463','deployment.configuration_admissibility','持久化后端配置可接受性','检查启用指定历史存储后端时，必需配置缺失会失败、有效组合可接受。')
group('467','deployment.effective_configuration','路由构造配置','检查按指定配置构造路由器后登记的节点数等被断言字段。')
group('493-518','request.access_control','请求身份与权限判定','根据给定凭据、签名、角色和请求路径检查接受/拒绝或身份字段；不把功能查找测试解释为常量时间证明。')
group('525-529','request.transport_authentication','TLS 连接与对端身份验证','按 CA、服务端证书及客户端凭据检查 TLS/mTLS 连接成功或失败。')
group('555-558','embedding.routing_text','向量请求的路由文本投影','检查字符串或 token 输入转换为路由文本的规则，不检查模型生成的向量数值。')
group('622-629','service.extension_registration','服务扩展模块注册与调用记录','检查扩展模块注册、移除、重复键、失败状态或执行指标。执行结果允许失败的入口不被记为扩展输出正确。')
for i,j in [(95,98),(98,107),(105,107),(106,84),(166,330),(265,262)]:
    if j in assigned: assigned[i]=dict(assigned[j])
assigned[95]=dict(assigned[107]);assigned[96]=dict(assigned[303])
group('163','input.text_tokenization','分词结果与参考 token 对应','按模型分词器和样本条件，将分词结果与固定参考 token ID 对比，同时检查有效样本计数。')
assert set(assigned)==set(range(630)), sorted(set(range(630))-set(assigned))
# Corrections from reading complete bodies, rather than accepting misleading names/comments.
corrections={184:'仅在完整收集响应后查找 first 字节；没有测量首块返回时间。',306:'只有一次工具调用且未超过限制；完成状态不能证明超限时能阻止工具调用。',419:'仅检查五次请求成功且收集五个可选节点 ID；没有检查节点 ID 相等。',424:'节点身份断言只在响应含相应字段时执行，需保留该条件。',501:'比较有效与无效 key 的查找结果，没有计时，不能证明抵抗时序攻击。',524:'实际请求使用普通 HTTP，没有建立 TLS 连接。',629:'允许扩展执行失败；实际断言是模块已注册并且至少一种执行计数增加。',276:'is_err() || is_ok() 恒真，没有断言路由正确。',259:'TODO 后直接关闭环境，未检查工作节点选择。',269:'未发送超限载荷，只有环境设置。',439:'未执行节点注册及策略变化断言，只有构造语句。'}
existing={u['id'] for u in json.loads((P/'semantic_space/space_seed.json').read_text())['units']}
# A reviewed subject alone is not a published space mapping. Only explicitly approved
# stronger existing-unit matches are emitted; the rest retains exact reviewed contracts.
allow={4,10,23,13,14,15,16,20,303,302,461,462,463,467}
outputs=[];cross=[]
for i,r in enumerate(rows):
    a=assigned[i]; assertions=r['assertions']; uid=a['family']
    o={k:r[k] for k in ['snapshot','path','test_name','line','end_line','source_sha256']}
    o.update(a);o['review_note']=corrections.get(i,a['review_note']);o['assertions']=assertions
    o['calls']=r['calls'];o['attributes']=r['attributes'];o['source_record']='functions.json#'+str(i)
    o['condition_source']=r['path']+':'+r['line']+'-'+str(r['end_line'])
    o['review_basis']='Read source assertion contract and grouped subjects; special weak/tautological/helper cases checked against complete function body.'
    o['space_status']='mapped_existing' if i in allow and uid in existing and assertions else ('support_or_incomplete' if a['decision']!='property_identified' else 'contract_reviewed_space_refinement')
    outputs.append(o)
    if o['space_status']=='mapped_existing':
        cross.append({**{k:r[k] for k in ['snapshot','path','test_name','line']},'unit_id':uid,
          'property':a['reviewed_property']+' 具体断言：'+' | '.join(' '.join(x['expression'].split()) for x in assertions),
          'oracle_source':';'.join(r['path']+':'+str(x['line']) for x in assertions),
          'conditions':'固定源码快照；采用该入口函数内声明的输入、配置及模拟引擎。条件与调用见 '+o['condition_source'],
          'rule_id':'rust_source_contract_review_'+str(i)})
(D/'reviewed-contracts.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
(D/'reviewed-families.json').write_text(json.dumps(list(families.values()),ensure_ascii=False,indent=2)+'\n')
with (D/'crosswalk.tsv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(cross[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(cross)
from collections import Counter
s={'source_records':len(outputs),'source_assertion_contracts_reviewed':len(outputs),'space_status':dict(Counter(x['space_status'] for x in outputs)),'decisions':dict(Counter(x['decision'] for x in outputs)),'name_claim_corrections':len(corrections)}
(D/'review-summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');print(s)
