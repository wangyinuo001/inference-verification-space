# SGLang 第十三批源码性质核对
本批检查 8 个单元测试文件及对应实现，整理 175 个仍待归类的测试入口和 2 个结构性辅助检查。36 项映射现有单元，139 项提出 5 个新增单元；25 个源码文件记录 SHA-256。没有执行上游测试或 GPU 任务，没有改动共享空间或已有确认项。

本批重点是状态和配置检查。采样批测试主要检查 prepare、filter、merge、teardown 后的标志、字段、形状、引用与回调；这些断言不能归为 logits 数值正确性。推测测试检查接受历史驱动的预算控制以及 batch/graph 配置路由，不验证推测输出分布。请求状态测试使用最小管理器、模拟调度器与本地事件，检查结束、异常和重新提交中的状态表；不将其视为跨进程竞态实验。

新单元及条目数：sampling.batch_state_transition 39；spec.adaptive_budget_transition 29；deployment.extension_dispatch_contract 34；deployment.parser_selection 24；request.identity_state_lifecycle 13。每个新单元提供具体函数/类源码位置及结构化条件。framework_scope 均为 sglang，需完成另一框架对照后再扩大适用范围。

crosswalk.tsv 的 property 保留该项实际断言；oracle_source 仅为 path:line。evidence.json 保存测试正文、输入赋值、循环、调用、共同辅助函数、模块和类常量以及对应实现文件。少数测试以正常返回作为判定，明确记录为 implicit_no_exception，不写成输出数值已验证。FimPosition 枚举和 CompletionTemplate 字段两项保留 supporting_records，未强行映射请求性质。

本批用静态解析检查导出结构及证据位置，结果不代表执行通过。既有确认项由 full_delivery 状态排除；复核源码摘要一致，入口标识无重复，所有断言位置落在测试体内，新增单元锚点均为实际相关定义。
