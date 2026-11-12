# 源码审查与空间补充

本目录保存候选筛选、逐文件源码审查及空间修订材料。审查通过的 1,139 条补充映射已由 `build_reviewed_supplement.py` 合入 `reviewed-supplement-crosswalk.tsv`，不再等待凑满数量后发布。原始候选池为 6,330 条，候选建议与审查结论分开保存。

每条接受的映射保留仓库快照、源码哈希、测试定义、参考或断言位置、实际检查的局部性质与适用条件。待细分的条目保留具体原因；新的性质建议须对照两个框架实现后再进入空间。

| 审查文件 | 已归类入口 |
|---|---:|
| [reviewed-reference-batch1.tsv](reviewed-reference-batch1.tsv) | 125 |
| [reviewed-sampling-batch1.tsv](reviewed-sampling-batch1.tsv) | 27 |
| [reviewed-tool-parser-batch1.tsv](reviewed-tool-parser-batch1.tsv) | 251 |
| [reviewed-tool-parser-batch2.tsv](reviewed-tool-parser-batch2.tsv) | 90 |
| [reviewed-ngram-draft-batch1.tsv](reviewed-ngram-draft-batch1.tsv) | 37 |
| [reviewed-parser-batch3.tsv](reviewed-parser-batch3.tsv) | 156 |
| [reviewed-reference-batch2.tsv](reviewed-reference-batch2.tsv) | 164 |
| [reviewed-reference-batch3.tsv](reviewed-reference-batch3.tsv) | 150 |
| [reviewed-sglang-runtime-batch.tsv](reviewed-sglang-runtime-batch.tsv) | 100 |
| [reviewed-rendering-batch1.tsv](reviewed-rendering-batch1.tsv) | 28 |
| [reviewed-sglang-logit-transform-proposal.tsv](reviewed-sglang-logit-transform-proposal.tsv) | 11 |

数值、解析、请求状态与提示词文本的独立说明和证据在同名 Markdown/JSON 中。`local-review-audit.json` 仅记录最早五批的历史审查；当前合并数量以 `../reviewed-supplement-summary.json` 为准。
