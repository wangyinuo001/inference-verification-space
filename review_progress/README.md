# 测试归类进度页

[打开页面](https://wangyinuo001.github.io/inference-verification-space/review_progress/)。每个方格是 vLLM 或 SGLang 中的一个测试，按仓库和文件排列。绿色表示已经阅读测试源码，确认它检查什么，并将它放到验证空间的一个具体位置。其他颜色表示目前只找到部分信息，例如测试所在的推理环节、相关函数或检查语句；这些测试还需要继续核对。颜色表示研究进度，不表示测试是否通过。

目前收录 31,968 个测试，其中 1,526 个已确认检查内容。全量源码扫描为 26,044 个测试找到一个可能所属的推理环节，并在 2,515 个测试中找到被测函数结果进入检查语句的关系。上述数字可以交叠，不能相加。还有 191 个测试的所在目录与实际调用指向不同环节，保留待核对。

有些测试方法写在公共测试类中，再由不同模型和配置的测试类继承。程序现已为全部 1,366 个继承的源码级测试入口找到实际方法源码；其中 484 个通过核对公共方法及其检查语句，确认了任务评分、输出样本检查、零温度重复生成和约束格式检查等内容。继承类的模型、服务配置和门槛仍作为适用条件记录。其他继承测试只标记方法位置，继续核对具体检查内容。

格式约束组有两项值得注意。正则和文法测试会逐项检查返回内容，却没有断言返回项数量，空列表也能绕过逐项检查。JSON 格式测试检查了能否解析及两个字段的类型，没有检查完整 Schema；无效 JSON 约束测试没有结果断言，因此未列入绿色入口。

页面由 [`render_review_progress.py`](../semantic_space/test_mapping_pilot/render_review_progress.py) 生成。它合并[测试源码索引](../semantic_space/test_mapping_pilot/full-scan-index.tsv)、[逐项审查结果](../semantic_space/oracle-review-queue.tsv)、[推理环节定位](../semantic_space/test_mapping_pilot/stage-placement-all.tsv)、[另一批确认的 204 个对应关系](../semantic_space/test_mapping_pilot/confirmed-property-crosswalk.tsv)，以及[继承方法位置](../semantic_space/test_mapping_pilot/inherited-method-sources.tsv)和[484 个已核对的公共方法检查](../semantic_space/test_mapping_pilot/inherited-property-crosswalk.tsv)。每条已确认的对应关系都保留测试名称、检查内容和源码位置。

```bash
cd semantic_space/test_mapping_pilot
python3 -m unittest discover -p 'test_*.py'
python3 render_review_progress.py
cp review_progress.html ../../review_progress/index.html
```

这一页面与[验证空间交互页](../semantic_space/space_view.html)分别展示测试归类进度和推理系统的验证需求。当前展示的验证需求仍在持续核对与修订。
