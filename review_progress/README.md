# 测试归类进度页

[打开页面](https://wangyinuo001.github.io/inference-verification-space/review_progress/)。每个方格是 vLLM 或 SGLang 中的一个测试，按仓库和文件排列。绿色表示已经阅读测试源码，确认它检查什么，并将它放到验证空间的一个具体位置。其他颜色表示目前只找到部分信息，例如测试所在的推理环节、相关函数或检查语句；这些测试还需要继续核对。颜色表示研究进度，不表示测试是否通过。总览按仓库和源码文件路径排列；搜索与状态筛选会收拢匹配方格，点击或用方向键可查看测试详情。方格状态互斥，顶部几项统计可交叠。

目前收录 31,968 个测试，其中 5,330 个已确认检查内容。全量源码扫描为 26,044 个测试找到一个可能所属的推理环节，并在 2,515 个测试中找到被测函数结果进入检查语句的关系。上述数字可以交叠，不能相加。还有 191 个测试的所在目录与实际调用指向不同环节，保留待核对。

有些测试方法写在公共测试类中，再由不同模型和配置的测试类继承。程序现已为全部 1,366 个继承的源码级测试入口找到实际方法源码；其中 891 个通过核对公共方法及其检查语句，确认了任务评分、输出样本检查、停止元数据、缓存命中与数值关系、工具调用解析、多模态样本回答、推测解码、会话取消恢复及 reasoning 用量等局部检查。此前两轮继承方法审查分别净增 153 和 214 条绿色入口；[第三轮普通测试审查](../semantic_space/test_mapping_pilot/direct-property-review-round3.md)净增 251 条；[第四轮](../semantic_space/test_mapping_pilot/direct-property-review-round4.md)净增 560 条；[第五轮](../semantic_space/test_mapping_pilot/direct-property-review-round5.md)净增 454 条；[第六轮](../semantic_space/test_mapping_pilot/direct-property-review-round6.md)净增 1,106 条；[第七轮](../semantic_space/test_mapping_pilot/direct-property-review-round7.md)净增 1,066 条。继承类的模型、服务配置和门槛仍作为适用条件记录。其他继承测试只标记方法位置，继续核对具体检查内容。

格式约束组有两项值得注意。正则和文法测试会逐项检查返回内容，却没有断言返回项数量，空列表也能绕过逐项检查。JSON 格式测试检查了能否解析及两个字段的类型，没有检查完整 Schema；无效 JSON 约束测试没有结果断言，因此未列入绿色入口。

本轮扩展也保留了[逐项审查边界](../semantic_space/test_mapping_pilot/inherited-request-behavior-review.md)：停止测试只检查原因与 `matched_stop`，不检查文本截断；缓存 KL 测试只比较实际命中的样本；vLLM 工具解析的 malformed-input 公共方法没有结果断言；SGLang 混合图像批次测试只检查响应角色和文本类型，不能推出图像绑定或各请求独立。源代码中的 `xfail`、环境变量跳过及子类配置仍影响实际执行和适用范围。

页面由 [`render_review_progress.py`](../semantic_space/test_mapping_pilot/render_review_progress.py) 生成。它合并[测试源码索引](../semantic_space/test_mapping_pilot/full-scan-index.tsv)、[逐项审查结果](../semantic_space/oracle-review-queue.tsv)、[推理环节定位](../semantic_space/test_mapping_pilot/stage-placement-all.tsv)、[244 个已确认的对应关系](../semantic_space/test_mapping_pilot/confirmed-property-crosswalk.tsv)、[3,713 个直接审查的对应关系](../semantic_space/test_mapping_pilot/direct-property-crosswalk.tsv)，以及[继承方法位置](../semantic_space/test_mapping_pilot/inherited-method-sources.tsv)和[891 个已核对的公共方法检查](../semantic_space/test_mapping_pilot/inherited-property-crosswalk.tsv)。每条已确认的对应关系都保留测试名称、检查内容和源码位置。

```bash
cd semantic_space/test_mapping_pilot
python3 -m unittest discover -p 'test_*.py'
python3 render_review_progress.py
cp review_progress.html ../../review_progress/index.html
```

这一页面与[验证空间交互页](../semantic_space/space_view.html)分别展示测试归类进度和推理系统的验证需求。当前展示的验证需求仍在持续核对与修订。

交互页的测试 Explorer 按需读取上述 891 条继承公共方法检查和 3,957 条单独的源码确认映射；用 [`sync_inherited_space_view.py`](../semantic_space/test_mapping_pilot/sync_inherited_space_view.py) 将交叉映射写入 `explorer_extra.json`。归类页的绿色单元名称可打开图中对应节点；图节点也可筛出归入它的已确认测试。显式跳过的测试类仍不在图中标亮候选单元，详情中说明其源码检查未执行。交互页仍使用早期候选空间，这一步不将映射自动转移到 76 单元的源码派生模型。
