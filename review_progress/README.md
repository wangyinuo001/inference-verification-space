# 测试归类进度页

[打开页面](https://wangyinuo001.github.io/inference-verification-space/review_progress/)。每个方格是 vLLM 或 SGLang 中的一个测试，按仓库和文件排列。绿色表示已经阅读测试源码，确认它检查什么，并将它放到验证空间的一个具体位置。其他颜色表示目前只找到部分信息，例如测试所在的推理环节、相关函数或检查语句；这些测试还需要继续核对。颜色表示研究进度，不表示测试是否通过。

目前收录 31,968 个测试，其中 1,042 个已确认检查内容。全量源码扫描为 26,044 个测试找到一个可能所属的推理环节，并在 2,515 个测试中找到被测函数结果进入检查语句的关系。上述数字可以交叠，不能相加。还有 191 个测试的所在目录与实际调用指向不同环节，保留待核对。

页面由 [`render_review_progress.py`](../semantic_space/test_mapping_pilot/render_review_progress.py) 生成。它合并[测试源码索引](../semantic_space/test_mapping_pilot/full-scan-index.tsv)、[逐项审查结果](../semantic_space/oracle-review-queue.tsv)、[推理环节定位](../semantic_space/test_mapping_pilot/stage-placement-all.tsv)和[新确认的 204 个对应关系](../semantic_space/test_mapping_pilot/confirmed-property-crosswalk.tsv)。后者的每一行都保留测试名称、具体检查内容和断言所在源码位置；生成规则见 [`build_confirmed_crosswalk.py`](../semantic_space/test_mapping_pilot/build_confirmed_crosswalk.py)。

```bash
cd semantic_space/test_mapping_pilot
python3 -m unittest discover -p 'test_stage_placement.py'
python3 -c 'import test_confirmed_crosswalk; test_confirmed_crosswalk.test_confirmed_crosswalk()'
python3 render_review_progress.py
cp review_progress.html ../../review_progress/index.html
```

这一页面与[验证空间交互页](../semantic_space/space_view.html)分别展示测试归类进度和推理系统的验证需求。当前展示的验证需求仍在持续核对与修订。
