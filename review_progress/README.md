# 测试归类进度页

[打开页面](https://wangyinuo001.github.io/inference-verification-space/review_progress/)。一个方格对应一个固定快照中的源码级测试入口，排列顺序按仓库和文件。颜色表示目前掌握的证据级别，不表示测试通过率。

这轮对 31,968 个入口进行全量路由。26,044 个具有单一验证环节线索，其中 1,371 个还有“被测调用的返回值进入断言”的直接源码关联；191 个入口的目录领域与直接调用目标指向不同环节，等待进一步判定。具体检查性质经源码核对并映射到早期候选空间的入口仍为 838 个，显示为绿色。青色的直接关联不自动证明输入和运行状态满足某项完整验证要求。

批量路由使用 [`build_stage_placement.py`](../semantic_space/test_mapping_pilot/build_stage_placement.py)，输入为[全量源码索引](../semantic_space/test_mapping_pilot/full-scan-index.tsv)、[逐入口审查台账](../semantic_space/oracle-review-queue.tsv)和[调用—断言关联](../semantic_space/bulk-call-oracle-links.tsv)，输出为[逐入口环节表](../semantic_space/test_mapping_pilot/stage-placement-all.tsv)。页面由 [`render_review_progress.py`](../semantic_space/test_mapping_pilot/render_review_progress.py) 生成。`domain_stage_candidate` 仅根据测试的领域线索定位大致环节；`linked_stage` 还要求直接调用与断言之间有源码数据流关联。两者均保留原始目标与断言信息供复查。

```bash
cd semantic_space/test_mapping_pilot
python3 -m unittest discover -p 'test_stage_placement.py'
python3 render_review_progress.py
cp review_progress.html ../../review_progress/index.html
```

这一页面与[验证空间交互页](../semantic_space/space_view.html)独立。后者仍使用早期候选空间；当前源码派生的 76 单元模型在 [`source_model/`](../source_model/) 中。
