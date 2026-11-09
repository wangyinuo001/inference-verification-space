# SGLang 批量调用—断言关联留出核查

逐项回读固定 SGLang 0.5.14 快照中 `bulk-call-oracle-holdout-80.tsv` 的 40 条 SGLang 记录。判为 `true` 的标准是候选调用产生了被检查的对象或值，且表中至少一项断言直接读取该返回值、其字段或同一对象的后续结果。判为 `false` 表示候选调用虽出现在断言的数据依赖中，却只生成参考值或准备材料，不能将它标为本项测试的被测结果。逐项源码行、判定与输入追踪见 [TSV](agent_bulk_sglang_holdout.tsv)。这项审查核对的是静态源码关联，不代表测试已实际运行。

40 条中，39 条候选调用与至少一项断言存在直接结果关联，1 条是参考值误连，未出现无法判断的条目。24 条仍需把 helper、fixture、随机值、mock 或对象状态继续绑定到具体输入；其余 16 条的调用条件可由测试体中的字面值及局部静态构造读出。这一划分只描述输入证据是否已经完整，不能直接推断这些测试覆盖的验证性质相同。

误连发生在 `test/registered/mem_cache/test_int8_checkpoint_store.py:54`。测试先用 `store.store` 和 `store.load` 得到待测输出 `out`（65–66 行），再调用 `Int8CheckpointStore.dequantize` 生成参考值 `ref`（68–71 行），最后比较二者误差（72 行）。批量关联把 69 行的 `dequantize` 当作候选产品调用；该调用确实进入断言，但作为预期值一侧，不能替代 `store.load` 所代表的被测路径。产品实现 `python/sglang/srt/mem_cache/mamba_checkpoint_pool.py:142–154` 也显示 `load` 会读取已存储的量化状态后反量化，测试要检查的是整条存取路径。

另有三种需要保留的限定。其一，多行断言的调用行可能晚于断言起始行，例如 `resolve_real_kv_read_bytes`、`group_concurrent_contiguous` 和 `trim_matched_stop`，这些仍是直接断言调用。其二，构造器返回的对象可被检查属性，后续方法的返回值则还需另记依赖；`NetworkAddress` 的 `is_ipv6` 和 `to_url()` 是一个例子。其三，表中只选取部分断言，例如 `process_content_for_template_format` 除返回字典外还会修改 `image_data`，原测试在 208 行额外检查该容器为空；单条候选关联不应被扩展成完整测试性质。上述情况已逐条写入 TSV 的 `review_note`。
