# vLLM 调用—断言留出核查

逐条回读固定 vLLM 源码 `76a0ccc209343971c6f4b56f6380a76202428385` 中 [`bulk-call-oracle-holdout-80.tsv`](bulk-call-oracle-holdout-80.tsv) 的 40 条 vLLM 关联。核查范围是候选产品调用的返回值或其构造状态，是否确实进入该行列出的断言。逐项结果、调用行、断言行及性质见 [`agent_vllm_bulk_holdout_review.tsv`](agent_vllm_bulk_holdout_review.tsv)。

这 40 条按上述**局部调用—所列断言**口径，true 40、false 0、unclear 0。其中 29 条直接比较产品函数的值或返回对象字段，6 条断言检查构造对象自身的状态，3 条先用 `model_validate` 解析上游服务响应，1 条仅检查用于后续测试的处理器类型，1 条除返回值外还测执行时间，但静态关联只列出返回值断言。这个留出集原本就选自 `direct_assertion_dataflow` 候选；不能据此推断所有静态候选或全部测试的正确率。

**局部链接成立仍需区分验证对象。** `tests/entrypoints/pooling/embed/test_online.py:771` 的 `PoolingResponse.model_validate`、`tests/entrypoints/pooling/embed/test_online_long_text.py:257` 的 `EmbeddingResponse.model_validate` 和 `tests/entrypoints/pooling/scoring/test_late_interaction_online_vision.py:123` 的 `RerankResponse.model_validate` 都返回被断言检查的对象，但维度、用量与相关性得分来自此前的服务或 engine 响应。把这些测试映射成“响应解析器保证 embedding/rerank 语义”会错误归因。`tests/models/multimodal/processing/test_unlimited_ocr.py:68` 的 `create_processor` 只在所列断言中验证处理器类型；crop-grid token 预算由后续 `get_dummy_mm_inputs` 和 `get_num_image_tokens` 比较，不能归功于构造动作本身。

**静态链接还可能遗漏同一测试内的另一项性质。** `tests/entrypoints/serve/exception_handling/test_error_sanitization.py:171` 的 `sanitize_message` 结果与原消息相等，关联成立，但测试还在第 173 行断言耗时低于 0.1 秒。`tests/models/transformers/fusers/test_rms_norm.py:349` 的 `get_fuser` 返回值被检查，随后又调用 `fuse`，检查 `variance_epsilon`。`tests/models/multimodal/processing/test_glm4_1v.py:50` 的帧数结果被比较，另外还核对两个 helper 的调用次数。结果说明候选链接可作局部证据，归类时仍要读取同一测试的其他调用和断言。

可用 [`agent_vllm_bulk_holdout_review.py`](agent_vllm_bulk_holdout_review.py) 重新生成逐项表。该脚本确认每条调用行和断言行均落在固定源码快照内；`true` 只表示这条局部数据关联，不表示测试运行通过，也不表示候选调用承担整项测试的全部验证性质。
