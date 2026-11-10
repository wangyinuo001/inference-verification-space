# 继承公共方法的请求行为检查：第二批源码回读

固定源码仍为 vLLM `76a0ccc` 与 SGLang `49e384c`。逐项重读公共方法、它调用的断言辅助函数，以及继承入口的源码定位；不运行 GPU 或服务端测试。每条确认记录的属性、断言位置和适用条件在 [inherited-property-crosswalk.tsv](inherited-property-crosswalk.tsv) 中。绿色表示**测试源码确实写了该局部检查**，不表示 CI 已执行，也不表示对应空间单元的全部条件得到覆盖。

| 公共方法组 | 新增交叉映射 | 源码中实际检查的范围 |
|---|---:|---|
| [SGLang 停止条件](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kits/matched_stop_kit.py#L13) | 40 | 两种接口的 `finish_reason` 与 `matched_stop`；不检查文本截断、命中位置或候选优先级 |
| [SGLang 缓存 logprob](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kl_test_utils.py#L219) | 26 | 实际缓存命中样本的生成 logprob 与清缓存重算的输入 logprob 的平均近似 KL；缓存关闭时直接返回，命中比例断言可由环境变量跳过 |
| [SGLang 分支前缀](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/kits/prefix_cache_branching_kit.py#L21) | 9 | 三条顺序请求的命中 token 数为 0、0、按块对齐的前缀长度；不比较生成结果 |
| [vLLM 工具解析](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/tests/tool_parsers/common_tests.py#L137) | 64 | 八种公共方法分别检查正文、调用数量、部分名称和参数、空参数、流式重建；各方法的断言强度不同，部分组合有 `xfail` |
| [SGLang 多模态回答](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/vlm_utils.py#L88) | 54 | 固定图像、音频、视频样本的关键词及部分响应字段；不据此断言媒体顺序、内部位置绑定或普遍任务质量 |

合计新增 193 条公共方法交叉映射，其中 40 条多模态入口此前已有人工绿色记录，本轮补齐了更精确的条件与断言；进度页绿色入口从 1,526 净增至 1,679。全部 1,366 个继承入口的方法源码已定位；目前 677 个具有逐项确认的公共方法属性，其余条目继续按各自证据保留，不因方法定位自动升级。

两个容易高估的检查单独保留边界：[vLLM `test_malformed_input`](https://github.com/vllm-project/vllm/blob/76a0ccc209343971c6f4b56f6380a76202428385/tests/tool_parsers/common_tests.py#L324) 遍历输入调用解析器，但没有对返回值作结果断言，因此没有升级；[SGLang `test_mixed_batch`](https://github.com/sgl-project/sglang/blob/49e384ce9d304648e9959666ecb8ce8cd98d0deb/python/sglang/test/vlm_utils.py#L200) 只检查角色与文本类型，没有比较每条请求与其图像或输出的对应关系。单图辅助检查第 225 行的 `"man" in text or "person"` 因非空字符串分支恒真，不构成人物关键词检查；本次只记录实际有效的车辆与熨衣相关条件。

本批将已核对的局部性质放到早期候选空间单元，尚未将这些旧单元自动转移到 76 单元的源码派生模型。转换时需再次核对输入、状态、判定方式与组合缺口。
