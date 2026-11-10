# 源码派生的 Verification Space 模型

从 vLLM（`76a0ccc`）与 SGLang 0.5.14（`49e384c`）源码中整理出的验证单元、分层依据与框架映射。源码与测试的对照见[研究说明](../docs/source-model.md)。

| 文件 | 内容 |
|---|---|
| `space_model.json` | 模型本体：维度、条件、约束、执行扰动、76 个单元 |
| `units.md` | 按判定层次列出的单元表，源码位置链接到固定 commit |
| `build_model.py` | 模型定义，运行后生成 `space_model.json` |
| `check_model.py` | 校验模型；`--changed` 查询改动文件影响的单元范围 |
| `render_units.py` | 生成 `units.md` |
| `traces/` | 八条源码调研线的原始记录；启动前配置与共享槽位单列 |
| `scope-*.txt` | 两次改动范围查询的输出示例 |

```bash
python build_model.py && python check_model.py && python render_units.py
python check_model.py --changed s:mem_cache/radix_cache.py
```

校验需要两份源码快照，位于上层目录的 `third_party/reference_repos/`，或用环境变量 `VERIFICATION_REFERENCE_REPOS` 指定。源码位置写作 `v:<相对 vllm/ 的路径>:<行区间>` 或 `s:<相对 python/sglang/srt/ 的路径>:<行区间>`。

`S0_preflight` 表示服务启动前的配置处理，后续 `S1` 至 `S9` 表示请求进入服务后的路径。旧候选空间中标为 `mapped` 的入口仍需按新单元的输入、状态和断言逐项核对；`legacy_crosswalk.tsv` 仅指出性质定义之间的对应或缺口。

交互页面目前仍展示旧候选空间；本目录是独立的源码派生模型，尚未替换页面数据。
