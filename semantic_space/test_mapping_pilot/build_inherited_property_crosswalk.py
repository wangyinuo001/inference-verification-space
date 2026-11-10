"""Reuse a reviewed base-method oracle across its inherited test entries.

The shared method and assertion are reviewed once. Each subclass keeps its
own model, server configuration and thresholds as applicability conditions.
Only explicit source-method rules are mapped; other inherited methods remain
source-located but not property-confirmed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from resolve_inherited_methods import HERE, SPACE, ROOTS, main as resolve_inherited_methods

OUTPUT = HERE / "inherited-property-crosswalk.tsv"
KEY = ("snapshot", "path", "test_name", "line")
FIELDS = (*KEY, "unit_id", "property", "oracle_source", "conditions", "rule_id")

RULES = {
    ("python/sglang/test/kits/eval_accuracy_kit.py", "test_gsm8k"):
        ("quality.task_score", "GSM8K 得分与测试类设置的门槛比较。",
         "python/sglang/test/kits/eval_accuracy_kit.py:38;python/sglang/test/kits/eval_accuracy_kit.py:196",
         "模型、服务地址、评测后端、题数及门槛由继承该方法的测试类决定。", "gsm8k_score"),
    ("python/sglang/test/ascend/gsm8k_ascend_mixin.py", "test_gsm8k"):
        ("quality.task_score;service.performance", "GSM8K 正确率和输出吞吐分别与测试类设置的门槛比较。",
         "python/sglang/test/ascend/gsm8k_ascend_mixin.py:94;python/sglang/test/ascend/gsm8k_ascend_mixin.py:99",
         "模型、设备、题数、并发和两项门槛由继承该方法的测试类决定。", "ascend_gsm8k_score_throughput"),
    ("python/sglang/test/kits/basic_decode_correctness_kit.py", "test_capital_france"):
        ("quality.sample_semantic_expectation", "请求回答法国首都，检查输出包含 paris。",
         "python/sglang/test/kits/basic_decode_correctness_kit.py:40",
         "使用继承测试类启动的模型与服务；温度为零。", "capital_answer"),
    ("python/sglang/test/kits/basic_decode_correctness_kit.py", "test_basic_math"):
        ("quality.sample_semantic_expectation", "请求计算 17×23，检查输出包含 391。",
         "python/sglang/test/kits/basic_decode_correctness_kit.py:47",
         "使用继承测试类启动的模型与服务；温度为零。", "math_answer"),
    ("python/sglang/test/kits/basic_decode_correctness_kit.py", "test_color_completion"):
        ("quality.sample_semantic_expectation", "请求补全颜色，检查输出包含 yellow。",
         "python/sglang/test/kits/basic_decode_correctness_kit.py:55",
         "使用继承测试类启动的模型与服务；温度为零。", "color_answer"),
    ("python/sglang/test/kits/basic_decode_correctness_kit.py", "test_determinism_temp_zero"):
        ("sampling.policy", "同一提示词以零温度生成两次，比较换行停止前的回答是否相同。",
         "python/sglang/test/kits/basic_decode_correctness_kit.py:101",
         "相同模型与服务；温度为零；以换行符停止。", "zero_temperature_repeat"),
    ("python/sglang/test/kits/basic_decode_correctness_kit.py", "test_max_token_one"):
        ("response.valid", "最大只生成一个 token 时，检查服务返回非空文本。",
         "python/sglang/test/kits/basic_decode_correctness_kit.py:114",
         "使用继承测试类启动的模型与服务；最大生成长度为一个 token。", "one_token_response"),
}

# The source methods below share one assertion helper.  Their inherited
# subclasses change the server/model, but not the checked output relation.
RULES.update({
    ("python/sglang/test/kits/json_constrained_kit.py", "test_json_generate"):
        ("sampling.structured_constraint", "JSON 输出能够解析，name 为字符串且 population 为整数。",
         "python/sglang/test/kits/json_constrained_kit.py:54;python/sglang/test/kits/json_constrained_kit.py:58",
         "使用继承类的模型与服务；JSON Schema 请求；不检查 name 的 pattern 或额外字段。", "json_generate_types"),
    ("python/sglang/test/kits/json_constrained_kit.py", "test_json_openai"):
        ("sampling.structured_constraint", "OpenAI 接口返回的 JSON 能够解析，name 为字符串且 population 为整数。",
         "python/sglang/test/kits/json_constrained_kit.py:89;python/sglang/test/kits/json_constrained_kit.py:95",
         "使用继承类的模型与服务及 JSON Schema response_format；不检查完整 Schema。", "json_openai_types"),
    ("python/sglang/test/kits/json_constrained_kit.py", "test_mix_json_and_other"):
        ("sampling.structured_constraint", "并发请求中，带 JSON Schema 的输出可解析且两个字段类型正确。",
         "python/sglang/test/kits/json_constrained_kit.py:45;python/sglang/test/kits/json_constrained_kit.py:98",
         "40 个并发请求中 20 个带 JSON Schema；其余 20 个无约束请求在该方法中没有输出断言。", "json_mixed_types"),
})

RULES.update({
    ("test/manual/dsv4/_common.py", "test_smoke_gsm8k"):
        ("quality.task_score", "运行 GSM8K 评测，读取汇总分数，并在门槛大于零时比较。",
         "test/manual/dsv4/_common.py:178;test/manual/dsv4/_common.py:268;test/manual/dsv4/_common.py:270",
         "手动测试；各子类启动各自模型和部署配方；样本数、重复次数、评分字段及门槛可由环境变量修改；门槛为零时不做分数断言。",
         "dsv4_gsm8k"),
    ("test/manual/dsv4/_common.py", "test_aime25"):
        ("quality.task_score", "运行 AIME25 评测，读取汇总分数，并与模型配方对应的门槛比较。",
         "test/manual/dsv4/_common.py:192;test/manual/dsv4/_common.py:268;test/manual/dsv4/_common.py:270",
         "手动测试；各子类启动各自模型和部署配方；Flash 默认门槛 0.93，Pro 默认 0.95；环境变量可覆盖门槛和评测配置。",
         "dsv4_aime25"),
    ("python/sglang/test/kits/eval_accuracy_kit.py", "test_mmlu"):
        ("quality.task_score", "运行 MMLU 评测，将汇总分数与测试类门槛比较。",
         "python/sglang/test/kits/eval_accuracy_kit.py:37;python/sglang/test/kits/eval_accuracy_kit.py:252",
         "模型、评测后端、题数、门槛由继承测试类设置；接受长度门槛如有设置则另行检查。", "mmlu_score"),
    ("python/sglang/test/kits/eval_accuracy_kit.py", "test_human_eval"):
        ("quality.task_score", "运行 HumanEval 评测，将汇总分数与测试类门槛比较。",
         "python/sglang/test/kits/eval_accuracy_kit.py:38;python/sglang/test/kits/eval_accuracy_kit.py:384",
         "模型和门槛由继承测试类设置；AMD CI 可采用专门门槛。", "humaneval_score"),
    ("python/sglang/test/kits/eval_accuracy_kit.py", "test_mgsm_en"):
        ("quality.task_score", "运行 MGSM 英文题评测，将汇总分数与测试类门槛比较。",
         "python/sglang/test/kits/eval_accuracy_kit.py:38;python/sglang/test/kits/eval_accuracy_kit.py:412",
         "模型、题数和门槛由继承测试类设置。", "mgsm_en_score"),
    ("python/sglang/test/kits/spec_server_kits.py", "test_gsm8k"):
        ("quality.task_score", "运行 GSM8K 评测，断言汇总分数高于测试类门槛。",
         "python/sglang/test/kits/spec_server_kits.py:195;python/sglang/test/kits/spec_server_kits.py:208",
         "推测解码测试；模型、服务、题数和门槛由继承测试类设置；接受长度指标有值时还进行另一项门槛检查。",
         "spec_gsm8k_score"),
})

_REGEX_CASES = {
    "email": 53, "greeting": 63, "number": 73, "phone": 83,
    "date": 93, "hex_color": 103, "complex_json": 113,
    "custom_log_format": 123,
}
for _name, _line in _REGEX_CASES.items():
    RULES[("python/sglang/test/kits/regex_constrained_kit.py", f"test_regex_generate_{_name}")] = (
        "sampling.structured_constraint",
        f"返回列表中的每项文本非空，且匹配 {_name} 测试声明的正则表达式。",
        f"python/sglang/test/kits/regex_constrained_kit.py:37;python/sglang/test/kits/regex_constrained_kit.py:45;python/sglang/test/kits/regex_constrained_kit.py:{_line}",
        "使用继承类的模型与服务；请求 n=3；以测试声明的锚定正则表达式检查返回项；未断言列表长度，空列表不会触发逐项检查。",
        f"regex_{_name}",
    )

_EBNF_CASES = {
    "email": 61, "greeting": 73, "number": 85, "phone": 100,
    "date": 117, "hex_color": 136, "complex_json": 151,
    "custom_log_format": 175, "all_optional_function_params": 195,
}
for _name, _line in _EBNF_CASES.items():
    RULES[("python/sglang/test/kits/ebnf_constrained_kit.py", f"test_ebnf_generate_{_name}")] = (
        "sampling.structured_constraint",
        f"返回列表中的每项文本非空，且匹配 {_name} 测试声明的至少一个允许格式。",
        f"python/sglang/test/kits/ebnf_constrained_kit.py:40;python/sglang/test/kits/ebnf_constrained_kit.py:53;python/sglang/test/kits/ebnf_constrained_kit.py:{_line}",
        "使用继承类的模型与服务；一般请求 n=3，可选函数参数请求 n=5；未断言列表长度，空列表不会触发逐项检查；检查预期正则，不证明完整文法语言等价。",
        f"ebnf_{_name}",
    )


def main() -> dict[str, int]:
    if all((root / ("vllm" if name == "vllm" else "python/sglang")).is_dir() for name, root in ROOTS.items()):
        resolve_inherited_methods()
    elif not (HERE / "inherited-method-sources.tsv").is_file():
        raise FileNotFoundError("Provide the checked inherited-method-sources.tsv or full source checkouts")
    units = {unit["id"] for unit in json.loads((SPACE / "space_seed.json").read_text())["units"]}
    with (HERE / "inherited-method-sources.tsv").open(encoding="utf-8", newline="") as f:
        entries = list(csv.DictReader(f, delimiter="\t"))
    rows = []
    for entry in entries:
        if entry["status"] != "resolved_unique":
            continue
        rule = RULES.get((entry["source_path"], entry["source_method"]))
        if rule is None:
            continue
        unit_id, prop, source, conditions, rule_id = rule
        assert all(item in units for item in unit_id.split(";"))
        rows.append({**{field: entry[field] for field in KEY}, "unit_id": unit_id,
                     "property": prop, "oracle_source": source,
                     "conditions": conditions, "rule_id": rule_id})
    assert len(rows) == len({tuple(row[field] for field in KEY) for row in rows})
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"mapped": len(rows), "output": str(OUTPUT)}, ensure_ascii=False))
    return {"mapped": len(rows)}


if __name__ == "__main__":
    main()
