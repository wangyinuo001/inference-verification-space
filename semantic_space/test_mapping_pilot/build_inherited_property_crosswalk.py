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

# Request-behavior methods reviewed at their shared definition and delegated
# assertion sites.  These describe only what the assertions check; a source
# location or method name alone never promotes an inherited entry to green.
RULES.update({
    ("python/sglang/test/kits/matched_stop_kit.py", "test_finish_stop_str"):
        ("output.stop_boundary", "补全和聊天接口对换行停止均返回 stop 原因与匹配的换行符。",
         "python/sglang/test/kits/matched_stop_kit.py:48;python/sglang/test/kits/matched_stop_kit.py:52;python/sglang/test/kits/matched_stop_kit.py:92;python/sglang/test/kits/matched_stop_kit.py:96",
         "两种接口、零温度、换行 stop；只检查停止元数据，不断言返回文本的截断边界。", "stop_string_metadata"),
    ("python/sglang/test/kits/matched_stop_kit.py", "test_finish_stop_regex_str"):
        ("output.stop_boundary", "补全和聊天接口对声明的正则停止返回 stop 原因与该正则；聊天接口另检查句末正则。",
         "python/sglang/test/kits/matched_stop_kit.py:48;python/sglang/test/kits/matched_stop_kit.py:52;python/sglang/test/kits/matched_stop_kit.py:92;python/sglang/test/kits/matched_stop_kit.py:96",
         "零温度；and|or 与句末正则；只检查元数据，未检查实际命中位置、文本边界或多个候选的优先级。", "stop_regex_metadata"),
    ("python/sglang/test/kits/matched_stop_kit.py", "test_finish_stop_eos"):
        ("output.stop_boundary", "Llama 格式输入及聊天请求停止时，返回 stop 原因，匹配对象属于给定 EOS token ID 列表。",
         "python/sglang/test/kits/matched_stop_kit.py:48;python/sglang/test/kits/matched_stop_kit.py:52;python/sglang/test/kits/matched_stop_kit.py:92;python/sglang/test/kits/matched_stop_kit.py:96",
         "测试类模型与 EOS 列表相关；只检查元数据，不检查文本边界。", "stop_eos_metadata"),
    ("python/sglang/test/kits/matched_stop_kit.py", "test_finish_length"):
        ("output.stop_boundary", "最大输出长度为五时，两种接口返回 length 原因且 matched_stop 为空。",
         "python/sglang/test/kits/matched_stop_kit.py:48;python/sglang/test/kits/matched_stop_kit.py:52;python/sglang/test/kits/matched_stop_kit.py:92;python/sglang/test/kits/matched_stop_kit.py:96",
         "零温度、max_tokens=5；未断言实际生成 token 数或文本长度。", "length_stop_metadata"),
    ("python/sglang/test/kits/kl_divergence_kit.py", "test_input_output_logprobs_match_prefill_cache_hit"):
        ("cache.result_relation", "对实际前缀缓存命中的样本，将生成时的输出 logprob 与清缓存后重算的输入 logprob 比较平均近似 KL。",
         "python/sglang/test/kl_test_utils.py:245;python/sglang/test/kl_test_utils.py:256;python/sglang/test/kl_test_utils.py:264;python/sglang/test/kl_test_utils.py:114",
         "radix cache 未关闭；只比较命中样本；默认要求命中数超过一半，但环境变量可跳过该断言；阈值、模型和样本数由继承类给定。", "prefill_hit_kl"),
    ("python/sglang/test/kits/kl_divergence_kit.py", "test_input_output_logprobs_match_decode_cache_hit"):
        ("cache.result_relation", "对实际 Decode 缓存命中的样本，将第二轮生成 logprob 与清缓存后重算的输入 logprob 比较平均近似 KL。",
         "python/sglang/test/kl_test_utils.py:314;python/sglang/test/kl_test_utils.py:325;python/sglang/test/kl_test_utils.py:333;python/sglang/test/kl_test_utils.py:114",
         "radix cache 未关闭；只比较命中样本；默认要求命中数超过一半，但环境变量可跳过该断言；阈值、模型和样本数由继承类给定。", "decode_hit_kl"),
    ("python/sglang/test/kits/prefix_cache_branching_kit.py", "test_prefix_cache_branching"):
        ("cache.hit_accounting", "相同前缀的三条分支请求依次报告 0、0、按缓存块大小对齐的前缀命中 token 数。",
         "python/sglang/test/kits/prefix_cache_branching_kit.py:44;python/sglang/test/kits/prefix_cache_branching_kit.py:48",
         "先清缓存；三个顺序请求的共同前缀为 257 次 hi；块大小由继承类设置；只核对 cached_tokens，不比较输出。", "prefix_branch_hit_count"),
})

_TOOL = "tests/tool_parsers/common_tests.py"
RULES.update({
    (_TOOL, "test_no_tool_calls"):
        ("tool.call_parse_relation", "纯文本解析后保持原文，工具调用列表为空。", f"{_TOOL}:152;{_TOOL}:155",
         "解析器、输入和流式模式来自子类 fixture；部分组合可标记 xfail。", "tool_plain_text"),
    (_TOOL, "test_single_tool_call_simple_args"):
        ("tool.call_parse_relation", "解析出一个函数调用，并核对类型、名称及预期参数键值；仅配置了期望正文时才比较正文。",
         f"{_TOOL}:174;{_TOOL}:177;{_TOOL}:178;{_TOOL}:179;{_TOOL}:183",
         "解析器、输入、期望值和流式模式来自子类 fixture；参数只逐项检查配置中列出的键；部分组合可标记 xfail。", "tool_single_call"),
    (_TOOL, "test_parallel_tool_calls"):
        ("tool.call_parse_relation", "核对并行调用的数量、顺序中的函数名与调用 ID 唯一性。",
         f"{_TOOL}:205;{_TOOL}:212;{_TOOL}:213;{_TOOL}:217",
         "期望数量与名称由子类 fixture 设置；不比较参数内容或正文；部分流式组合可标记 xfail。", "tool_parallel_calls"),
    (_TOOL, "test_various_data_types"):
        ("tool.call_parse_relation", "解析出一个调用，参数 JSON 包含七类预期字段；启用类型检查时逐项核对精确类型。",
         f"{_TOOL}:236;{_TOOL}:250;{_TOOL}:255",
         "是否核对类型由子类 supports_typed_arguments 决定；不比较所有字段的具体值；部分组合可标记 xfail。", "tool_argument_types"),
    (_TOOL, "test_empty_arguments"):
        ("tool.call_parse_relation", "解析出一个调用，参数是空字符串或空 JSON 对象的允许形式。",
         f"{_TOOL}:275;{_TOOL}:279;{_TOOL}:281",
         "允许形式由子类 allow_empty_or_json_empty_args 决定；部分组合可标记 xfail。", "tool_empty_args"),
    (_TOOL, "test_surrounding_text"):
        ("tool.call_parse_relation", "带正文的输入至少解析出一个工具调用。", f"{_TOOL}:298",
         "输入由子类 fixture 设置；不比较保留的正文、名称、参数或顺序；部分组合可标记 xfail。", "tool_surrounding_count"),
    (_TOOL, "test_escaped_strings"):
        ("tool.call_parse_relation", "解析出一个调用，其参数可作为非空 JSON 对象解析。",
         f"{_TOOL}:317;{_TOOL}:322",
         "转义输入由子类 fixture 设置；不逐字符比较转义结果；部分组合可标记 xfail。", "tool_escaped_parseable"),
    (_TOOL, "test_streaming_reconstruction"):
        ("tool.call_parse_relation", "同一输入的流式与非流式解析正文及工具数量相等；若有调用，只核对首项名称与参数。",
         f"{_TOOL}:366;{_TOOL}:367;{_TOOL}:369;{_TOOL}:370",
         "解析器和输入由子类 fixture 设置；不比较后续调用或 ID；流式组合可标记 xfail。", "tool_stream_reconstruction"),
})

_VLM = "python/sglang/test/vlm_utils.py"
RULES.update({
    (_VLM, "test_single_image_chat_completion"):
        ("quality.sample_semantic_expectation", "图像回答包含车辆和熨衣或持衣相关关键词，并检查响应字段及 token 用量为正。",
         f"{_VLM}:219;{_VLM}:227;{_VLM}:235;{_VLM}:238;{_VLM}:240",
         "固定熨衣图像、继承类模型与服务；第 225 行的 `or 'person'` 恒真，不构成人物识别检查；关键词仅为样本级条件。", "vlm_single_image"),
    (_VLM, "test_multi_turn_chat_completion"):
        ("quality.sample_semantic_expectation", "图像对话的后续回答包含 man 或 cab，并检查响应字段及 token 用量为正。",
         f"{_VLM}:313;{_VLM}:315;{_VLM}:316;{_VLM}:319;{_VLM}:321",
         "固定图像、先前 assistant 文本和复述请求；模型与服务由继承类设置；不验证图像在内部的绑定。", "vlm_multi_turn"),
    (_VLM, "test_multi_images_chat_completion"):
        ("quality.sample_semantic_expectation", "双图回答分别包含车辆类与 logo 类关键词，并检查响应字段及 token 用量为正。",
         f"{_VLM}:355;{_VLM}:357;{_VLM}:361;{_VLM}:368;{_VLM}:371;{_VLM}:373",
         "固定两图及提示词；模型与服务由继承类设置；不直接检查图像占位位置或顺序。", "vlm_two_images"),
    (_VLM, "test_video_images_chat_completion"):
        ("quality.sample_semantic_expectation", "抽帧后的视频回答命中设备、人物和动作三组关键词且非空。",
         f"{_VLM}:441;{_VLM}:451;{_VLM}:465;{_VLM}:476;{_VLM}:477",
         "固定视频预抽取十帧作为图像输入；只检查样本级词组，不核对帧选择正确性。", "vlm_video_frames_answer"),
    (_VLM, "test_video_chat_completion"):
        ("quality.sample_semantic_expectation", "原生 video_url 请求的回答命中设备、人物、动作和颜色四组关键词且非空。",
         f"{_VLM}:520;{_VLM}:526;{_VLM}:534;{_VLM}:540;{_VLM}:543;{_VLM}:544",
         "固定视频及提示词；继承类模型与服务；只检查样本级词组，不核对视频帧时序。", "vlm_video_answer"),
    (_VLM, "test_audio_speech_completion"):
        ("quality.sample_semantic_expectation", "语音转写回答非空，并包含五个预期短语。",
         f"{_VLM}:147;{_VLM}:148;{_VLM}:98",
         "固定演讲音频、英文转写提示和零温度；模型与服务由继承类设置。", "vlm_speech_answer"),
    (_VLM, "test_audio_ambient_completion"):
        ("quality.sample_semantic_expectation", "环境声音回答非空，且包含 bird。",
         f"{_VLM}:147;{_VLM}:148;{_VLM}:168",
         "固定鸟鸣音频、英文转写提示和零温度；模型与服务由继承类设置。", "vlm_bird_answer"),
    (_VLM, "test_mixed_modality_chat_completion"):
        ("quality.sample_semantic_expectation", "图像加语音回答满足车辆或熨衣相关词组及五个语音短语检查。",
         f"{_VLM}:585;{_VLM}:98;{_VLM}:227;{_VLM}:235",
         "固定图像和演讲音频；第 225 行人物词判断恒真；不验证跨模态位置绑定或两段内容的相互独立性。", "vlm_image_audio_answer"),
})


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
