"""Map the 61 held renderer/tokenizer entries to source-grounded needs.

The proposed IDs are *not* added to the public ontology or crosswalk.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from review_tool_parser_batch1 import REPOS, rows

HERE = Path(__file__).resolve().parent
SG = REPOS["sglang"]
V = REPOS["vllm"]

# Lines are tied to source hashes in the batch-three evidence. Each held entry
# is deliberately assigned once after inspecting its asserted expression.
ROUTES = {
    "tests/renderers/test_inkling.py": {
        "input.media_marker_rendering": {154, 243},
        "input.reasoning_control_rendering": {189, 325, 346, 362, 391, 409},
        "input.prompt_text_rendering": {192, 196, 206, 216, 227, 254, 264, 288},
        "input.rendering_input_validation": {303, 317, 422, 432},
    },
    "tests/tokenizers_/test_deepseek_v4.py": {
        "input.renderer_registry_binding": {68},
        "input.reasoning_control_rendering": {77, 90, 104, 300, 314, 338, 366, 377},
        "input.prompt_text_rendering": {118, 138, 173, 202, 235, 534, 556, 576, 610},
        "input.rendering_input_validation": {191},
        "input.media_marker_rendering": {451},
        "input.rendering_input_stability": {640},
    },
    "tests/tokenizers_/test_deepseek_v41.py": {
        "input.prompt_text_rendering": {41, 166, 230, 304, 420, 439, 492},
        "input.reasoning_control_rendering": {79, 134, 160, 473},
        "input.rendering_input_validation": {104, 111, 146, 299, 511},
        "input.rendering_input_stability": {196},
        "input.rendering_interface_equivalence": {216},
        "input.tokenizer_invocation_contract": {254},
        "input.media_marker_rendering": {271},
    },
}

SUGGESTIONS = {
    "input.prompt_text_rendering": {
        "title": "聊天消息到提示词文本的构造",
        "property": "固定模板、角色、消息内容、工具和生成续写条件下，输出提示词文本中的内容、角色边界、工具描述、分隔符及终止/续写标记符合声明格式。只比较已断言的文本部分；不推断 token ID 正确。",
        "conditions": ["template_profile", "message_roles_and_order", "tools", "generation_or_continuation_mode"],
        "observables": ["rendered_prompt_text", "role_markers", "tool_sections", "generation_prefix"],
        "sglang_evidence": [("python/sglang/srt/parser/conversation.py", 107, "implementation"),
                            ("test/registered/unit/parser/test_conversation.py", 39, "test"),
                            ("python/sglang/srt/entrypoints/openai/serving_chat.py", 723, "implementation")],
        "cross_framework_status": "shared_function_and_test_evidence",
    },
    "input.reasoning_control_rendering": {
        "title": "推理模式控制与提示词前缀",
        "property": "请求中的推理开关或预算，经指定模型格式映射为解析/渲染模式和相应提示词前缀；默认值、关闭值及不支持值按格式处理。",
        "conditions": ["model_format", "reasoning_effort", "thinking_flag", "template_profile"],
        "observables": ["resolved_mode", "rendered_effort_header", "thinking_prefix"],
        "sglang_evidence": [("python/sglang/srt/entrypoints/openai/serving_chat.py", 493, "implementation"),
                            ("python/sglang/srt/entrypoints/openai/serving_chat.py", 739, "implementation"),
                            ("test/registered/unit/entrypoints/openai/test_serving_chat.py", 1729, "test")],
        "cross_framework_status": "shared_function_and_test_evidence",
    },
    "input.media_marker_rendering": {
        "title": "媒体占位标记在提示词文本中的位置",
        "property": "输入媒体项目在渲染文本中生成所声明的占位标记，并保留与文本片段的出现顺序；本单元只检查标记文本，不代替媒体张量与占位位置的绑定检查。",
        "conditions": ["template_profile", "media_type", "text_media_order"],
        "observables": ["rendered_prompt_text", "media_marker_positions"],
        "sglang_evidence": [("python/sglang/srt/parser/conversation.py", 574, "implementation"),
                            ("test/registered/unit/parser/test_conversation.py", 520, "test"),
                            ("test/registered/unit/parser/test_conversation.py", 554, "test")],
        "cross_framework_status": "shared_function_and_test_evidence",
    },
    "input.rendering_input_validation": {
        "title": "渲染输入与配置的有效性判定",
        "property": "渲染器对指定的不合法角色、媒体类型、工具实参、推理参数或缺失特殊 token 返回相应的异常或请求错误；逐条记录拒绝条件和错误类型。",
        "conditions": ["invalid_input_kind", "template_profile", "validation_stage"],
        "observables": ["exception_type", "error_message_or_status"],
        "sglang_evidence": [("python/sglang/srt/entrypoints/openai/serving_chat.py", 498, "implementation"),
                            ("test/registered/unit/entrypoints/openai/test_serving_embedding.py", 331, "test")],
        "cross_framework_status": "shared_general_need_different_error_cases",
    },
    "input.rendering_input_stability": {
        "title": "渲染过程不改变调用方输入",
        "property": "渲染后调用方提供的消息与可重复读取的内容保持原状态；对迭代器还需分别记录再次渲染是否产生相同结果。",
        "conditions": ["mutable_message_input", "iterable_content", "repeated_render"],
        "observables": ["input_before_after", "repeated_prompt"],
        "sglang_evidence": [("python/sglang/srt/entrypoints/openai/serving_chat.py", 691, "implementation"),
                            ("python/sglang/srt/entrypoints/openai/serving_chat.py", 699, "implementation")],
        "cross_framework_status": "shared_implementation_candidate_sglang_oracle_pending",
    },
    "input.rendering_interface_equivalence": {
        "title": "等义请求表示的提示词关系",
        "property": "同一语义消息经 Chat 与 Responses 等不同入口表示时，规范化后的消息及最终渲染提示词满足声明的等价关系。",
        "conditions": ["request_api_kind", "content_part_representation", "template_profile"],
        "observables": ["normalized_messages", "rendered_prompt_text"],
        "sglang_evidence": [("test/registered/unit/entrypoints/openai/test_serving_responses.py", 80, "test"),
                            ("python/sglang/srt/entrypoints/openai/serving_chat.py", 723, "implementation")],
        "cross_framework_status": "shared_normalization_need_sglang_final_render_oracle_pending",
    },
    "input.tokenizer_invocation_contract": {
        "title": "提示词编码调用的特殊 token 与截断参数",
        "property": "渲染提示词交给指定分词器时，特殊 token 策略和截断参数按调用约定传递，避免重复 BOS；此项检查参数传递，完整 token ID 序列另行核对。",
        "conditions": ["tokenizer_special_token_policy", "truncation", "max_length"],
        "observables": ["encoder_call_kwargs", "encoded_token_ids"],
        "sglang_evidence": [("python/sglang/srt/entrypoints/openai/serving_chat.py", 805, "implementation"),
                            ("python/sglang/srt/entrypoints/openai/serving_chat.py", 818, "implementation")],
        "cross_framework_status": "shared_implementation_candidate_sglang_oracle_pending",
    },
    "input.renderer_registry_binding": {
        "title": "格式标识到渲染器实现的绑定",
        "property": "给定格式或模型标识，注册表选择相应的分词器和渲染器/模板实现；此项只检查选择结果，不推断其输出正确。",
        "conditions": ["format_id", "model_id", "registry_configuration"],
        "observables": ["selected_renderer", "selected_tokenizer_or_template"],
        "sglang_evidence": [("python/sglang/srt/parser/conversation.py", 495, "implementation"),
                            ("test/registered/unit/parser/test_conversation.py", 729, "test")],
        "cross_framework_status": "analogous_registry_semantics_different_api",
    },
}


def main() -> None:
    previous = json.loads((HERE / "reviewed-parser-batch3-evidence.json").read_text(encoding="utf-8"))
    held = [e for e in previous["entries"] if e["status"] == "needs_refinement"
            and e["path"] in ROUTES]
    assert len(held) == 61
    route_by_line = {}
    for path, groups in ROUTES.items():
        for unit, lines in groups.items():
            for line in lines:
                key = path, line
                assert key not in route_by_line, key
                route_by_line[key] = unit
    assert len(route_by_line) == 61
    candidates = {(r["path"], int(r["line"])): r for r in rows(HERE / "candidates.tsv")
                  if r["snapshot"] == "vllm"}
    evidence_files = {}
    for suggestion in SUGGESTIONS.values():
        anchors = []
        for path, line, kind in suggestion["sglang_evidence"]:
            source = (SG / path).read_bytes()
            lines = source.decode("utf-8", errors="replace").splitlines()
            assert 1 <= line <= len(lines) and lines[line - 1].strip(), (path, line)
            digest = hashlib.sha256(source).hexdigest()
            evidence_files[path] = digest
            anchors.append({"snapshot": "sglang", "path": path, "line": line,
                            "source_sha256": digest, "kind": kind,
                            "source_line": lines[line - 1].strip()})
        suggestion["sglang_evidence"] = anchors
    mapped = []
    for entry in held:
        key = entry["path"], entry["line"]
        assert key in route_by_line, key
        candidate = candidates[key]
        assert candidate["test_name"] == entry["test_name"]
        assert candidate["source_sha256"] == entry["source_sha256"]
        unit = route_by_line[key]
        source = (V / entry["path"]).read_bytes()
        assert hashlib.sha256(source).hexdigest() == entry["source_sha256"]
        mapped.append({
            "snapshot": "vllm", "path": entry["path"],
            "test_name": entry["test_name"], "line": entry["line"],
            "source_sha256": entry["source_sha256"],
            "oracle_source": ";".join(f"{entry['path']}:{n}" for n in entry["oracle_lines"]),
            "asserted_checks": candidate["checks"],
            "proposed_unit": unit,
            "review_status": "proposal_only_not_published",
            "scope_note": "The mapping covers only the directly asserted behavior, not the whole proposed unit.",
        })
    assert len(mapped) == 61
    out = {
        "status": "ontology_proposals_only",
        "source_snapshot": "vllm-20261105",
        "sglang_snapshot": "sglang-v0.5.14-complete",
        "entry_count": len(mapped),
        "proposed_unit_counts": dict(sorted(Counter(m["proposed_unit"] for m in mapped).items())),
        "proposed_units": SUGGESTIONS,
        "entry_suggestions": sorted(mapped, key=lambda x: (x["path"], x["line"])),
    }
    (HERE / "reviewed-parser-batch3-refinement.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ("snapshot", "path", "test_name", "line", "source_sha256", "oracle_source",
              "asserted_checks", "proposed_unit", "review_status", "scope_note")
    with (HERE / "reviewed-parser-batch3-refinement.tsv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(out["entry_suggestions"])
    print(json.dumps({"entries": len(mapped), "units": out["proposed_unit_counts"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
