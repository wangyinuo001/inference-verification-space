"""Place independently reviewed test properties in the candidate verification space.

This is a narrow, reviewable crosswalk. Every accepted entry already has a
source-backed property and assertion location in master-review-index.tsv.
Mixed files are matched by test class; unrelated entries stay unplaced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE = HERE.parent
OUTPUT = HERE / "confirmed-property-crosswalk.tsv"
KEY = ("snapshot", "path", "test_name", "line")
FIELDS = (*KEY, "unit_id", "property", "oracle_source", "conditions", "rule_id")


def target(row: dict[str, str]) -> tuple[str, str] | None:
    path, name = row["path"], row["test_name"]
    if path == "test/registered/unit/entrypoints/openai/test_serving_chat.py":
        if name in {
            "ServingChatTestCase::test_kimi_tool_call_keeps_default_reasoning",
            "ServingChatTestCase::test_kimi_tool_call_keeps_explicit_reasoning",
            "ServingChatTestCase::test_kimi_tool_call_respects_explicit_reasoning_disable",
            "ServingChatTestCase::test_jinja_rejects_non_object_tool_call_arguments",
            "ServingChatTestCase::test_jinja_accepts_object_tool_call_arguments_string",
            "ServingChatTestCase::test_dsv_encoders_reject_non_object_tool_call_arguments",
        } or name.startswith("TestNormalizeToolContent::"):
            return "request.admission", "chat_input_binding"
        if name in {
            "ServingChatTestCase::test_jinja_uses_openai_tool_schema_first",
            "ServingChatTestCase::test_jinja_tool_schema_fallback_to_flat_function",
            "ServingChatTestCase::test_xgrammar_tag_omits_reasoning_when_parser_owns_it",
        }:
            return "tool.constraint_schema_construction", "chat_tool_constraint"
        if name.startswith("ServingChatTestCase::test_unstreamed_tool_args_"):
            return "response.stream_payload_consistency", "streamed_tool_arguments"
        if name.startswith("TestProcessToolCallsWithRequiredToolChoice::"):
            return "tool.call_parse_relation", "required_tool_parse"
    if path == "test/registered/unit/entrypoints/openai/test_serving_responses.py":
        if name == "ChatToolForwardingTestCase::test_required_tool_choice_without_function_tool_returns_400" or name == "InputItemNormalizationTestCase::test_unknown_input_item_type_raises":
            return "input.error_response_classification", "responses_invalid_input"
        if name.startswith("InputItemNormalizationTestCase::") or name == "ChatToolForwardingTestCase::test_make_request_passes_function_tools_to_chat_processing":
            return "request.admission", "responses_input_normalization"
        if name.startswith("OutputItemsTestCase::"):
            return "tool.call_parse_relation", "responses_tool_output"
    if path == "test/registered/utils/test_network_address.py":
        if name.startswith("TestPortArgsIPv6::"):
            return ("deployment.configuration_admissibility" if "invalid" in name or "malformed" in name
                    else "deployment.effective_configuration"), "distributed_endpoint"
        if name.startswith("TestServerArgsIPv6Url::"):
            return "deployment.effective_configuration", "server_url"
    exact_files = {
        "tests/tool_parsers/test_utils.py": ("output.tool_argument_type_binding", "tool_schema_coercion"),
        "test/registered/unit/function_call/test_normalize_json_schema_types.py": ("tool.constraint_schema_construction", "schema_normalization"),
        "tests/tool_use/test_tool_choice_required.py": ("tool.constraint_schema_construction", "required_tool_schema"),
        "tests/parser/test_chat_parsing.py": ("output.tool_argument_type_binding", "chat_tool_coercion"),
        "tests/renderers/chat_utils/test_audio.py": ("input.multimodal_binding", "audio_binding"),
        "tests/entrypoints/serve/middleware/test_authentication_middleware.py": ("request.access_control", "http_auth"),
        "test/registered/unit/function_call/test_json_schema_constraint.py": ("tool.constraint_schema_construction", "sglang_tool_schema"),
    }
    if path in exact_files:
        return exact_files[path]
    if path == "test/registered/unit/utils/test_auth.py" and name.startswith("TestDecideRequestAuth::"):
        return "request.access_control", "auth_decision"
    if path == "tests/v1/core/test_deferred_block_free.py" and name in {
        "test_finish_defers_free_until_inflight_step_done",
        "test_finish_frees_immediately_when_no_inflight_step",
        "test_abort_defers_free",
        "test_preempt_defers_free_and_clears_bookkeeping",
        "test_allocation_retry_waits_for_fence_then_succeeds",
        "test_multiple_deferred_frees_drain_in_order",
        "test_fence_held_across_multiple_inflight_steps",
        "test_max_tokens_finish_frees_immediately_with_other_inflight",
        "test_abort_mid_prefill_defers_free",
        "test_non_async_abort_defers_via_last_sched_seq",
        "test_cow_retentions_deferred_until_copy_step_processed",
        "test_abort_deferred_async_load_frees_after_inflight_step",
    }:
        return "cache.lifetime", "inflight_cache_lifetime"
    if path == "test/registered/unit/function_call/test_hunyuan_detector.py":
        cls = name.split("::", 1)[0]
        if name == "TestHunyuanDetectorStreaming::test_streaming_typed_args_coerced":
            return "output.tool_argument_type_binding", "hunyuan_streaming_argument_types"
        if cls == "TestHunyuanDetectorArgDeserialization":
            return "output.tool_argument_type_binding", "hunyuan_argument_types"
        if cls == "TestHunyuanDetectorHasToolCall":
            return "output.structured_marker_boundary", "hunyuan_marker"
        if cls in {"TestHunyuanDetectorAccuracy", "TestHunyuanDetectorDetectAndParse", "TestHunyuanDetectorStreaming"}:
            return "tool.call_parse_relation", "hunyuan_parse"
    return None


def main() -> None:
    units = {unit["id"] for unit in json.loads((SPACE / "space_seed.json").read_text())["units"]}
    with (SPACE / "master-review-index.tsv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    accepted: list[dict[str, str]] = []
    for row in rows:
        if row["evidence_level"] != "source_property_confirmed":
            continue
        match = target(row)
        if not match:
            continue
        # A review has to name both the property and its actual assertion site.
        assert row["evidence_level"] == "source_property_confirmed", (row["path"], row["test_name"])
        assert row["agent_property_statement"] and row["agent_oracle_source"]
        unit_id, rule_id = match
        assert unit_id in units
        accepted.append({**{field: row[field] for field in KEY}, "unit_id": unit_id,
                         "property": row["agent_property_statement"],
                         "oracle_source": row["agent_oracle_source"],
                         "conditions": row["agent_conditions"], "rule_id": rule_id})
    keys = [tuple(row[field] for field in KEY) for row in accepted]
    assert len(keys) == len(set(keys))
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(accepted)
    print(json.dumps({"mapped": len(accepted), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
