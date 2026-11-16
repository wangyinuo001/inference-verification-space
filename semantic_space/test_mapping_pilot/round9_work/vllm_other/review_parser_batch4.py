"""Source-reviewed vLLM parser batch. This script never infers from a filename alone."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKETS = HERE / "source-packets.jsonl"
FILES = {
    "tests/parser/engine/test_minimax_m2.py": "vllm/parser/minimax_m2.py",
    "tests/parser/engine/test_granite.py": "vllm/parser/granite.py",
    "tests/parser/engine/test_glm47_moe.py": "vllm/parser/glm47_moe.py",
}


def verdict(path, name):
    cls, _, method = name.partition("::")
    if path.endswith("test_minimax_m2.py"):
        if cls in {"TestNonStreaming", "TestStreaming"}:
            if method == "test_schema_type_coercion":
                return ("output.tool_argument_type_binding", "Schema-declared integer and boolean arguments are converted to asserted values and types.")
            return ("tool.call_parse_relation", "Asserted tool-call presence, names, arguments, order or rejection under the fixed MiniMax format.")
        if cls == "TestArgValueExtender":
            if method == "test_long_call_streams_non_streaming_arguments":
                return ("tool.call_parse_relation", "Concatenated streamed arguments equal the non-streaming parse, with asserted content and schema-specific mode value.")
            if method == "test_conversions_do_not_grow_with_argument_length":
                return (None, "converter_call_count_is_a_cost_property")
            return (None, "internal_argument_extender_boundary_requires_new_unit")
        return (None, "reasoning_content_token_boundary_requires_new_unit")
    if path.endswith("test_granite.py"):
        if cls == "TestArgConverter":
            if method in {"test_complete_wrapper", "test_arguments_before_name", "test_missing_arguments_defaults"}:
                return ("tool.call_parse_relation", "Granite argument wrapper is converted to the asserted argument text or empty default; this is a local parser check.")
            return (None, "partial_argument_converter_or_rejection_requires_precise_unit")
        if cls == "TestNonStreaming":
            return ("tool.call_parse_relation", "Asserted content, tool names/order, arguments, or absence of calls under Granite marker/JSON-array format.")
        if cls == "TestStreaming":
            if method == "test_incomplete_array_at_eos":
                return ("tool.call_parse_relation", "Only asserts the tool marker does not leak into returned content at EOS.")
            return ("tool.call_parse_relation", "Asserted streamed tool names, arguments or retained surrounding text for the supplied chunk size.")
        if cls == "TestReplay":
            return ("tool.call_parse_relation", "Helper replays token-by-token and compares content plus tool names, count and arguments to the scenario's expected values.")
    if path.endswith("test_glm47_moe.py"):
        if cls in {"TestIsReasoningEnd", "TestExtractContentIds"}:
            return (None, "reasoning_content_token_boundary_requires_new_unit")
        if cls == "TestStreamedToolArguments":
            if method in {"test_long_call_streams_non_streaming_arguments", "test_repeated_closed_args_stream_non_streaming_arguments"}:
                return ("tool.call_parse_relation", "Streamed argument text is compared with non-streaming parsed tool-call arguments for the fixed GLM format.")
            if "conversions" in method or method == "test_closed_pairs_are_parsed_once":
                return (None, "converter_call_count_is_a_cost_property")
            return (None, "internal_argument_extender_boundary_requires_new_unit")
    return (None, "unreviewed_parser_assertion")


def main():
    packets = []
    for line in PACKETS.open():
        packet = json.loads(line)
        if packet["key"][1] in FILES:
            packets.append(packet)
    rows, pending = [], []
    reviews = {}
    for packet in packets:
        snapshot, path, name, line = packet["key"]
        unit, property_or_reason = verdict(path, name)
        checks = packet["evidence"]["checks"]
        calls = packet["evidence"]["calls"]
        check_refs = [f"{path}:{c['line']}" for c in checks]
        direct_call = next((c for c in calls if c["name"].startswith(("parser.", "no_thinking_parser.", "_granite_arg_converter", "_glm47_arg_value_extender", "_minimax_m2_arg_value_extender"))), None)
        conditions = f"test={name}; call={direct_call['line']}:{direct_call['source'][:180]}" if direct_call else f"test={name}; local/helper call inspected in source"
        if unit and checks:
            rows.append([snapshot, path, name, line, unit, property_or_reason, ";".join(check_refs[:4]), conditions, "vllm_parser_batch4_source_review_20261112"])
        elif unit and name == "TestReplay::test_replay_scenarios":
            rows.append([snapshot, path, name, line, unit, property_or_reason, "tests/parser/engine/trace_builder.py:292;tests/parser/engine/replay_harness.py:378;tests/parser/engine/replay_harness.py:384;tests/parser/engine/replay_harness.py:392;tests/parser/engine/replay_harness.py:408", conditions, "vllm_parser_batch4_helper_review_20261112"])
        else:
            pending.append([snapshot, path, name, line, property_or_reason if not unit else "assertion_not_located", ";".join(check_refs) or f"{path}:{line}"])
        reviews[path] = {"snapshot": snapshot, "path": path, "source_sha256": packet["sha256"], "review": "Read test function assertions, shared fixtures and parser/converter implementation; mapping is limited to the actual asserted parser property.", "product_path": FILES[path]}
    with (HERE / "crosswalk-batch4.tsv").open("w", newline="") as f:
        wr=csv.writer(f,delimiter="\t"); wr.writerow(["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]);wr.writerows(rows)
    with (HERE / "pending-batch4.tsv").open("w",newline="") as f:
        wr=csv.writer(f,delimiter="\t");wr.writerow(["snapshot","path","test_name","line","reason","evidence_source"]);wr.writerows(pending)
    (HERE / "evidence-batch4.json").write_text(json.dumps({"batch":"parser_engine_minimax_granite_glm47", "file_reviews":list(reviews.values()),"shared_helpers":["tests/parser/engine/streaming_helpers.py:20-75", "tests/parser/engine/trace_builder.py:281-292", "tests/parser/engine/replay_harness.py:368-410"],"counts":{"scope":len(packets),"mapped":len(rows),"pending":len(pending)}},ensure_ascii=False,indent=2)+"\n")
    (HERE / "needs-space-batch4.json").write_text(json.dumps({"candidates":[{"id":"output.reasoning_content_token_boundary","property":"Reasoning end and content-start token IDs are chosen for the current turn, including implicit tool-call boundary and prior-turn isolation.","evidence":["tests/parser/engine/test_glm47_moe.py:62-118","tests/parser/engine/test_minimax_m2.py:363-385","vllm/parser/engine/parser_engine.py:733-754"]},{"id":"output.tool_argument_fragment_boundary","property":"Partial streamed tool argument text is emitted or retained at syntax boundaries without consuming an incomplete end tag or next parameter.","evidence":["tests/parser/engine/test_minimax_m2.py:327-359","tests/parser/engine/test_glm47_moe.py:308-328","vllm/parser/minimax_m2.py:115"]},{"id":"output.tool_argument_conversion_cost","property":"Parser argument conversion calls remain bounded as an argument grows and closed argument pairs are parsed once.","evidence":["tests/parser/engine/test_minimax_m2.py:295-325","tests/parser/engine/test_glm47_moe.py:191-306","vllm/parser/engine/parser_engine.py:674"]}]},ensure_ascii=False,indent=2)+"\n")
    print(len(packets),len(rows),len(pending))


if __name__ == "__main__": main()
