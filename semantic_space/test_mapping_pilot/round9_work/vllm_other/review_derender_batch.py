"""Review two pinned vLLM derender test files by their actual assertions.

The named groups were curated after reading both complete source files and
OnlineDerenderer/ServingDerender. This script checks that the source at every
entry still contains a supporting assertion before emitting a crosswalk row.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {
    "tests/entrypoints/scale_out/derender/test_derender.py",
    "tests/entrypoints/scale_out/derender/test_derender_stream.py",
}
FIELDS = ["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]


def names(s): return set(s.split())

ERROR = names("""
test_derender_chat_empty_token_ids test_derender_chat_null_token_ids
test_derender_chat_unknown_model test_derender_chat_oversized_prompt_token_ids_rejected
test_derender_completion_prompt_tokens_length_mismatch
test_derender_completion_prompt_token_ids_length_mismatch
test_derender_completion_oversized_prompt_token_ids_rejected
test_derender_completion_empty_generate_responses
test_derender_chat_oversized_token_ids_rejected
test_derender_chat_too_many_choices_rejected
test_derender_completion_too_many_generate_responses_rejected
test_derender_chat_negative_token_ids_rejected
test_derender_chat_oversized_logprobs_rejected
test_derender_chat_oversized_top_logprobs_rejected
test_derender_completion_oversized_token_ids_rejected
test_e2e_no_chat_request_rejected
test_streaming_derender_invalid_body_returns_400
test_streaming_derender_non_object_body_returns_400
test_stream_parsed_missing_chat_request_rejected
test_stream_parsed_missing_prompt_token_ids_rejected
test_completion_stream_bad_state_returns_400
test_chat_stream_bad_state_returns_400
test_missing_chat_request_with_parser_rejected
test_chat_stop_strings_rejected test_completion_stop_strings_rejected
test_missing_prompt_token_ids_with_parser_rejected
test_oversized_output_token_ids_rejected
test_oversized_prompt_token_ids_rejected test_too_many_choices_rejected
""")
USAGE = names("""
test_derender_chat_usage test_derender_chat_usage_default
test_derender_completion_usage_aggregation test_prompt_tokens_in_usage
""")
LOGPROB_ID = names("""
test_derender_chat_logprobs test_derender_chat_top_logprobs
test_derender_chat_prompt_logprobs_passthrough
test_derender_completion_logprobs test_derender_completion_prompt_logprobs_passthrough
test_chat_stream_logprobs_resolved_per_chunk
test_top_logprobs_follow_the_chat_cut test_logprobs_kept_when_parser_swallows_chunk
test_include_reasoning_false_drops_logprobs_on_every_chunk
test_stream_without_logprobs_unchanged
""")
LOGPROB_TEXT = names("""
test_derender_chat_logprobs_bytes test_derender_chat_prompt_logprobs_decoded_token
test_chat_stream_logprobs_multibyte_across_chunks
test_logprobs_multibyte_across_chunks
""")
FINAL = names("""
test_derender_chat_passes_stop_reason_through
test_derender_completion_passes_stop_reason_through
test_derender_chat_finish_reason_matches_chat_completion
test_derender_batch_trims_stop_like_engine
test_derender_completion_stream_leaves_out_stop_token
test_finish_reason_forwarded test_stop_reason_forwarded
test_finish_reason_rewritten_to_tool_calls
test_finish_reason_stop_for_named_tool_choice
""")
STREAM = names("""
test_single_chunk test_two_equal_chunks test_single_token_per_chunk
test_multibyte_char_split_across_chunks test_state_carries_across_calls
test_n_independent_streams_same_result test_chat_stream_first_chunk_seeded
test_completion_stream_first_chunk_seeded
test_chunked_equals_oneshot test_chunking_invariance
test_streaming_completion_derender_roundtrip
test_streaming_chat_derender_roundtrip
test_stream_parsed_matches_batch_reasoning
test_stream_parsed_matches_batch_tool_call
test_stream_matches_batch
""")
PARSE = names("""
test_e2e_parsed_reasoning test_e2e_parsed_tool_call
test_e2e_parsed_reasoning_and_tool_call
test_e2e_harmony_tool_choice_none_drops_tool_calls
test_e2e_harmony_reasoning_default_parser test_e2e_harmony_reasoning
test_derender_chat_drops_tool_calls_without_tool_choice
test_derender_text_choices_builds_request_from_parse_context
test_derender_text_choices_drops_tool_calls_without_tool_choice
test_dispatches_and_emits_content test_parser_path_ephemeral_state_seeded
test_include_reasoning_false_suppresses_reasoning
test_reasoning_never_leaks_as_content_midstream
test_stream_parsed_include_reasoning_false
""")
DELIVERY = names("""
test_derender_chat_roundtrip test_derender_completion_roundtrip
test_derender_chat_returns_requested_prompt_token_ids
test_derender_chat_model_omitted_resolves_served_name
test_derender_chat_kv_transfer_params_passthrough
test_derender_completion_kv_transfer_params_passthrough
test_derender_completion_preserves_requested_token_ids
test_derender_chat_bounded_payload_succeeds
test_role_on_first_chunk_only test_role_sent_once
test_usage_chunk_passthrough test_metrics_passthrough
test_derender_chat_metrics_passthrough
""")

GROUPS = [
    ("input.error_response_classification", ERROR, ("400","404","ErrorResponse"), "Malformed or unsupported derender request produces the asserted HTTP/error class and field."),
    ("response.usage_accounting", USAGE, ("usage", "prompt_tokens"), "Reported prompt, output or total token usage equals the asserted count for the constructed response."),
    ("response.logprob_identity", LOGPROB_ID, ("logprob", "top_logprob"), "Returned token logprob entry or candidate list follows the asserted identity, count and placement relation."),
    ("response.logprob_text_decoding", LOGPROB_TEXT, ("decoded_token", "byte", "�"), "Decoded logprob token text/bytes follow the asserted decoder and chunk-boundary behavior."),
    ("output.finalization", FINAL, ("finish_reason", "stop_reason", "stop"), "Stop/finish reason and final text follow the asserted request and generated token boundary."),
    ("response.stream_payload_consistency", STREAM, ("one_shot", "streamed", "stream", "expected", "chunk"), "Ordered stream chunks reconstruct the asserted one-shot/batch text or structured content under the same token input."),
    ("output.incremental_parse_relation", PARSE, ("reasoning", "tool_calls", "content", "delta"), "Parsed reasoning, content or tool-call fields follow the asserted channel/choice relation for the supplied token stream."),
    ("response.delivery", DELIVERY, ("response", "choice", "role", "usage", "metrics", "model"), "Derendered response exposes the asserted field, role or choice at the API boundary."),
]


def assertions(e):
    stack=[e.get("evidence")]
    while stack:
        ev=stack.pop()
        if not ev: continue
        for c in ev.get("checks",[]): yield c
        stack.extend(ev.get("local_helpers",{}).values())


def main():
    rows=[];pending=[];files={};counts={}
    for line in (HERE/"source-packets.jsonl").open():
        p=json.loads(line); path=p["key"][1]
        if path not in FILES:continue
        files[path]=p["sha256"]
        name=p["key"][2].split("::")[-1]
        checks=list(assertions(p));checks_text=" ".join(x["source"] for x in checks)
        matches=[g for g in GROUPS if name in g[1]]
        if len(matches)!=1:
            pending.append([*p["key"],"no_unique_curated_property", "", ""]);continue
        unit,_,markers,property_text=matches[0]
        relevant=[c for c in checks if any(m in c["source"] for m in markers)]
        if not relevant:
            pending.append([*p["key"],"curated_property_lacks_expected_assertion",unit,checks_text[:300]]);continue
        # Exception and response-status tests are checked against their actual
        # error branch; a mere 200/success check is insufficient.
        if unit=="input.error_response_classification" and not any(m in c["source"] for c in relevant for m in ('== 400','== 404','code == 400')):
            pending.append([*p["key"],"missing_error_status_assertion",unit,checks_text[:300]]);continue
        rows.append({"snapshot":"vllm","path":path,"test_name":p["key"][2],"line":p["key"][3],"unit_id":unit,"property":property_text,"oracle_source":";".join(f"{path}:{c['line']}" for c in relevant)[:1200],"conditions":"Constructed token IDs, chunks, prompt IDs, model choice, parser, request options and response state as specified in this test; evidence is the local assertions rather than all behavior implied by its name.","rule_id":"vllm_derender_two_files_source_review_20261112"})
        counts[unit]=counts.get(unit,0)+1
    with (HERE/"crosswalk-batch2.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,delimiter="\t");w.writeheader();w.writerows(rows)
    with (HERE/"pending-batch2.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["snapshot","path","test_name","line","reason","candidate_unit","assertion_excerpt"]);w.writerows(pending)
    evidence={"batch":"derender_two_files_source_review","file_reviews":[{"snapshot":"vllm","path":p,"source_sha256":sha,"review":"Reviewed the source-grounded per-entry AST call and assertion packets, plus representative helper and implementation paths. Curated groups are emitted only when their actual assertion source supports the named local property; ungrouped or ambiguous cases stay pending. Broader full-function review remains to be done before treating this as final confirmation.","product_source":["vllm/entrypoints/scale_out/derender/serving.py:142","vllm/entrypoints/scale_out/derender/serving.py:294","vllm/renderers/online_derenderer.py:539","vllm/renderers/online_derenderer.py:1260"]} for p,sha in sorted(files.items())],"reviewed_entries":len(rows),"pending_entries":len(pending),"unit_counts":counts,"exceptions":[{"test_name":x[2],"reason":x[4]} for x in pending]}
    (HERE/"evidence-batch2.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    print(len(rows),len(pending),counts)


if __name__=="__main__": main()
