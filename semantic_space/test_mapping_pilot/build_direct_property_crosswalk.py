"""Rebuild a small, manually reviewed direct-test crosswalk from pinned source.

The case list records the property judged from each test body. AST is used only
to locate its actual assertions and to fail if the reviewed source moves.
"""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
REPOS = PROJECT / "experiments/third_party/reference_repos"
ROOTS = {"vllm": REPOS / "vllm-20261105", "sglang": REPOS / "sglang-v0.5.14-complete"}
OUTPUT = HERE / "direct-property-crosswalk.tsv"
FIELDS = ("snapshot", "path", "test_name", "line", "unit_id", "property", "oracle_source", "conditions", "rule_id")

# Each description was checked against the body and its assertion, not inferred
# from the test name or the static call-to-assertion index.
CONFIG = {
    "test_mamba_cache_mode_all_is_rejected": ("deployment.configuration_admissibility", "Mamba cache mode 'all' is rejected with a validation error."),
    "test_hisparse_connector_implies_attention_config": ("deployment.effective_configuration", "Selecting the HiSparse connector creates a HiSparse attention configuration."),
    "test_hisparse_connector_preserves_explicit_attention_config": ("deployment.effective_configuration", "An explicit HiSparse device buffer size survives connector configuration."),
    "test_no_hisparse_connector_keeps_attention_config_unset": ("deployment.effective_configuration", "Without the HiSparse connector, the HiSparse attention field stays unset."),
    "test_breakable_cudagraph_platform_default": ("deployment.effective_configuration", "The platform default determines breakable CUDA graph use and does not set its environment override."),
    "test_breakable_cudagraph_env_override": ("deployment.effective_configuration", "The environment override determines breakable CUDA graph use."),
    "test_models_default_to_v2_model_runner": ("deployment.effective_configuration", "The listed model configurations resolve to the expected v2 model runner default."),
    "test_with_hf_config_populates_missing_architectures_from_causal_lm_mapping": ("deployment.effective_configuration", "A missing architecture is inferred from the causal-LM mapping without mutating the original HF config."),
    "test_with_hf_config_preserves_explicit_architectures_override": ("deployment.effective_configuration", "An explicit architecture override survives HF configuration loading."),
    "test_with_hf_config_leaves_unknown_model_type_without_architectures": ("deployment.effective_configuration", "An unknown model type is not given an invented architecture."),
    "test_async_scheduling_with_pipeline_parallelism_is_allowed": ("deployment.effective_configuration", "The selected runner keeps asynchronous scheduling with pipeline parallelism."),
    "test_v1_model_runner_drops_async_scheduling_with_pipeline_parallelism": ("deployment.effective_configuration", "The v1 runner disables asynchronous scheduling with pipeline parallelism."),
    "test_data_parallel_rpc_port_has_fixed_default": ("deployment.effective_configuration", "The default data-parallel RPC port is 29550."),
    "test_all2all_backend_has_portable_default": ("deployment.effective_configuration", "The default all-to-all backend is allgather_reducescatter."),
    "test_engram_dp_shared_memory_requires_cpu_offload": ("deployment.configuration_admissibility", "Engram shared memory without CPU offload is rejected."),
    "test_engram_thp_rejects_explicit_shared_memory": ("deployment.configuration_admissibility", "Engram THP with explicit shared memory is rejected."),
    "test_engram_cpu_offload_default": ("deployment.effective_configuration", "Engram CPU offload defaults to enabled."),
    "test_nnodes_within_dp_when_replicas_outnumber_nodes": ("deployment.effective_configuration", "The listed DP and node counts derive one node per DP group, node rank zero and the expected local world size."),
    "test_draft_model_enables_async_scheduling_by_default": ("deployment.effective_configuration", "A draft model enables asynchronous scheduling by default."),
    "test_max_num_new_slots_for_drafting": ("deployment.effective_configuration", "Draft configuration computes the expected number of new slots for the listed parameters."),
    "test_auto_runner": ("deployment.effective_configuration", "Automatic runner selection yields the expected runner and conversion type."),
    "test_pooling_runner": ("deployment.effective_configuration", "Pooling runner selection yields the expected runner and conversion type."),
    "test_draft_runner": ("deployment.effective_configuration", "Draft runner selection yields the expected runner and conversion type."),
    "test_disable_sliding_window": ("deployment.effective_configuration", "Disabling the sliding window yields the expected maximum model length."),
    "test_get_pooling_config": ("deployment.effective_configuration", "Pooling configuration enables activation and derives MEAN sequence and ALL token pooling."),
    "test_get_pooling_config_supports_sentence_transformers_schemas": ("deployment.effective_configuration", "The listed Sentence Transformers schemas produce the expected pooling config without an unknown-mode warning."),
    "test_get_pooling_config_warns_when_pooling_mode_is_unknown": ("deployment.effective_configuration", "Unknown pooling mode falls back to activation disabled and emits a warning."),
    "test_get_dense_modules_supports_sentence_transformers_schemas": ("deployment.effective_configuration", "The listed Sentence Transformers schema resolves the expected dense module folder."),
    "test_default_seq_pooling_type": ("deployment.effective_configuration", "The model default and effective sequence pooling types match the listed expectations."),
    "test_default_tok_pooling_type": ("deployment.effective_configuration", "The model default and effective token pooling types match the listed expectations."),
    "test_moe_model_detection": ("deployment.effective_configuration", "The listed model configurations receive the expected MoE classification."),
    "test_is_quantized": ("deployment.effective_configuration", "The listed configurations receive the expected quantized classification."),
    "test_model_class_overrides_registers_target": ("deployment.effective_configuration", "A model-class override registers the stated Llama module and class in the configuration registry."),
    "test_is_encoder_decoder": ("deployment.effective_configuration", "The listed model configurations receive the expected encoder-decoder classification."),
    "test_uses_mrope": ("deployment.effective_configuration", "The listed model configurations receive the expected M-RoPE classification."),
    "test_load_config_pt_load_map_location": ("deployment.effective_configuration", "The load configuration preserves the requested PyTorch map location."),
    "test_is_chunked_prefill_supported": ("deployment.configuration_admissibility", "The listed attention types produce the expected chunked-prefill support decision and reason."),
    "test_is_prefix_caching_supported": ("deployment.configuration_admissibility", "The listed attention types produce the expected prefix-caching support decision and reason."),
    "test_is_custom_op_enabled": ("deployment.effective_configuration", "The custom-op configuration gives the expected fused-layernorm enablement."),
    "test_scheduler_config_rejects_zero_max_num_scheduled_tokens": ("deployment.configuration_admissibility", "Zero maximum scheduled tokens is rejected during configuration validation."),
    "test_needs_dp_coordination": ("deployment.effective_configuration", "The listed parallel configurations yield the expected DP-coordinator requirement."),
    "test_fault_tolerance_requires_single_api_server": ("deployment.configuration_admissibility", "Fault tolerance with multiple API servers is rejected."),
    "test_draft_sample_method_probabilistic_is_accepted": ("deployment.configuration_admissibility", "Probabilistic draft sampling is accepted as the effective method."),
    "test_draft_sample_method_gumbel_is_rejected": ("deployment.configuration_admissibility", "Unsupported Gumbel draft sampling is rejected."),
    "test_load_config_rejects_invalid_safetensors_load_strategy": ("deployment.configuration_admissibility", "An invalid safetensors loading strategy is rejected."),
}

EVICTION = {name: ("cache.eviction_policy_configuration", f"Policy name '{name[5:]}' selects its corresponding eviction strategy class.")
            for name in ("test_lru", "test_lfu", "test_fifo", "test_mru", "test_filo", "test_priority", "test_slru")}
EVICTION["test_unknown_policy_raises_valueerror"] = ("cache.eviction_policy_configuration", "An unknown policy raises ValueError and lists the available policy names.")

ENV = {
    "test_default_value_returned_when_env_not_set": "Absent scalar environment setting returns its declared string default.",
    "test_none_default_returned_when_env_not_set": "Absent scalar environment setting can return a None default.",
    "test_valid_value_returned_case_sensitive": "A listed scalar option is accepted with case-sensitive matching.",
    "test_valid_lowercase_value_returned_case_insensitive": "Case-insensitive scalar matching accepts the lowercase environment value.",
    "test_valid_uppercase_value_returned_case_insensitive": "Case-insensitive scalar matching accepts the uppercase environment value.",
    "test_callable_choices_resolved_correctly": "A callable list of allowed options is resolved before parsing the environment value.",
    "test_default_list_returned_when_env_not_set": "An absent environment setting returns the declared collection default.",
    "test_empty_default_list_returned_when_env_not_set": "An absent environment setting returns an empty collection default.",
    "test_single_valid_value_parsed_correctly": "A single allowed environment value produces a one-element collection.",
    "test_multiple_valid_values_parsed_correctly": "Comma-separated allowed environment values produce the expected collection.",
    "test_values_with_whitespace_trimmed": "Whitespace around comma-separated values is removed.",
    "test_empty_values_filtered_out": "Empty segments within a comma-separated environment value are removed.",
    "test_empty_string_returns_default": "An empty environment value returns the declared collection default.",
    "test_only_commas_returns_default": "An environment value containing only commas returns the declared collection default.",
    "test_case_insensitive_validation": "Case-insensitive collection validation accepts mixed-case listed options.",
    "test_duplicate_values_preserved": "List parsing preserves duplicate allowed environment values.",
    "test_duplicate_values_deduped": "Set parsing removes duplicate allowed environment values.",
}

RESPONSE_HARMONY = {
    "test_user_message_string_content": "A user string becomes a user Harmony message with the same text and no channel.",
    "test_no_type_key_defaults_to_message_branch": "A missing item type defaults to an ordinary user message.",
    "test_system_message": "System input becomes developer instructions with the same text.",
    "test_assistant_message_gets_final_channel": "Assistant input becomes a final-channel message with the same text.",
    "test_developer_message_gets_instructions_prefix": "Developer input becomes developer instructions with the same text.",
    "test_message_with_array_content": "An array of user text parts keeps both parts and their order.",
    "test_developer_message_array_content_concatenated": "Developer text parts are concatenated into one instruction string.",
    "test_reasoning_gets_analysis_channel": "Reasoning input becomes an assistant analysis-channel message.",
    "test_reasoning_pydantic_model_input": "A Pydantic reasoning item becomes an assistant analysis-channel message with its text.",
    "test_function_call_channel_recipient_and_content_type": "Function-call input keeps its name and JSON arguments in an assistant commentary message.",
    "test_function_call_empty_arguments": "A function call with empty arguments keeps an empty argument string.",
    "test_function_call_output_channel_recipient_and_author_name": "A matched tool output binds call ID to tool name and preserves output text.",
    "test_function_call_output_uses_most_recent_matching_call": "Repeated call IDs bind a tool output to the most recent matching function call.",
    "test_function_call_output_skips_non_function_call_items_in_prev_responses": "Reasoning items in history are skipped when matching a tool output's call ID.",
}

HARMONY = {
    "TestCreateToolDefinition::test_chat_completion_omitted_description_defaults_to_empty_string": ("tool.constraint_schema_construction", "A Chat tool with omitted description yields the same name/schema and an empty description."),
    "TestCreateToolDefinition::test_chat_completion_none_description_defaults_to_empty_string": ("tool.constraint_schema_construction", "A Chat tool with None description yields the same name/schema and an empty description."),
    "TestCreateToolDefinition::test_response_tool_none_description_defaults_to_empty_string": ("tool.constraint_schema_construction", "A Responses tool with None description yields the same name/schema and an empty description."),
    "TestParseChatInputToHarmonyMessage::test_assistant_message_with_empty_content": ("request.admission", "An assistant history item with empty string content yields no Harmony messages."),
    "TestParseChatInputToHarmonyMessage::test_assistant_message_with_none_content": ("request.admission", "An assistant history item with None content yields no Harmony messages."),
    "TestAutoDropAnalysisMessages::test_no_analysis_messages": ("request.admission", "A history without analysis messages remains unchanged."),
    "TestAutoDropAnalysisMessages::test_only_analysis_message": ("request.admission", "A history containing only an analysis message remains unchanged."),
    "TestAutoDropAnalysisMessages::test_multiple_analysis_messages_without_final_message": ("request.admission", "Analysis-only history remains unchanged without a final message."),
    "TestAutoDropAnalysisMessages::test_only_final_message": ("request.admission", "A final-only history remains unchanged."),
    "TestAutoDropAnalysisMessages::test_drops_one_analysis_messages_before_final_message": ("request.admission", "An analysis message before final output is dropped while later analysis is retained."),
    "TestAutoDropAnalysisMessages::test_drops_all_analysis_messages_before_final_message": ("request.admission", "All analysis messages preceding the first final message are removed."),
    "TestAutoDropAnalysisMessages::test_multiple_analysis_messages_with_multiple_final_messages": ("request.admission", "A multi-final history retains the two final messages and drops analysis messages."),
    "TestAutoDropAnalysisMessages::test_drops_non_assistant_analysis_messages": ("request.admission", "An analysis-channel tool message before final output is removed."),
    "TestResponseInputToHarmonyReasoningItem::test_reasoning_with_single_content": ("request.admission", "One reasoning text part becomes assistant analysis text."),
    "TestResponseInputToHarmonyReasoningItem::test_reasoning_with_multiple_content_items": ("request.admission", "Multiple reasoning text parts are joined in order with newlines on the analysis channel."),
    "TestResponseInputToHarmonyReasoningItem::test_reasoning_without_content_returns_empty_message": ("request.admission", "Reasoning input without content yields an empty message, not the summary text."),
    "TestResponseInputToHarmonyReasoningItem::test_reasoning_with_none_content_returns_empty_message": ("request.admission", "Reasoning input with None content yields an empty message."),
    "TestResponseInputToHarmonyReasoningItem::test_reasoning_with_empty_content_returns_empty_message": ("request.admission", "Reasoning input with an empty content array yields an empty message."),
}

GEMMA_ARGS = {
    "test_empty_string": "An empty argument string produces an empty argument object.",
    "test_whitespace_only": "Whitespace-only arguments produce an empty argument object.",
    "test_single_string_value": "One delimited string argument is parsed with its key and text.",
    "test_string_value_with_comma": "A comma inside a delimited string stays inside the argument value.",
    "test_multiple_string_values": "Two delimited string arguments retain their separate keys and values.",
    "test_integer_value": "An integer-looking argument is retained as string '42' at this parser layer.",
    "test_float_value": "A float-looking argument is retained as string '3.14' at this parser layer.",
    "test_boolean_true": "A true-looking argument is retained as string 'true' at this parser layer.",
    "test_boolean_false": "A false-looking argument is retained as string 'false' at this parser layer.",
    "test_null_value": "A null-looking argument is retained as string 'null' at this parser layer.",
    "test_mixed_types": "Mixed Gemma4 argument forms produce the expected four string values.",
    "test_nested_object": "Nested object syntax produces a nested argument dictionary.",
    "test_array_of_strings": "Array syntax produces an ordered two-string argument list.",
    "test_unterminated_string": "An unterminated delimited string consumes the remaining text as its value.",
    "test_empty_value": "A key followed by a colon and no value gets an empty string in complete mode.",
    "test_empty_value_after_other_keys_partial_withheld": "Partial mode keeps complete earlier arguments but withholds a trailing empty value.",
    "test_malformed_partial_array": "A malformed array returns a dictionary within the test timeout.",
    "test_delimited_keys_stripped": "Delimiter-wrapped keys are stripped at top level and inside nested objects.",
    "test_empty_value_partial_withheld": "A missing argument value is withheld in partial parse mode.",
    "test_trailing_dot_float_partial_withheld": "Partial parsing withholds a trailing-dot float and retains earlier stable fields; complete mode preserves its string form.",
}
GEMMA_ARRAY = {
    "test_string_array": "Delimited string array elements keep their order.",
    "test_empty_array": "An empty array body produces an empty list.",
    "test_bare_values": "Bare numeric and boolean-looking array elements remain strings.",
    "test_string_element_with_closing_bracket": "A closing bracket inside a delimited string is retained within a nested array element.",
    "test_stray_closing_bracket": "A stray closing bracket stops array parsing after the valid leading value.",
    "test_trailing_dot_float_partial_withheld": "Partial array parsing withholds a trailing-dot float while retaining earlier stable elements.",
}

GEMMA_EXTRACT = {
    "test_no_tool_calls": "Plain text stays content with no reported tool calls.",
    "test_single_tool_call": "One Gemma4 tool block yields the expected tool name and JSON arguments.",
    "test_multiple_arguments": "Two Gemma4 arguments retain their keys and values in one tool call.",
    "test_text_before_tool_call": "Leading prose is separated from a subsequent tool call.",
    "test_multiple_tool_calls": "Two sequential Gemma4 calls retain their order and names.",
    "test_nested_arguments": "Nested object and list arguments retain their structure in a tool call.",
    "test_tool_call_with_number_and_boolean": "Schema-based tool extraction converts boolean, integer and number fields to their JSON types.",
    "test_incomplete_tool_call": "An unterminated Gemma4 tool block is still extracted with its parsed argument text.",
    "test_hyphenated_function_name": "A hyphenated function name is preserved.",
    "test_dotted_function_name": "A dotted function name is preserved.",
    "test_no_arguments": "A tool call with empty braces yields an empty argument object.",
}
GEMMA_STREAM = {
    "test_basic_streaming_single_tool": "Chunked Gemma4 output reconstructs one tool name and valid JSON with a comma in a string value.",
    "test_streaming_multi_arg": "Chunked Gemma4 output reconstructs two tool arguments.",
    "test_streaming_no_extra_brace": "Streaming arguments remain valid JSON without an extra closing brace.",
    "test_streaming_no_unquoted_keys": "Streaming arguments start with an object and quote the argument key.",
    "test_streaming_name_no_call_prefix": "The streamed function name excludes the Gemma4 call prefix.",
    "test_streaming_text_before_tool_call": "Prose before a tool call remains ordinary streamed content.",
    "test_streaming_numeric_args": "Chunked numeric and boolean arguments reconstruct as JSON number and boolean.",
    "test_streaming_boolean_split_across_chunks": "A boolean true split across chunks remains true in reconstructed JSON.",
    "test_streaming_false_split_across_chunks": "A boolean false split across chunks remains false in reconstructed JSON.",
    "test_streaming_number_split_across_chunks": "A number split across chunks remains the integer 42.",
    "test_streaming_empty_args": "A streamed empty-argument tool call retains its function name.",
    "test_streaming_split_delimiter_no_invalid_json": "A delimiter split across chunks does not leak raw delimiter text into JSON arguments.",
    "test_streaming_does_not_duplicate_plain_text_after_tool_call": "Prose following a tool call is emitted exactly once.",
    "test_streaming_html_argument_does_not_duplicate_tag_prefixes": "HTML argument text split at tag prefixes reconstructs without duplicated characters.",
    "test_streaming_single_chunk_complete_tool_call": "One chunk holding a complete tool call emits one indexed call, not prose.",
    "test_streaming_multi_chunk_batched_tool_calls": "One chunk holding two complete calls emits distinct indices with both names and argument objects.",
    "test_streaming_trailing_bare_bool_not_duplicated": "A trailing bare boolean appears once in reconstructed tool arguments.",
}
GEMMA_BARE = {
    "test_bare_opener_non_streaming": "A bare colon opener yields the expected name and argument object.",
    "test_documented_form_still_works": "The documented call-prefix form still yields the expected tool.",
    "test_bare_opener_multiple_args": "A bare opener preserves two argument fields.",
    "test_malformed_no_brace_bounded": "A malformed call without braces does not absorb the end marker into its name.",
    "test_malformed_then_well_formed": "A malformed call followed by a valid call yields both calls in order.",
    "test_bare_malformed_then_well_formed": "A malformed bare opener followed by a valid call yields both names in order.",
    "test_three_mixed_calls": "Three mixed bare and documented calls retain count and order.",
    "test_colon_in_content_not_broken": "A colon in leading prose remains ordinary content before the tool call.",
    "test_bare_opener_empty_args": "A bare opener with empty braces retains its function name.",
}

SAMPLING_BATCH = {
    "TestMergeBiasTensor::test_both_none_returns_none": ("sampling.per_request_modifier_binding", "Merging two absent logit-bias tensors keeps the modifier absent."),
    "TestMergeBiasTensor::test_both_present_concatenates": ("sampling.per_request_modifier_binding", "Merging two bias tensors preserves their values in separate batch rows."),
    "TestMergeBiasTensor::test_lhs_none_fills_default": ("sampling.per_request_modifier_binding", "Absent left-hand bias fills only its rows with the default value."),
    "TestMergeBiasTensor::test_rhs_none_fills_default": ("sampling.per_request_modifier_binding", "Absent right-hand bias fills only its rows with the default value."),
    "TestMergeBiasTensor::test_custom_default_value": ("sampling.per_request_modifier_binding", "A custom missing-row fill value applies only to the missing left-hand rows."),
    "TestMergeCustomLogitProcessor::test_both_none_returns_none": ("sampling.per_request_modifier_binding", "Merging two absent custom processors keeps the modifier absent."),
    "TestMergeCustomLogitProcessor::test_same_key_merges_masks": ("sampling.per_request_modifier_binding", "A shared custom processor keeps each source batch's boolean row mask."),
    "TestMergeCustomLogitProcessor::test_disjoint_keys": ("sampling.per_request_modifier_binding", "Disjoint processors pad absent request rows with false masks."),
    "TestMergeCustomLogitProcessor::test_lhs_none_rhs_present": ("sampling.per_request_modifier_binding", "A right-only processor gets an aligned mask after padding missing left rows."),
    "TestFromScheduleBatch::test_basic_construction": ("request.effective_sampling_parameters", "Scheduled request temperature, top-p and top-k enter the corresponding batch fields."),
    "TestFromScheduleBatch::test_greedy_detection": ("request.effective_sampling_parameters", "A scheduled request with top-k one marks the batch greedy."),
    "TestFromScheduleBatch::test_logit_bias_construction": ("sampling.per_request_modifier_binding", "Request logit-bias entries appear at their token columns, with other columns zero."),
    "TestFromScheduleBatch::test_deterministic_seed": ("request.effective_sampling_parameters", "Explicit seed 123 and absent-seed default 42 occupy their respective batch rows."),
    "TestFromScheduleBatch::test_from_schedule_batch_sampling_flags": ("request.effective_sampling_parameters", "The listed top-p, top-k and min-p values enable their batch sampling flags without greedy mode."),
    "TestFromScheduleBatch::test_no_logit_bias_when_all_none": ("sampling.per_request_modifier_binding", "A batch with no request logit bias keeps its bias tensor absent."),
    "TestFromScheduleBatch::test_custom_logit_processor_merging": ("sampling.per_request_modifier_binding", "Only the request declaring a custom logit processor enters the merged processor mask."),
}

CONTINUE_FINAL = {
    "test_string_input_returns_false": "A string input does not continue a final message.",
    "test_empty_list_returns_false": "Empty input does not continue a final message.",
    "test_completed_message_returns_false": "A completed assistant message does not continue.",
    "test_in_progress_message_returns_true": "An in-progress assistant message requests continuation.",
    "test_incomplete_message_returns_true": "An incomplete assistant message requests continuation.",
    "test_in_progress_reasoning_returns_true": "An in-progress reasoning item requests continuation.",
    "test_completed_reasoning_returns_false": "A completed reasoning item does not continue.",
    "test_reasoning_with_none_status_returns_false": "A reasoning item without status does not continue.",
    "test_dict_in_progress_message_returns_true": "Dictionary-form in-progress assistant input requests continuation.",
    "test_dict_incomplete_message_returns_true": "Dictionary-form incomplete assistant input requests continuation.",
    "test_dict_completed_message_returns_false": "Dictionary-form completed assistant input does not continue.",
    "test_dict_reasoning_in_progress_returns_true": "Dictionary-form in-progress reasoning input requests continuation.",
    "test_dict_without_status_returns_false": "Dictionary-form assistant input without status does not continue.",
    "test_dict_with_none_status_returns_false": "Dictionary-form assistant input with None status does not continue.",
}
RESPONSES_UTILS = {
    **{f"TestShouldContinueFinalMessage::{name}": prop for name, prop in CONTINUE_FINAL.items()},
    "TestConstructChatMessagesCombinePolicy::test_assistant_side_items_merge_until_tool_output": "The listed contiguous assistant-side items merge with the expected content, reasoning and tool-call IDs.",
    "TestConstructChatMessagesCombinePolicy::test_merge_chain_breaks": "Output messages, reasoning messages and tool outputs break assistant-side merge chains at the listed boundaries.",
    "TestConstructInputMessagesInstructionsLeak::test_non_system_messages_preserved": "New instructions replace prior system text while earlier user and assistant messages remain ordered.",
    "TestConstructInputMessagesInstructionsLeak::test_no_prev_msg": "Without prior messages, the new instructions and user input become the two input messages.",
}

CASES = {
    ("vllm", "tests/test_config.py"): {name: (*value, "configuration") for name, value in CONFIG.items()},
    ("sglang", "test/registered/unit/mem_cache/test_mem_cache_utils.py"): {
        **{f"TestGetEvictionStrategy::{name}": (*value, "eviction_policy") for name, value in EVICTION.items()},
    },
    ("vllm", "tests/test_envs.py"): {
        **{f"{cls}::{name}": ("deployment.effective_configuration", ("List: " if cls == "TestEnvListWithChoices" else "Set: " if cls == "TestEnvSetWithChoices" else "Scalar: ") + prop, "environment_choice")
           for cls, names in {
               "TestEnvWithChoices": ("test_default_value_returned_when_env_not_set", "test_none_default_returned_when_env_not_set", "test_valid_value_returned_case_sensitive", "test_valid_lowercase_value_returned_case_insensitive", "test_valid_uppercase_value_returned_case_insensitive", "test_callable_choices_resolved_correctly"),
               "TestEnvListWithChoices": ("test_default_list_returned_when_env_not_set", "test_empty_default_list_returned_when_env_not_set", "test_single_valid_value_parsed_correctly", "test_multiple_valid_values_parsed_correctly", "test_values_with_whitespace_trimmed", "test_empty_values_filtered_out", "test_empty_string_returns_default", "test_only_commas_returns_default", "test_case_insensitive_validation", "test_callable_choices_resolved_correctly", "test_duplicate_values_preserved"),
               "TestEnvSetWithChoices": ("test_default_list_returned_when_env_not_set", "test_empty_default_list_returned_when_env_not_set", "test_single_valid_value_parsed_correctly", "test_multiple_valid_values_parsed_correctly", "test_values_with_whitespace_trimmed", "test_empty_values_filtered_out", "test_empty_string_returns_default", "test_only_commas_returns_default", "test_case_insensitive_validation", "test_callable_choices_resolved_correctly", "test_duplicate_values_deduped"),
           }.items() for name in names for prop in (ENV[name],)},
    },
    ("vllm", "tests/entrypoints/openai/responses/test_response_input_to_harmony.py"): {
        f"TestResponseInputToHarmonyMessage::{name}": ("request.admission", prop, "responses_harmony_input")
        for name, prop in RESPONSE_HARMONY.items()
    },
    ("vllm", "tests/entrypoints/openai/parser/test_harmony_utils.py"): {
        name: (*value, "harmony_input_or_schema") for name, value in HARMONY.items()
    },
    ("vllm", "tests/tool_parsers/test_gemma4_tool_parser.py"): {
        **{f"TestParseGemma4Args::{name}": ("tool.call_parse_relation", prop, "gemma4_argument_parse")
           for name, prop in GEMMA_ARGS.items()},
        **{f"TestParseGemma4Array::{name}": ("tool.call_parse_relation", prop, "gemma4_array_parse")
           for name, prop in GEMMA_ARRAY.items()},
        **{f"TestExtractToolCalls::{name}": ("tool.call_parse_relation", prop, "gemma4_full_parse")
           for name, prop in GEMMA_EXTRACT.items()},
        **{f"TestStreamingExtraction::{name}": ("tool.call_parse_relation", prop, "gemma4_stream_parse")
           for name, prop in GEMMA_STREAM.items()},
        **{f"TestBareAndMalformedOpeners::{name}": ("tool.call_parse_relation", prop, "gemma4_malformed_parse")
           for name, prop in GEMMA_BARE.items()},
        "test_named_custom_tool_choice_skips_structured_outputs": (
            "tool.constraint_schema_construction",
            "A named custom-tool choice keeps special tokens and does not install generic structured output constraints.",
            "custom_tool_choice",
        ),
    },
    ("sglang", "test/registered/unit/sampling/test_sampling_batch_info.py"): {
        name: (*value, "sampling_batch_binding") for name, value in SAMPLING_BATCH.items()
    },
    ("vllm", "tests/entrypoints/openai/responses/test_responses_utils.py"): {
        name: ("request.admission", prop, "responses_history_binding")
        for name, prop in RESPONSES_UTILS.items()
    },
}

CONDITIONS = {
    "configuration": "Pinned vLLM snapshot; parametrized model, platform, environment and config inputs are fixed in the linked test definition. Local config result only; no server start.",
    "eviction_policy": "Pinned SGLang snapshot; strategy factory called with the policy name in the linked test. Class selection only; no runtime eviction sequence.",
    "environment_choice": "Pinned vLLM snapshot; patched TEST_ENV or absent environment variable and declared allowed choices in the linked test. Parsing only; no service start.",
    "responses_harmony_input": "Pinned vLLM snapshot; Response item and prior-call fixtures in the linked test. Local Harmony input conversion only; no scheduler or generation.",
    "harmony_input_or_schema": "Pinned vLLM snapshot; constructed Harmony/Chat input or tool definition in the linked test. Local conversion only; no tokenizer execution.",
    "gemma4_argument_parse": "Pinned vLLM snapshot; Gemma4 argument text and partial flag in the linked test. Parser output only; no model generation.",
    "gemma4_array_parse": "Pinned vLLM snapshot; Gemma4 array text and partial flag in the linked test. Parser output only; no model generation.",
    "gemma4_full_parse": "Pinned vLLM snapshot; mocked request/tokenizer and complete Gemma4 tool text in the linked test. No generation.",
    "gemma4_stream_parse": "Pinned vLLM snapshot; ordered chunks, mocked tokenizer and request in the linked test. Checks emitted deltas; no server stream.",
    "gemma4_malformed_parse": "Pinned vLLM snapshot; malformed or bare tool opener text in the linked test. Parser output only.",
    "custom_tool_choice": "Pinned vLLM snapshot; named Responses custom-tool choice and mocked tokenizer in the linked test. Request adjustment only.",
    "sampling_batch_binding": "Pinned SGLang snapshot; CPU tensors or mocked scheduled requests in the linked test. Checks batch construction, not sampled tokens.",
    "responses_history_binding": "Pinned vLLM snapshot; listed Response item statuses or previous-message chain in the linked test. Input helper result only.",
}


def test_nodes(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    found = {}
    for outer in tree.body:
        if isinstance(outer, ast.ClassDef):
            for node in outer.body:
                if isinstance(node, ast.FunctionDef):
                    found[f"{outer.name}::{node.name}"] = node
        elif isinstance(outer, ast.FunctionDef):
            found[outer.name] = outer
    return found


def oracle_lines(node: ast.FunctionDef) -> list[int]:
    sites = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            sites.append(child.lineno)
        elif isinstance(child, ast.With) and any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "raises"
            for item in child.items
        ):
            sites.append(child.lineno)
        elif (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
              and child.func.attr.startswith("assert")):
            sites.append(child.lineno)
    return sorted(set(sites))


def main() -> dict[str, int]:
    with (HERE / "stage-placement-all.tsv").open(encoding="utf-8", newline="") as f:
        index = {(r["snapshot"], r["path"], r["test_name"]): r for r in csv.DictReader(f, delimiter="\t")}
    units = {u["id"] for u in json.loads((HERE.parent / "space_seed.json").read_text())["units"]}
    rows = []
    for (snapshot, path), cases in CASES.items():
        source = ROOTS[snapshot] / path
        nodes = test_nodes(ast.parse(source.read_text(encoding="utf-8")))
        for name, (unit, prop, rule) in cases.items():
            node = nodes[name]
            indexed = index[(snapshot, path, name)]
            assert indexed["line"] == str(node.lineno), (path, name, indexed["line"], node.lineno)
            # Manual property review does not require the static linker to
            # identify the same relation; the source test and oracle are the evidence.
            assert unit in units
            sites = oracle_lines(node)
            assert sites, (path, name)
            rows.append({
                "snapshot": snapshot, "path": path, "test_name": name, "line": str(node.lineno),
                "unit_id": unit, "property": prop,
                "oracle_source": ";".join(f"{path}:{line}" for line in sites),
                "conditions": CONDITIONS[rule],
                "rule_id": rule,
            })
    keys = [(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in rows]
    assert len(keys) == len(set(keys))
    rows.sort(key=lambda r: (r["snapshot"], r["path"], int(r["line"])))
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    result = {"reviewed_direct": len(rows)}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    main()
