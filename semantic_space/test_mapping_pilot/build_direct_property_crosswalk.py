"""Rebuild a small, manually reviewed direct-test crosswalk from pinned source.

The case list records the property judged from each test body. AST is used only
to locate its actual assertions and to fail if the reviewed source moves.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

from build_core_direct_cases import PATH_HASHES as CORE_SOURCE_HASHES, reviewed_cases as core_reviewed_cases
from build_round7_direct_cases import PATH_HASHES as ROUND7_SOURCE_HASHES, reviewed_cases as round7_reviewed_cases

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
ENV_OTHER = {
    "test_getattr_without_cache": ("deployment.effective_configuration", "Without the env cache, host and port settings reflect changed environment values."),
    "test_p2p_side_channel_defaults_and_override": ("deployment.effective_configuration", "P2P host and port use declared defaults and accept environment overrides."),
    "test_p2p_resource_limit_defaults_and_override": ("deployment.effective_configuration", "P2P peer and timeout limits use declared defaults and accept overrides."),
    "test_getattr_with_cache": ("deployment.effective_configuration", "Enabling the env cache decorates lookup and records hits for host and port reads."),
    "test_getattr_with_reset": ("deployment.effective_configuration", "Disabling the env cache restores visibility of changed host environment values."),
    "test_is_envs_cache_enabled": ("deployment.effective_configuration", "Repeated env-cache enables are cleared by one disable call."),
    "test_precompiled_install_flags_are_orthogonal": ("deployment.runtime_mode_binding", "Rust-frontend and C-extension precompiled flags are parsed independently."),
    "test_rust_bench_auto_path_missing_fails_fast": ("deployment.configuration_admissibility", "Requesting the Rust bench frontend without its binary raises FileNotFoundError."),
    "test_gdn_decode_kernel_env": ("deployment.effective_configuration", "GDN decode kernel defaults to CUDA, accepts CUDA/Triton and rejects unsupported values."),
}
ENV_LOGGING = {
    "test_configure_logging_defaults_to_true": "Logging configuration defaults to boolean true when unset.",
    "test_configure_logging_with_zero_string": "Logging configuration parses environment string '0' as boolean false.",
    "test_configure_logging_with_one_string": "Logging configuration parses environment string '1' as boolean true.",
    "test_configure_logging_with_invalid_value_raises_error": "Logging configuration rejects a nonnumeric environment string.",
}
ENV_MAX_N = {
    "test_default_value": "The effective maximum number of sequences defaults to 16384, as asserted by the source test.",
    "test_custom_value": "The effective maximum number of sequences accepts environment override 128.",
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

REASONING_KEY = ("sglang", "test/registered/unit/parser/test_reasoning_parser.py")
REASONING_SHA256 = "56b4f38839ef1b6c8cbe19ec6fcb733f926e56abb383b2e79a4a5f4462be53ef"
# We reviewed each class's test assertions. The excluded entries only check
# result-container initialization, detector constructor fields, or a marker
# insertion path that does not establish the selected channel-parse relation.
REASONING_EXCLUDE = {
    "TestStreamingParseResult": {"test_init_default", "test_init_with_values"},
    "TestBaseReasoningFormatDetector": {"test_init"},
    "TestDeepSeekR1Detector": {"test_init", "test_init_no_stream_reasoning"},
    "TestQwen3Detector": {"test_init"},
    "TestQwen3ForcedReasoningDetector": {"test_init"},
    "TestKimiDetector": {"test_init"},
    "TestKimiK2Detector": {"test_init"},
    "TestGlm45Detector": {"test_init"},
    "TestHunyuanDetector": {"test_init"},
    "TestNemotron3Detector": {"test_init"},
    "TestGemma4Detector": {"test_init"},
    "TestReasoningParser": {"test_stream_reasoning_parameter"},
    "TestMiniMaxAppendThinkDetector": {
        "test_detect_and_parse_prepends_think", "test_streaming_first_chunk_prepends_think",
        "test_streaming_second_chunk_no_prepend",
    },
    "TestReasoningParserAdvanced": {
        "test_continue_final_message_with_request",
        "test_force_nonempty_content_via_chat_template_kwargs",
    },
}
REASONING_SELECTION_CLASSES = {
    "TestBaseReasoningFormatDetector", "TestDeepSeekR1Detector", "TestQwen3Detector",
    "TestQwen3ForcedReasoningDetector", "TestKimiDetector", "TestKimiK2Detector",
    "TestGlm45Detector", "TestHunyuanDetector", "TestNemotron3Detector",
    "TestGemma4Detector", "TestReasoningParser", "TestIntegrationScenarios",
    "TestBufferLossBugFix", "TestGptOssDetector", "TestReasoningParserAdvanced",
    "TestContinueFinalMessage", "TestGptOssDetectorToolCall", "TestPoolsideV1Registered",
}

CONFIG_KEY = ("vllm", "tests/test_config.py")
CONFIG_SHA256 = "66339ade2425d76e68ca643a6aafb081265c5ea17e7b0975b160a7ee77a100e3"
# These remaining functions check formatting, log messages, hashes or paths
# outside the present configuration units, or have no asserted result.
CONFIG_EXTRA_EXCLUDE = {
    "test_dspark_adaptive_verification_separates_graph_cache",
    "test_compile_config_repr_succeeds",
    "test_v2_frontend_config_can_be_created_without_triton",
    "test_engram_hash_tracks_execution_options",
    "test_get_field",
    "test_validate_mamba_align_subblock_prefill",
    "test_target_only_gumbel_allows_speculative_decoding",
    "test_speculative_watermarking_without_context_dedup_does_not_warn",
    "test_speculative_context_dedup_logs_no_unsupported_warning",
    "test_dual_key_gumbel_warns_that_configured_alpha_is_unused",
    "test_dual_key_gumbel_alpha_warning_is_not_emitted",
    "test_dual_key_gumbel_supports_probabilistic_speculative_decoding",
    "test_gumbel_watermark_allows_unwatermarked_beam_search",
    "test_watermark_key_is_excluded_from_serialization",
    "test_mtp_draft_uses_model_weights_not_local_cache",
    "test_qwen3_omni_dspark_checkpoint_contract_is_accepted",
    "test_qwen3_omni_dspark_allows_draft_only_noise_token_row",
    "test_qwen3_omni_dspark_allows_smaller_input_vocabulary",
    "test_ir_op_platform_defaults_support_sparse_gelu",
    "test_ir_op_priority_ctx",
}

PARSER_ENGINE_KEY = ("vllm", "tests/parser/engine/test_parser_engine.py")
PARSER_ENGINE_SHA256 = "8326dd91459d65665f86285ebf2862066fbc8adb3a5a3e03885336673c12d10f"
PARSER_ENGINE_EXCLUDE = {"TestReasoningEndTokenIds", "TestValuePostprocessorRemoved"}
PARSER_ENGINE_TOP_EXCLUDE = {"test_implicit_reasoning_end_of_parsers"}

DEEPSEEK_V4_KEY = ("vllm", "tests/parser/engine/test_deepseek_v4.py")
QWEN3_TOOL_KEY = ("vllm", "tests/parser/engine/test_qwen3.py")
QWEN3_REASONING_KEY = ("vllm", "tests/parser/engine/test_qwen3_reasoning.py")
ENGINE_CORE_KEY = ("vllm", "tests/parser/engine/test_engine.py")
ENGINE_FROZEN_HASHES = {
    DEEPSEEK_V4_KEY: "160c33dbc50fc4cbd726d882b1f04292a30e8261f8c4636f428baa3b84a47daf",
    QWEN3_TOOL_KEY: "5aecbeeec62ba98e9fa576503ba5311cc68a8d4c036e0013f14d4457cc724daa",
    QWEN3_REASONING_KEY: "bd5c0613536e48a414bc31b603b99f0ba3a7228e4c40313c3a714de02f550f44",
    ENGINE_CORE_KEY: "788c71ae1314fd5d1658fc361c8d92e09711af0660925bad49040791f949e1a8",
}
ENGINE_REVIEW_EXCLUDED = {
    DEEPSEEK_V4_KEY: set(),
    QWEN3_TOOL_KEY: {
        "TestArgValueExtender::test_conversions_do_not_grow_with_argument_length",
        "TestArgValueExtender::test_closed_params_are_parsed_once",
    },
    QWEN3_REASONING_KEY: set(),
    ENGINE_CORE_KEY: {
        "TestReasoningTokenCounts::test_counts_reasoning_and_excludes_boundaries_and_content",
        "TestReasoningTokenCounts::test_counts_tokens_after_final_deferred_start_terminal",
        "TestReasoningTokenCounts::test_deferred_reasoning_tokens_stay_before_next_end_terminal",
        "TestNoUnusedTokenizerAttr::test_no_tokenizer_attribute",
        "TestRegexTerminalInfraRemoved::test_terminal_def_no_priority",
        "TestRegexTerminalInfraRemoved::test_lexer_shape_no_regex_terminals",
        "TestSkipToolParsing::test_reset_preserves_skip_tool_parsing",
    },
}
ENGINE_REVIEW_COUNTS = {DEEPSEEK_V4_KEY: 78, QWEN3_TOOL_KEY: 62, QWEN3_REASONING_KEY: 63, ENGINE_CORE_KEY: 37}

FUNCTION_CALL_KEY = ("sglang", "test/registered/unit/function_call/test_function_call_parser.py")
HARMONY_PARSER_KEY = ("sglang", "test/registered/unit/parser/test_harmony_parser.py")
SAMPLING_PARAMS_KEY = ("sglang", "test/registered/unit/sampling/test_sampling_params.py")
GEMMA_REASONING_KEY = ("vllm", "tests/parser/engine/test_gemma4_streaming_reasoning.py")
INKLING_KEY = ("vllm", "tests/parser/engine/test_inkling.py")
ROUND5_HASHES = {
    FUNCTION_CALL_KEY: "be0f71dcec16267a8125f77727d99fe48e98afbe53aa08988866254090e7ecd6",
    HARMONY_PARSER_KEY: "a6469086cd57299782c889d28f3bdc301fe5440350189ee4fbb83e57ba3a5663",
    SAMPLING_PARAMS_KEY: "1a8969753168c8abfd97b1165854b4b683890b6e94bc72c7693833898a126939",
    GEMMA_REASONING_KEY: "69328a95c3f53ffc34a8f5398aedd37e441669925c574847571d9bbf9a9d3a28",
    INKLING_KEY: "10a2e8a09b9bd9087013a0484a379248758b953f8a5164a206501d6d8ca0549e",
}
ROUND5_EXCLUDED = {
    FUNCTION_CALL_KEY: {
        "TestBaseFormatDetector::test_current_tool_id_increment_on_completion",
        "TestKimiK2Detector::test_tool_call_completion",
        "TestQwen3CoderDetector::test_incomplete_tool_call",
        "TestQwen3CoderDetector::test_supports_structural_tag",
        "TestGlm4MoeDetector::test_tool_call_id",
        "TestGlm47MoeDetector::test_tool_call_id",
        "TestJsonArrayParser::test_json_detector_has_no_ebnf",
        "TestJsonArrayParser::test_parse_streaming_increment_malformed_json",
        "TestJsonArrayParser::test_parse_streaming_increment_whitespace_handling",
        "TestJsonArrayParser::test_parse_streaming_increment_nested_objects",
        "TestJsonArrayParser::test_separator_in_separate_chunk",
        "TestJsonArrayParser::test_incomplete_json_across_chunks",
        "TestJsonArrayParser::test_malformed_json_recovery",
        "TestJsonArrayParser::test_empty_objects",
        "TestJsonArrayParser::test_whitespace_handling",
        "TestLfm2Detector::test_supports_structural_tag",
        "TestGetStructureConstraint::test_default_thinking_mode_is_false",
    },
    HARMONY_PARSER_KEY: {
        "TestEvent::test_init", "TestToken::test_init",
        "TestCanonicalStrategy::test_init", "TestTextStrategy::test_init",
        "TestHarmonyParser::test_init",
        "TestAdditionalEdgeCases::test_canonical_standalone_structural_token_filtered",
        "TestAdditionalEdgeCases::test_canonical_return_without_final",
    },
    SAMPLING_PARAMS_KEY: set(),  # Eleven no-assert acceptance tests are skipped below.
    GEMMA_REASONING_KEY: {
        "TestAdapterExtractReasoning::test_skip_tool_parsing_restored_after_extraction",
        "TestGemma4ArgValueExtender::test_conversions_do_not_grow_with_value_length",
        "TestGemma4ArgValueExtender::test_factory_declines_without_open_last_string_value",
    },
    INKLING_KEY: {
        "TestPromptSeededState::test_prompt_ending_in_thinking_starts_reasoning",
        "TestPromptSeededState::test_prompt_ending_in_text_starts_content",
        "TestPromptSeededState::test_generation_prompt_tail_starts_message_header",
        "TestRegisteredAdapters::test_adapters_resolve",
    },
}
ROUND5_COUNTS = {
    FUNCTION_CALL_KEY: 193, HARMONY_PARSER_KEY: 66, SAMPLING_PARAMS_KEY: 57,
    GEMMA_REASONING_KEY: 63, INKLING_KEY: 61,
}

COHERE_CONVERSION_KEY = ("vllm", "tests/entrypoints/cohere/test_serving_conversion.py")
ANTHROPIC_CONVERSION_KEY = ("vllm", "tests/entrypoints/anthropic/test_anthropic_messages_conversion.py")
STRUCTURAL_TAG_KEY = ("vllm", "tests/tool_parsers/test_structural_tag_registry.py")
CHAT_PARSING_KEY = ("vllm", "tests/parser/test_chat_parsing.py")
ROUND6_HASHES = {
    COHERE_CONVERSION_KEY: "8c1da6b4b7a9d883c9f9c1296c2fb5d3c30bae71d73417c6ff6fc54b0c3c3820",
    ANTHROPIC_CONVERSION_KEY: "7f8ef99054fb6f07fb3e4b31241b183c8bed8b69f3c92ce7f1d8760e705313be",
    STRUCTURAL_TAG_KEY: "cd6e6b52fc8d981fdbf16ab326918293bb122eebf777a3254d0fcd6a72e868d7",
    CHAT_PARSING_KEY: "503d3e7f6ce404a101c215dd8aea5dce9ced251b2274dd44d4fcae58372c11ac",
}
ROUND6_EXCLUDED = {
    COHERE_CONVERSION_KEY: {
        "TestMapFinishReason::test_finish_reason_map_is_complete",
        "TestEngineChatTemplateKwargs::test_all_internal_keys_dropped",
    },
    ANTHROPIC_CONVERSION_KEY: set(),
    STRUCTURAL_TAG_KEY: {
        "test_supported_structural_tag_models_include_vllm_builtins",
        "test_plamo3_registered_as_vllm_structural_tag_model",
        "test_kimi_k3_registered_as_vllm_builtin",
        "test_tool_parsers_declare_matching_xgrammar_builtin_model",
        "test_tool_parsers_without_structural_tag_support_required_and_named",
    },
    CHAT_PARSING_KEY: {
        # Already retained in the separate confirmed-property crosswalk.
        "ToolArgCoercionTest::test_tools_is_a_noop_for_typed_json_tool_calls",
        "ToolArgCoercionTest::test_merge_duplicates_arguments_are_cast_element_wise",
        "ToolArgCoercionTest::test_coercion_never_reworks_already_typed_values",
        "ToolArgCoercionTest::test_kv_lines_string_args_are_coerced",
        "ToolArgCoercionTest::test_union_with_container_still_keeps_scalar_text",
        "ToolArgCoercionTest::test_non_tool_call_regions_are_untouched",
        "ToolArgCoercionTest::test_qwen3_tools_coerces_strings_left_by_value_parser",
        "ToolArgCoercionTest::test_parse_response_tools_coerces_xml_inline_string_args",
        "ToolArgCoercionTest::test_parse_response_tools_coerces_one_of_string_args",
    },
}
ROUND6_COUNTS = {
    COHERE_CONVERSION_KEY: 104, ANTHROPIC_CONVERSION_KEY: 87,
    STRUCTURAL_TAG_KEY: 73, CHAT_PARSING_KEY: 66,
}


def round6_unit(key: tuple[str, str], class_name: str, name: str) -> str:
    if key == COHERE_CONVERSION_KEY:
        if class_name == "TestMapFinishReason":
            return "output.finalization"
        if class_name == "TestBuildUsage":
            return "response.usage_accounting"
        if class_name in {"TestChatCompletionToV2", "TestExtractCitations"}:
            return "response.delivery"
        if class_name == "TestCreateErrorResponse":
            return "input.error_response_classification"
        if not class_name and "watermarking" in name:
            return "request.effective_sampling_parameters"
        return "request.admission"
    if key == ANTHROPIC_CONVERSION_KEY:
        if class_name == "TestBuildAnthropicUsage":
            return "response.usage_accounting"
        if class_name in {"TestMessagesFullConverter", "TestStopSequenceReason", "TestToolUseStopReason"}:
            return "response.delivery"
        if class_name == "TestClientErrorResponses" or name == "test_empty_cache_salt_returns_bad_request":
            return "input.error_response_classification"
        if class_name == "TestWatermarking":
            return "request.effective_sampling_parameters"
        return "request.admission"
    if key == STRUCTURAL_TAG_KEY:
        return "tool.constraint_schema_construction"
    if class_name == "ToolArgCoercionTest":
        return "output.tool_argument_type_binding"
    if class_name == "PrefixAndTruncationTest":
        return "output.structured_marker_boundary"
    if class_name == "ResponseEventStreamTest":
        return "output.incremental_parse_relation"
    if "validation" in name or "rejected" in name or "raises" in name:
        return "output.parser_format_selection"
    return "tool.call_parse_relation" if "tool" in name else "output.channel_event_relation"


def round6_cases(key: tuple[str, str], source: Path) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ROUND6_HASHES[key], source
    body = raw.decode("utf-8")
    selected = {}
    for outer in ast.parse(body).body:
        if isinstance(outer, ast.ClassDef):
            members, class_name = outer.body, outer.name
        elif isinstance(outer, ast.FunctionDef):
            members, class_name = [outer], ""
        else:
            continue
        for node in members:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            name = f"{class_name}::{node.name}" if class_name else node.name
            if name in ROUND6_EXCLUDED[key]:
                continue
            if not oracle_lines(node):
                continue
            checks = []
            for child in ast.walk(node):
                if isinstance(child, ast.Assert) or (
                    isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr.startswith("assert")
                ):
                    checks.append((child.lineno, " ".join(ast.unparse(child).split())))
                elif isinstance(child, ast.With):
                    for item in child.items:
                        if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == "raises":
                            checks.append((child.lineno, " ".join(ast.unparse(item.context_expr).split())))
            checks = [item for _, item in sorted(set(checks))]
            assert checks, name
            summary = (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]
            selected[name] = (
                round6_unit(key, class_name, node.name),
                f"{summary}. Direct checks for the source test's input: {'; '.join(checks)}",
                "round6_source_review",
            )
    assert len(selected) == ROUND6_COUNTS[key], (key, len(selected))
    return selected


def round5_unit(key: tuple[str, str], class_name: str, name: str) -> str:
    if key == FUNCTION_CALL_KEY:
        if class_name == "TestGetStructureConstraint" or "structural_tag" in name or name == "test_structure_info":
            return "tool.constraint_schema_construction"
        return "tool.call_parse_relation"
    if key == HARMONY_PARSER_KEY:
        if class_name in {"TestPrefixHold", "TestIterTokens"} or name.startswith(("test_prefix_hold", "test_iter_tokens")):
            return "output.structured_marker_boundary"
        if class_name == "TestHarmonyParser" and name.startswith("test_strategy_selection"):
            return "output.parser_format_selection"
        return "output.channel_event_relation"
    if key == SAMPLING_PARAMS_KEY:
        if class_name == "TestRegexMaxLength":
            return "sampling.stop_regex_buffer_bound"
        if class_name == "TestSamplingParamsNormalize":
            return "sampling.stop_condition_normalization"
        if class_name == "TestSamplingParamsInit":
            return "sampling.zero_temperature_mode" if "temperature" in name and "greedy" in name else "sampling.parameter_normalization"
        if "nan_temperature" in name or "inf_temperature" in name:
            return "sampling.temperature_finiteness"
        if "top_p" in name:
            return "sampling.top_p_range"
        if "min_exceeds_max" in name or "grammar" in name:
            return "sampling.cross_parameter_admissibility"
        return "sampling.numeric_parameter_bounds"
    if key == GEMMA_REASONING_KEY:
        if "Schema" in class_name:
            return "output.tool_argument_type_binding"
        if class_name == "TestGemma4IsReasoningEnd":
            return "output.structured_marker_boundary"
        if class_name == "TestGemma4ArgValueExtender":
            return "output.incremental_parse_relation"
        if "Reasoning" in class_name or class_name in {"TestBareThoughtWithoutChannelOpener", "TestAdapterExtractReasoning"}:
            return "output.channel_event_relation"
        return "tool.call_parse_relation"
    if class_name == "TestArgConverter":
        return "output.tool_argument_type_binding"
    if class_name == "TestPromptSeededState":
        return "output.parser_format_selection"
    if class_name == "TestRegisteredAdapters":
        return "tool.call_parse_relation"
    return "output.incremental_parse_relation" if "Streaming" in class_name or "streaming" in name else "output.channel_event_relation"


def round5_cases(key: tuple[str, str], source: Path) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ROUND5_HASHES[key], source
    body = raw.decode("utf-8")
    selected = {}
    for outer in ast.parse(body).body:
        if isinstance(outer, ast.ClassDef):
            members = outer.body
            class_name = outer.name
        elif key == INKLING_KEY and isinstance(outer, ast.FunctionDef):
            members = [outer]
            class_name = ""
        else:
            continue
        for node in members:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            name = f"{class_name}::{node.name}" if class_name else node.name
            if name in ROUND5_EXCLUDED[key]:
                continue
            sites = oracle_lines(node)
            if key == SAMPLING_PARAMS_KEY and not sites:
                continue
            assert sites, name
            checks = []
            for child in ast.walk(node):
                if isinstance(child, ast.Assert) or (
                    isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr.startswith("assert")
                ):
                    checks.append((child.lineno, " ".join(ast.unparse(child).split())))
                elif isinstance(child, ast.With):
                    for item in child.items:
                        if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute) and item.context_expr.func.attr == "raises":
                            checks.append((child.lineno, " ".join(ast.unparse(item.context_expr).split())))
            checks = [item for _, item in sorted(set(checks))]
            assert checks, name
            summary = (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]
            selected[name] = (
                round5_unit(key, class_name, node.name),
                f"{summary}. Direct checks for the source test's input: {'; '.join(checks)}",
                "round5_source_review",
            )
    assert len(selected) == ROUND5_COUNTS[key], (key, len(selected))
    return selected


def family_parser_unit(key: tuple[str, str], class_name: str, name: str) -> str:
    streamed = "stream" in name and "non_stream" not in name
    if key == DEEPSEEK_V4_KEY:
        if class_name == "TestArgConverter":
            return "output.tool_argument_type_binding"
        if class_name == "TestThinkingModeConfig" and name in {
            "test_thinking_true_starts_in_reasoning", "test_thinking_false_starts_in_content",
            "test_parser_thinking_mode_matches_tokenizer_default",
            "test_reasoning_effort_none_overrides_enable_thinking",
        }:
            return "output.parser_format_selection"
        if name == "test_reasoning_end_token_ids_drop_multi_token_tool_opener":
            return "output.structured_marker_boundary"
        if class_name in {"TestThinkTagAbsorption"} or "reasoning" in name and "tool" not in name:
            return "output.incremental_parse_relation" if streamed else "output.channel_event_relation"
        if class_name in {"TestArgValueExtender", "TestDelegatingParserLargeDelta"} or streamed:
            return "output.incremental_parse_relation"
        return "tool.call_parse_relation"
    if key == QWEN3_TOOL_KEY:
        if class_name in {
            "TestSchemaAwareTypeCoercion", "TestAnyOfTypeCoercion", "TestSchemaCoercionBoolNumberNull",
            "TestNestedSchemaCoercion",
        }:
            return "output.tool_argument_type_binding"
        if class_name in {"TestStreaming", "TestArgValueExtender"}:
            return "output.incremental_parse_relation"
        return "tool.call_parse_relation"
    if key == QWEN3_REASONING_KEY:
        if class_name in {
            "TestIsReasoningEnd", "TestIsReasoningEndTurnBoundaries", "TestDelegatingPromptDetection",
        }:
            return "output.structured_marker_boundary"
        if class_name == "TestIsReasoningEndDerivedFromGrammar":
            return "output.parser_format_selection" if "wait_for_reasoning" in name else "output.structured_marker_boundary"
        if class_name == "TestThinkingDisabled" and "initial_state" in name:
            return "output.parser_format_selection"
        return "output.incremental_parse_relation" if "stream" in name and "non_stream" not in name else "output.channel_event_relation"
    if class_name in {"TestNonStreaming", "TestLexerBufferFlush", "TestSkipToolParsingFromMessageHeader"}:
        return "output.channel_event_relation"
    if class_name in {"TestTokenIdFiltering", "TestTextOnlyFallbackFiltering", "TestTerminalAliases", "TestRegexTerminalInfraRemoved"}:
        return "output.structured_marker_boundary"
    if class_name == "TestSkipToolParsing":
        return "tool.call_parse_relation"
    return "output.incremental_parse_relation"


def family_parser_cases(key: tuple[str, str], source: Path) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ENGINE_FROZEN_HASHES[key], source
    body = raw.decode("utf-8")
    selected = {}
    for outer in ast.parse(body).body:
        if isinstance(outer, ast.ClassDef):
            members = [(f"{outer.name}::{node.name}", outer.name, node) for node in outer.body
                       if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
        elif isinstance(outer, ast.FunctionDef) and outer.name.startswith("test_"):
            members = [(outer.name, "", outer)]
        else:
            continue
        for name, class_name, node in members:
            if name in ENGINE_REVIEW_EXCLUDED[key]:
                continue
            checks = []
            for child in ast.walk(node):
                if isinstance(child, ast.Assert) or (
                    isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr.startswith("assert")
                ):
                    fragment = ast.get_source_segment(body, child)
                    assert fragment
                    checks.append((child.lineno, " ".join(fragment.split())))
                elif isinstance(child, ast.With):
                    for item in child.items:
                        fragment = ast.get_source_segment(body, item.context_expr)
                        if fragment and "raises" in fragment:
                            checks.append((child.lineno, " ".join(fragment.split())))
            checks = [fragment for _, fragment in sorted(set(checks))]
            assert checks, name
            summary = (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]
            selected[name] = (
                family_parser_unit(key, class_name, node.name),
                f"{summary}. Direct checks for the test's parser input: {'; '.join(checks)}",
                "vllm_parser_family_review",
            )
    assert len(selected) == ENGINE_REVIEW_COUNTS[key], (key, len(selected))
    return selected


def parser_engine_unit(class_name: str, name: str) -> str:
    if class_name in {"TestFixArgTypes", "TestCoercionInstabilityRegression"}:
        return "output.tool_argument_type_binding"
    if class_name == "TestArgDeltaWithConverter" and "type_coercion" in name:
        return "output.tool_argument_type_binding"
    if class_name in {"TestEngineBasedPath", "TestToolAdapterForwardsKwargs"} and (
        "engine_based" in name or "kwargs" in name
    ):
        return "output.parser_format_selection"
    if name.startswith("test_parser_manager_preserves_"):
        return "output.parser_format_selection"
    if class_name in {"TestDropSpecialTokens", "TestExtractContentIdsNoEmptyReturn"}:
        return "output.structured_marker_boundary"
    if class_name in {"TestEventsToDelta", "TestThinkMarkupWithoutReasoningParser"}:
        return "output.channel_event_relation"
    if class_name in {
        "TestCoalesceToolCallDeltas", "TestBuildExtractedResult", "TestParseTokenIdPassthrough",
        "TestSkipToolSpanForwarding", "TestTruncatedToolOpenerStreamParity",
    }:
        return "tool.call_parse_relation"
    return "output.incremental_parse_relation"


def parser_engine_cases(source: Path) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARSER_ENGINE_SHA256, source
    body = raw.decode("utf-8")
    selected = {}
    tree = ast.parse(body)
    for outer in tree.body:
        if isinstance(outer, ast.ClassDef):
            if outer.name in PARSER_ENGINE_EXCLUDE:
                continue
            members = [(f"{outer.name}::{node.name}", outer.name, node) for node in outer.body
                       if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
        elif isinstance(outer, ast.FunctionDef) and outer.name.startswith("test_"):
            members = [] if outer.name in PARSER_ENGINE_TOP_EXCLUDE else [(outer.name, "", outer)]
        else:
            continue
        for name, class_name, node in members:
            checks = []
            for child in ast.walk(node):
                if isinstance(child, ast.Assert) or (
                    isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr.startswith("assert")
                ):
                    fragment = ast.get_source_segment(body, child)
                    assert fragment
                    checks.append((child.lineno, " ".join(fragment.split())))
                elif isinstance(child, ast.With):
                    for item in child.items:
                        fragment = ast.get_source_segment(body, item.context_expr)
                        if fragment and "raises" in fragment:
                            checks.append((child.lineno, " ".join(fragment.split())))
            checks = [fragment for _, fragment in sorted(set(checks))]
            assert checks, name
            summary = (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]
            selected[name] = (
                parser_engine_unit(class_name, node.name),
                f"{summary}. Direct checks under the test's parser configuration: {'; '.join(checks)}",
                "vllm_parser_engine_review",
            )
    assert len(selected) == 90, len(selected)
    return selected


def reviewed_config_cases(source: Path, existing: set[str]) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONFIG_SHA256, source
    body = raw.decode("utf-8")
    selected = {}
    for node in ast.parse(body).body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if node.name in existing or node.name in CONFIG_EXTRA_EXCLUDE:
            continue
        checks = []
        for child in ast.walk(node):
            if isinstance(child, ast.Assert) or (
                isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                and child.func.attr.startswith("assert")
            ):
                fragment = ast.get_source_segment(body, child)
                assert fragment
                checks.append((child.lineno, " ".join(fragment.split())))
            elif isinstance(child, ast.With):
                for item in child.items:
                    fragment = ast.get_source_segment(body, item.context_expr)
                    if fragment and "raises" in fragment:
                        checks.append((child.lineno, " ".join(fragment.split())))
        checks = [fragment for _, fragment in sorted(set(checks))]
        assert checks, node.name
        if any("raises(" in check or "assertRaises" in check for check in checks) or any(
            word in node.name for word in ("reject", "requires", "validation", "unsupported", "support")
        ):
            unit = "deployment.configuration_admissibility"
        else:
            unit = "deployment.effective_configuration"
        summary = (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]
        selected[node.name] = (
            unit, f"{summary}. Direct checks in the constructed configuration: {'; '.join(checks)}",
            "vllm_config_source_review",
        )
    assert len(selected) == 110, len(selected)
    return selected


def reasoning_unit(class_name: str, name: str) -> str:
    if class_name in {"TestReasoningParserAdvanced", "TestPoolsideV1Registered"}:
        return "output.parser_format_selection"
    if class_name == "TestReasoningParser" and name in {
        "test_init_valid_model", "test_init_invalid_model", "test_init_no_model",
        "test_case_insensitive_model_type",
    }:
        return "output.parser_format_selection"
    if "non_stream" in name:
        return "output.channel_event_relation"
    if class_name in {"TestBufferLossBugFix", "TestContinueFinalMessage"} or (
        "stream" in name or "partial" in name
    ):
        return "output.incremental_parse_relation"
    return "output.channel_event_relation"


def reasoning_cases(source: Path) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REASONING_SHA256, source
    body = raw.decode("utf-8")
    selected = {}
    observed_classes = set()
    for outer in ast.parse(body).body:
        if not isinstance(outer, ast.ClassDef) or outer.name not in REASONING_SELECTION_CLASSES:
            continue
        observed_classes.add(outer.name)
        for node in outer.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            if node.name in REASONING_EXCLUDE.get(outer.name, set()):
                continue
            checks = []
            for child in ast.walk(node):
                if isinstance(child, ast.Assert) or (
                    isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and child.func.attr.startswith("assert")
                ):
                    fragment = ast.get_source_segment(body, child)
                    assert fragment
                    checks.append((child.lineno, " ".join(fragment.split())))
            checks = [fragment for _, fragment in sorted(checks)]
            assert checks, (outer.name, node.name)
            summary = (ast.get_docstring(node) or "Poolside parser selection").splitlines()[0]
            unit = reasoning_unit(outer.name, node.name)
            selected[f"{outer.name}::{node.name}"] = (
                unit, f"{summary} Direct checks: {'; '.join(checks)}", "reasoning_parser_review"
            )
    assert observed_classes == REASONING_SELECTION_CLASSES
    assert len(selected) == 105, len(selected)
    return selected

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
        **{name: (unit, prop, "environment_effective") for name, (unit, prop) in ENV_OTHER.items()},
        **{f"TestVllmConfigureLogging::{name}": ("deployment.effective_configuration", prop, "logging_environment")
           for name, prop in ENV_LOGGING.items()},
        **{f"TestVllmMaxNSequences::{name}": ("deployment.effective_configuration", prop, "max_sequences_environment")
           for name, prop in ENV_MAX_N.items()},
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
    REASONING_KEY: {},
    PARSER_ENGINE_KEY: {},
    DEEPSEEK_V4_KEY: {},
    QWEN3_TOOL_KEY: {},
    QWEN3_REASONING_KEY: {},
    ENGINE_CORE_KEY: {},
    FUNCTION_CALL_KEY: {},
    HARMONY_PARSER_KEY: {},
    SAMPLING_PARAMS_KEY: {},
    GEMMA_REASONING_KEY: {},
    INKLING_KEY: {},
    COHERE_CONVERSION_KEY: {},
    ANTHROPIC_CONVERSION_KEY: {},
    STRUCTURAL_TAG_KEY: {},
    CHAT_PARSING_KEY: {},
}
CASES.update({("vllm", path): {} for path in CORE_SOURCE_HASHES})
CASES.update({key: {} for key in ROUND7_SOURCE_HASHES})

CONDITIONS = {
    "configuration": "Pinned vLLM snapshot; parametrized model, platform, environment and config inputs are fixed in the linked test definition. Local config result only; no server start.",
    "eviction_policy": "Pinned SGLang snapshot; strategy factory called with the policy name in the linked test. Class selection only; no runtime eviction sequence.",
    "environment_choice": "Pinned vLLM snapshot; patched TEST_ENV or absent environment variable and declared allowed choices in the linked test. Parsing only; no service start.",
    "environment_effective": "Pinned vLLM snapshot; patched environment, cache state or binary-path mock in the linked test. Local setting resolution only.",
    "logging_environment": "Pinned vLLM snapshot; VLLM_CONFIGURE_LOGGING environment value and cache clear in the linked test. No logging emission measured.",
    "max_sequences_environment": "Pinned vLLM snapshot; VLLM_MAX_N_SEQUENCES unset or overridden in the linked test. Checks the effective value only.",
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
    "reasoning_parser_review": "Pinned SGLang snapshot; detector/model type and exact text or chunk sequence in the linked test. Source-level parser output or selection only; no model generation.",
    "vllm_config_source_review": "Pinned vLLM snapshot; model/platform/parameter combinations and mocks in the linked definition. Local configuration fields or rejection only; no deployed model run.",
    "vllm_parser_engine_review": "Pinned vLLM snapshot; mocked tokenizer, declared parser transitions and input events/chunks in the linked test. Local parser/adaptor assertions only.",
    "vllm_parser_family_review": "Pinned vLLM snapshot; model-specific parser grammar, mocked tokenizer and exact text/token/chunk inputs at the linked definition. Parser output only; no model generation.",
    "round5_source_review": "Pinned framework snapshot; exact constructor, input text/chunks or sampling parameters and assertions are at the linked source definition. Local parser or parameter behavior only; no model generation or server run.",
    "round6_source_review": "Pinned vLLM snapshot; exact API message, parser template, tool grammar, model output and assertion inputs are in the linked definition. Local conversion, grammar or parsing behavior only; no model generation or server run.",
    "core_source_review": "Pinned vLLM snapshot; requests, block layouts, transfer mocks, budgets and exact assertions are at the linked test definition. This is a local state or protocol check, not a GPU tensor-content or end-to-end model run.",
    "round7_source_review": "Pinned framework snapshot; constructed messages, model metadata, tensor fixtures, configuration and direct assertions are in the linked test. Local relation only; this review did not execute the test, model or server.",
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
    prior: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for filename in ("confirmed-property-crosswalk.tsv", "inherited-property-crosswalk.tsv"):
        with (HERE / filename).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                prior.setdefault((row["snapshot"], row["path"]), set()).add((row["test_name"], row["line"]))
    with (HERE.parent / "oracle-review-queue.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["source_review_status"] == "mapped" and row["source_reviewed_units"]:
                prior.setdefault((row["snapshot"], row["path"]), set()).add((row["test_name"], row["line"]))
    rows = []
    for (snapshot, path), cases in CASES.items():
        source = ROOTS[snapshot] / path
        if (snapshot, path) == REASONING_KEY:
            cases = reasoning_cases(source)
        elif (snapshot, path) == CONFIG_KEY:
            cases = {**cases, **reviewed_config_cases(source, set(cases))}
        elif (snapshot, path) == PARSER_ENGINE_KEY:
            cases = parser_engine_cases(source)
        elif (snapshot, path) in ENGINE_FROZEN_HASHES:
            cases = family_parser_cases((snapshot, path), source)
        elif (snapshot, path) in ROUND5_HASHES:
            cases = round5_cases((snapshot, path), source)
        elif (snapshot, path) in ROUND6_HASHES:
            cases = round6_cases((snapshot, path), source)
        elif snapshot == "vllm" and path in CORE_SOURCE_HASHES:
            cases = core_reviewed_cases(path, source, prior.get((snapshot, path), set()))
        elif (snapshot, path) in ROUND7_SOURCE_HASHES:
            cases = round7_reviewed_cases((snapshot, path), source, prior.get((snapshot, path), set()))
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
