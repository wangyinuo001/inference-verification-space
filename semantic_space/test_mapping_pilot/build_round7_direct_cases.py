"""Pinned source review for distributed weights, parser, renderer and cache tests."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PATH_HASHES = {
    ("vllm", "tests/distributed/test_weight_transfer.py"): "f1839b0105c4820c79e7a8785597d3f72fe0da7fca438558d8ec8117ff71f9b5",
    ("vllm", "tests/distributed/test_sharded_rdt_plan.py"): "205f2577c7f147aad14c33268f70f94c995a745b87edc743d38a0299615b9403",
    ("vllm", "tests/distributed/test_weight_transfer_m2n.py"): "c055cc2308f8489228cc2b39693766ccacd385eb406718b58e8f9753d0dc7be1",
    ("vllm", "tests/distributed/test_sharded_rdt_producer.py"): "12495a99f1f5de6351ace3d8a2e09808ad024e295bd00c624253d4f3a97327d3",
    ("vllm", "tests/renderers/test_cohere.py"): "c64cd13d8d6a6124c61b2e410f49570953d5dfbf3c11ee207a6f952641a93f80",
    ("vllm", "tests/tool_parsers/test_utils.py"): "f77b593bd04979257978ef733cd55d616eb31482670f4d19464c6d63df4fe61c",
    ("vllm", "tests/parser/mistral/test_tool_calls.py"): "e736082e8f5ebaec73a23ecee2bd4ec7180680a3b43761f5fc0ba6d07a07a5ac",
    ("vllm", "tests/tool_parsers/test_deepseekv32_tool_parser.py"): "c66305c5fd7f8c6f909e97fbb74c4a514d7961cad5b2f910304fa5ea449249f8",
    ("vllm", "tests/engine/test_arg_utils.py"): "d88bd2a93c96343b29a2b9e229f707740c4a87dcd66716bdcaf83524c995ffa6",
    ("vllm", "tests/entrypoints/launchers/test_launch_cli.py"): "ba66b16287ba893304c14c2e8dd06ec9d2fcff0e78b0539bbd55bbf7b364f8d2",
    ("vllm", "tests/v1/kv_offload/tiering/p2p/test_data_transport.py"): "0c1efba3924cf8aef38dc5c9e9b42cd47280334bfcf251493ed2a71842826305",
    ("vllm", "tests/v1/kv_connector/unit/test_moriio_hma_scheduler.py"): "24eb45b3346a36bc4124e7fb4b511f315a2a36750b1c8fed15aad495d7e06094",
    ("vllm", "tests/distributed/aux_output_connector/test_store.py"): "d41a8e5c5f69fdc286db7a7d9beb02895260e87ee45fb0cca00b6501d4ae6059",
    ("vllm", "tests/entrypoints/openai/responses/test_serving_responses.py"): "f40ca03754ab0dbe8f1921de951a636aebef42882b56f970c9b302e6fe679d61",
    ("vllm", "tests/entrypoints/openai/chat_completion/test_serving_chat.py"): "219e90987a02afe32ff034d8e9cd2f5e2b311c0baa8bbca72f5914167a720e06",
    ("sglang", "test/registered/unit/parser/test_conversation.py"): "92afc08e9697102c2df8d43f89304190068afeb7c78584a1216b87329ddd4cb7",
    ("sglang", "test/registered/unit/platforms/test_platform_interface.py"): "1aea210289e45577946931c040938e76c6ed613a6d909b741452cdf5368e913b",
    ("sglang", "test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py"): "72bc7deba13183e82b2251c55dc1b941ab2e1de9e55938af2e4e63adcda6d802",
}

EXPECTED_COUNTS = {
    ("vllm", "tests/distributed/test_weight_transfer.py"): 82,
    ("vllm", "tests/distributed/test_sharded_rdt_plan.py"): 87,
    ("vllm", "tests/distributed/test_weight_transfer_m2n.py"): 64,
    ("vllm", "tests/distributed/test_sharded_rdt_producer.py"): 58,
    ("vllm", "tests/renderers/test_cohere.py"): 90,
    ("vllm", "tests/tool_parsers/test_utils.py"): 78,
    ("vllm", "tests/parser/mistral/test_tool_calls.py"): 48,
    ("vllm", "tests/tool_parsers/test_deepseekv32_tool_parser.py"): 49,
    ("vllm", "tests/engine/test_arg_utils.py"): 45,
    ("vllm", "tests/entrypoints/launchers/test_launch_cli.py"): 7,
    ("vllm", "tests/v1/kv_offload/tiering/p2p/test_data_transport.py"): 59,
    ("vllm", "tests/v1/kv_connector/unit/test_moriio_hma_scheduler.py"): 69,
    ("vllm", "tests/distributed/aux_output_connector/test_store.py"): 64,
    ("vllm", "tests/entrypoints/openai/responses/test_serving_responses.py"): 25,
    ("vllm", "tests/entrypoints/openai/chat_completion/test_serving_chat.py"): 8,
    ("sglang", "test/registered/unit/parser/test_conversation.py"): 99,
    ("sglang", "test/registered/unit/platforms/test_platform_interface.py"): 63,
    ("sglang", "test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py"): 71,
}
assert EXPECTED_COUNTS.keys() == PATH_HASHES.keys()

_ARG_TYPE_HELPERS = {
    "test_is_type", "test_contains_type", "test_get_type", "test_literal_to_kwargs",
    "test_is_not_builtin", "test_get_type_hints", "test_get_kwargs", "test_hf_token_get_kwargs",
}
_LAUNCH_REVIEW = {
    "test_parse_launch_render", "test_parse_launch_requires_component", "test_parse_launch_invalid_component",
    "test_cmd_launch_render_calls_run", "test_cmd_launch_model_tag_overrides", "test_cmd_launch_model_tag_none",
    "test_cmd_dispatches",
}


def unit_for_test(snapshot: str, path: str, class_name: str, name: str) -> str | None:
    low = name.lower()
    if path.endswith("distributed/test_weight_transfer.py"):
        if class_name in {"TestEngineRegistry", "TestDeferredProcessingContract", "TestTrainerFactory", "TestTrainerEngineBase"}:
            return None
        if class_name == "TestTrainerClients" and "protocol" in low:
            return None
        if low in {"test_plugin_does_not_replace_native_backend", "test_modelexpress_plugin_does_not_replace_native_backend"}:
            return None
        if "shutdown" in low or "release" in low:
            return "shared_resource.lifetime"
        if class_name in {"TestModuleSource", "TestModuleSourceDTensor", "TestParamMetaOwnership"}:
            return "distributed.partition_binding"
        if class_name == "TestTrainerClients" or "client" in low and "order" in low:
            return "distributed.message_delivery"
        if "missing_backend" in low or "invalid_backend" in low:
            return "deployment.configuration_admissibility"
        return "model.parameter_binding"
    if path.endswith("distributed/test_sharded_rdt_plan.py"):
        if class_name == "TestOpAllowlistAgreement" or low == "test_unsupported_is_a_notimplementederror":
            return None
        if class_name == "TestBufferAllocBytes":
            return "deployment.shared_memory_capacity_admission"
        if class_name == "TestRequiresTheRayExecutor":
            return "deployment.configuration_admissibility"
        if class_name == "TestSignalCompleteness" or "free_signal" in low:
            return "shared_resource.lifetime"
        return "distributed.partition_binding"
    if path.endswith("distributed/test_weight_transfer_m2n.py"):
        if class_name == "TestRegistration":
            return None
        if "shutdown" in low or "cleanup" in low or "partial_worker_initialization_releases" in low:
            return "shared_resource.lifetime"
        if "socket" in low:
            return "distributed.remote_endpoint_binding"
        if low == "test_m2n_communicator_uses_nccl4py_uid_under_unpinned_env":
            return "deployment.effective_configuration"
        if low == "test_fallback_tensors_share_one_load_weights_invocation":
            return "model.parameter_binding"
        if class_name in {"TestLayout", "TestSourceLayout", "TestDestinationResolution", "TestPreflightAgreement", "TestWireTypes"}:
            return "distributed.partition_binding"
        return "model.parameter_binding"
    if path.endswith("distributed/test_sharded_rdt_producer.py"):
        if not class_name or low in {"test_the_default_lookahead_is_one", "test_default_timeout_is_the_documented_one"}:
            return None
        if class_name in {"TestExportRing", "TestExportRingSlotSafety", "TestSharedServeSlots", "TestSharedReservation"}:
            return "shared_resource.buffer_slot_integrity"
        if class_name in {"TestFreeBarrier", "TestGatherCredit", "TestEndSync", "TestBeginSync", "TestStallWatchdog", "TestConcurrentPublishAndFree", "TestFakeServerAgreesWithTheRealOne"}:
            return "shared_resource.lifetime"
        return "model.parameter_binding"
    if path.endswith("renderers/test_cohere.py"):
        if class_name == "TestContentBlocks" and ("image" in low or "document" in low):
            return "input.multimodal_binding"
        if "invalid" in low or "rejected" in low or "raise" in low:
            return "request.admission"
        return "input.chat_template_tokenization"
    if path.endswith("tool_parsers/test_utils.py"):
        return "output.tool_argument_type_binding" if class_name == "TestExtractTypesFromSchema" or "parameter_value" in class_name.lower() else "tool.call_parse_relation"
    if path.endswith("parser/mistral/test_tool_calls.py"):
        if "adjust_request" in low or "grammar" in low:
            return "tool.constraint_schema_construction"
        if "reasoning" in low:
            return "output.channel_event_relation"
        return "tool.call_parse_relation"
    if path.endswith("tool_parsers/test_deepseekv32_tool_parser.py"):
        if "type_conversion" in low or "schema" in low or "multi_typed" in low or "number_param" in low:
            return "output.tool_argument_type_binding"
        return "tool.call_parse_relation"
    if path.endswith("engine/test_arg_utils.py"):
        if low in _ARG_TYPE_HELPERS or "deprecation_warning" in low:
            return None
        if "validation" in low or "invalid" in low:
            return "deployment.configuration_admissibility"
        return "deployment.effective_configuration"
    if path.endswith("entrypoints/launchers/test_launch_cli.py"):
        return "deployment.effective_configuration" if low in _LAUNCH_REVIEW else None
    if path.endswith("p2p/test_data_transport.py"):
        if low == "test_raising_state_check_warns_once":
            return None
        if class_name == "TestNixlAgentConfigSelection" or "config_fingerprint" in low:
            return "deployment.effective_configuration"
        if class_name == "TestNixlRegistrationRollback" or "release" in low or "remove" in low:
            return "cache.lifetime"
        if "peer_id" in low or "metadata" in low:
            return "disagg.request_binding"
        return "disagg.transfer_gate"
    if path.endswith("test_moriio_hma_scheduler.py"):
        if "reject" in low or "invalid" in low:
            return "deployment.configuration_admissibility"
        if any(k in low for k in ("truncate_mamba_request", "on_new_request_truncates")):
            return "request.prefill_token_view"
        if any(k in low for k in ("split_block", "group", "offset", "layout", "block_size", "layer")):
            return "disagg.kv_mapping"
        if "match" in low or "hit" in low:
            return "cache.hit_accounting"
        return "disagg.transfer_gate"
    if path.endswith("aux_output_connector/test_store.py"):
        if low == "test_background_store_publishes_without_blocking_caller":
            return None
        if low == "test_non_output_rank_skips_capture_snapshot":
            return "distributed.partition_binding"
        if low == "test_worker_ignores_cudagraph_query_padding":
            return "request.batch_field_binding"
        if low == "test_store_lru_and_immutable_put":
            return "cache.eviction_priority"
        if low.startswith("test_scheduler_connector") or low in {"test_scheduler_consumes_ordered_stale_aux_outputs", "test_prompt_end_offset_through_worker_and_scheduler"}:
            return "distributed.message_delivery"
        if "reject" in low or "fail" in low:
            return "cache.state_invariants"
        if any(k in low for k in ("key", "hash")):
            return "cache.namespace"
        if any(k in low for k in ("release", "retain", "evict", "pin", "ref", "close", "pending")):
            return "cache.lifetime"
        return "cache.safe_reuse"
    if path.endswith("responses/test_serving_responses.py"):
        if class_name == "TestHarmonyPreambleStreaming":
            return "response.stream_event_lifecycle"
        if low == "test_prompt_cache_options_request_accepts_prewarm":
            return "request.admission"
        if "validate" in low or "reject" in low or "non_single_prompt" in low:
            return "request.admission"
        if "extract_tool_types" in low or "allowed_tools" in low:
            return "tool.constraint_schema_construction"
        if "stream" in low or "event" in low:
            return "response.stream_event_lifecycle"
        return "response.delivery"
    if path.endswith("chat_completion/test_serving_chat.py"):
        if "timing_metrics" in low or low == "test_async_serving_chat_init" or low == "test_serving_chat_starts_without_tokenizer":
            return None
        if "finish_reason" in low:
            return "output.finalization"
        if "tokens_details" in low:
            return "response.usage_accounting"
        return "request.admission"
    if path.endswith("parser/test_conversation.py"):
        if class_name == "TestGetModelType":
            return "deployment.effective_configuration"
        if class_name in {"TestGenerateEmbeddingConvs", "TestGetFullMultimodalTextPrompt"} or "image" in low or "video" in low or "audio" in low:
            return "input.multimodal_binding"
        if class_name in {"TestConversationMethods", "TestGenerateChatConv"}:
            return "request.admission"
        return "input.chat_template_tokenization"
    if path.endswith("platforms/test_platform_interface.py"):
        if class_name in {"TestPlatformEnum", "TestCpuArchEnum"}:
            return None
        if "invalid" in low or "raises" in low or "not_found" in low or "unavailable" in low:
            return "deployment.configuration_admissibility"
        return "deployment.effective_configuration"
    if path.endswith("mem_cache/test_unified_radix_cache_unittest.py"):
        if "diagnostics" in low:
            return None
        if class_name == "TestUnifiedRadixComponentRegistryOverride":
            return "deployment.effective_configuration"
        if class_name == "TestUnifiedRadixCacheEagleHiCacheStorageKey":
            return "cache.hash_bigram_equivalence"
        if class_name == "TestUnifiedRadixCacheKVEvents":
            return "cache.event_payload_binding"
        if class_name == "UnifiedLRUListBoundedRefreshTest" or "lru" in low or "priority" in low:
            return "cache.eviction_priority"
        if "event" in low:
            return "cache.event_payload_binding"
        if "storage_key" in low or "prefix_hash" in low:
            return "cache.namespace"
        if "evict" in low and "result" in low:
            return "cache.eviction_outcome"
        if "match" in low or "hit" in low:
            return "cache.hit_accounting"
        return "cache.state_invariants"
    raise AssertionError((snapshot, path))


def _checks(node: ast.FunctionDef) -> list[tuple[int, str]]:
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assert) or (
            isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            and child.func.attr.startswith("assert")
        ):
            found.append((child.lineno, " ".join(ast.unparse(child).split())))
        elif isinstance(child, ast.With):
            for item in child.items:
                expr = item.context_expr
                if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "raises":
                    found.append((child.lineno, " ".join(ast.unparse(expr).split())))
    return sorted(set(found))


def reviewed_cases(key: tuple[str, str], source: Path, prior: set[tuple[str, str]]) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PATH_HASHES[key], source
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
            if (name, str(node.lineno)) in prior:
                continue
            unit = unit_for_test(key[0], key[1], class_name, node.name)
            checks = _checks(node)
            if unit is None or not checks:
                continue
            summary = (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]
            selected[name] = (
                unit,
                f"{summary}. Direct source checks under the test's constructed input: {'; '.join(expr for _, expr in checks)}",
                "round7_source_review",
            )
    assert len(selected) == EXPECTED_COUNTS[key], (key, len(selected))
    return selected
