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
            assert indexed["placement_status"] in {"linked_stage", "linked_domain_stage"}, name
            assert unit in units
            sites = oracle_lines(node)
            assert sites, (path, name)
            rows.append({
                "snapshot": snapshot, "path": path, "test_name": name, "line": str(node.lineno),
                "unit_id": unit, "property": prop,
                "oracle_source": ";".join(f"{path}:{line}" for line in sites),
                "conditions": "Pinned source snapshot; test inputs, parametrization, patches and fixtures at the linked test definition. This records a local assertion, not a model run or whole-unit proof.",
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
