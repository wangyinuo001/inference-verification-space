import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_test_catalog import (file_areas, py_test_items, request_route_candidates,
                                route_candidates, test_classification)  # noqa: E402

# This imported production helper is called by the tests below; it is not a
# pytest test function with fixtures named after its arguments.
test_classification.__test__ = False


class TestSourceCatalog(unittest.TestCase):
    def test_ci_cancellation_is_not_a_request_cancellation_route(self):
        path = ".github/workflows/scripts/test_run_ci_command.py"
        name = "test_ci_cancel_cancels_active_builds_for_pr_branch"
        item = {"name": name, "routes": route_candidates(path, name, ""),
                "source_calls": [], "source_call_count": 0}
        self.assertEqual(test_classification(path, item)["primary_area"], "diagnostics")
        self.assertEqual(request_route_candidates(item, test_classification(path, item)), [])
        request_path = "tests/v1/engine/test_abort.py"
        request_item = {"name": "test_cancel_running_request",
                        "routes": route_candidates(request_path, "test_cancel_running_request", ""),
                        "source_calls": [], "source_call_count": 0}
        self.assertIn("cancel_running_request", [route["route"] for route in
                      request_route_candidates(request_item, test_classification(request_path, request_item))])

    def test_extracts_test_and_assertion_without_expanding_parameters(self):
        source = '''
import pytest

@pytest.mark.parametrize("prefix", ["a", "b"])
def test_prefix_cache_hit(prefix):
    assert prefix
'''
        items, error = py_test_items("tests/cache/test_prefix_cache.py", source)
        self.assertIsNone(error)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["start_line"], 4)
        self.assertEqual(items[0]["line"], 5)
        self.assertTrue(items[0]["parameterized"])
        self.assertEqual(items[0]["assertion_count"], 1)
        self.assertEqual(items[0]["source_assertions"][0]["line"], 6)
        self.assertIn("prefix_reuse", [route["route"] for route in items[0]["routes"]])

    def test_records_direct_product_call_and_source_assertion_as_clues(self):
        source = '''
from sglang.srt.mem_cache.utils import get_hash_str as cache_hash

def test_hash_order():
    a = cache_hash([1, 2])
    b = cache_hash([2, 1])
    assert a != b
'''
        items, error = py_test_items("test/test_hash.py", source)
        self.assertIsNone(error)
        self.assertEqual(items[0]["source_call_count"], 2)
        self.assertEqual(items[0]["source_calls"][0]["target"],
                         "sglang.srt.mem_cache.utils.get_hash_str")
        self.assertEqual(items[0]["source_assertions"][0]["expression"], "assert a != b")

    def test_multimodal_position_order_is_an_input_check(self):
        source = '''
from vllm.multimodal.utils import argsort_mm_positions

def test_argsort_mm_positions(case):
    result = argsort_mm_positions(case["mm_positions"])
    assert result == case["expected_modality_idxs"]
'''
        items, error = py_test_items("tests/multimodal/test_utils.py", source)
        self.assertIsNone(error)
        placement = test_classification("tests/multimodal/test_utils.py", items[0])
        self.assertEqual(placement["primary_area"], "ingress")

    def test_per_test_object_uses_direct_call_before_shared_file_theme(self):
        source = '''
from sglang.srt.mem_cache.utils import get_hash_str
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

def test_hash():
    assert get_hash_str([1, 2])

def test_temperature():
    assert SamplingParams(temperature=0).temperature == 0
'''
        items, error = py_test_items("test/registered/unit/mem_cache/test_mixed.py", source)
        self.assertIsNone(error)
        objects = {item["name"]: test_classification("test/registered/unit/mem_cache/test_mixed.py", item)
                   for item in items}
        self.assertEqual(objects["test_hash"]["primary_area"], "cache")
        self.assertEqual(objects["test_temperature"]["primary_area"], "sampling")
        self.assertEqual(objects["test_temperature"]["basis"], "named_direct_product_call")
        self.assertTrue(all("ci_register" not in call["target"] for item in items
                            for call in item["source_calls"]))

    def test_specific_paths_do_not_turn_kernel_correctness_into_task_quality(self):
        item = {"name": "test_activation_correctness", "source_calls": [], "source_call_count": 0}
        actual = test_classification("test/registered/jit/test_activation.py", item)
        self.assertEqual(actual["primary_area"], "model")
        eval_item = {"name": "test_grok_accuracy", "source_calls": [], "source_call_count": 0}
        actual = test_classification("test/registered/amd/accuracy/test_grok.py", eval_item)
        self.assertEqual(actual["primary_area"], "quality")
        self.assertEqual(actual["basis"], "specific_path")

    def test_unknown_kernel_is_not_forced_onto_request_path(self):
        self.assertEqual(route_candidates("tests/kernels/test_rmsnorm.py", "test_mixed_dtype", "assert close"), [])

    def test_video_route_uses_test_identifier(self):
        routes = route_candidates("tests/entrypoints/test_video.py", "test_select_frames", "")
        self.assertIn("video_request", [route["route"] for route in routes])

    def test_file_path_placement_takes_precedence_over_incidental_names(self):
        items = [{"name": "test_input_correctness"}]
        areas = file_areas("test/registered/jit/test_rmsnorm_hf.py", items)
        self.assertEqual([area["id"] for area in areas], ["model"])

    def test_local_test_does_not_claim_request_route(self):
        areas = file_areas("tests/kernels/test_rmsnorm.py", [{"name": "test_mixed_dtype"}])
        self.assertEqual(areas[0]["phase"], "model_execution")
        self.assertEqual(route_candidates("tests/kernels/test_rmsnorm.py", "test_mixed_dtype", ""), [])

    def test_component_directory_beats_setup_calls(self):
        item = {"name": "test_attention_backend",
                "source_calls": [{"target": "vllm.v1.kv_cache_interface.FullAttentionSpec"}],
                "source_call_count": 1}
        actual = test_classification("tests/kernels/attention/test_backend.py", item)
        self.assertEqual(actual["primary_area"], "model")
        self.assertEqual(actual["basis"], "component_test_directory")

    def test_repository_ci_is_not_inference_sampling(self):
        item = {"name": "test_cdi_sampling", "source_calls": [], "source_call_count": 0}
        actual = test_classification(".buildkite/scripts/ci-otel/tests/test_ci_otel.py", item)
        self.assertEqual(actual["primary_area"], "diagnostics")

    def test_gateway_brand_does_not_imply_model_computation(self):
        item = {"name": "test_jwt_expiry", "source_calls": [], "source_call_count": 0}
        actual = test_classification("sgl-model-gateway/tests/security/auth_integration_test.rs", item)
        self.assertEqual(actual["primary_area"], "security")

    def test_local_sglang_import_is_recorded(self):
        source = '''
def test_choice():
    from sglang.lang.choices import greedy_token_selection
    assert greedy_token_selection({})
'''
        items, error = py_test_items("test/manual/lang_frontend/test_choices.py", source)
        self.assertIsNone(error)
        self.assertEqual(items[0]["source_calls"][0]["target"],
                         "sglang.lang.choices.greedy_token_selection")

    def test_unknown_source_is_explicit(self):
        item = {"name": "test_it", "source_calls": [], "source_call_count": 0}
        actual = test_classification("tests/test_unfamiliar.py", item)
        self.assertEqual(actual["status"], "needs_source_review")

    def test_special_case_name_does_not_imply_speculative_sampling(self):
        item = {"name": "TestSpecialCaseMixedChunk::test_mixed_chunk_with_running_batch",
                "source_calls": [], "source_call_count": 0}
        actual = test_classification("test/manual/chunked_prefill/test_scripted_special_case.py", item)
        self.assertEqual(actual["primary_area"], "scheduling")

    def test_metrics_collector_is_observability_even_when_scheduler_named(self):
        item = {"name": "test_engine_custom_scheduler_collector",
                "source_calls": [{"target": "sglang.Engine"}], "source_call_count": 1}
        actual = test_classification("test/registered/observability/test_metrics.py", item)
        self.assertEqual(actual["primary_area"], "service")

    def test_completion_prompt_limit_is_input_validation_not_cache_clear(self):
        item = {"name": "test_exact_limit_allowed", "source_calls": [
            {"target": "vllm.envs.__getattr__.cache_clear"},
            {"target": "vllm.entrypoints.openai.completion.protocol.CompletionRequest"}],
            "source_call_count": 2}
        actual = test_classification("tests/entrypoints/openai/completion/test_completion_error.py", item)
        self.assertEqual(actual["primary_area"], "ingress")

    def test_model_config_shapes_are_configuration(self):
        item = {"name": "TestModelConfigShapes::test_optional_head_dims_default_when_none",
                "source_calls": [], "source_call_count": 0}
        actual = test_classification("test/registered/unit/configs/test_model_config_shapes.py", item)
        self.assertEqual(actual["primary_area"], "configuration")

    def test_protocol_spec_distinguishes_request_response_and_combination(self):
        path = "sgl-model-gateway/tests/spec/rerank.rs"
        def classify(name):
            return test_classification(path, {"name": name, "kind": "rust_test",
                                              "source_calls": [], "source_call_count": 0})
        self.assertEqual(classify("test_rerank_request_validation_success")["primary_area"], "ingress")
        self.assertEqual(classify("test_rerank_result_serialization")["primary_area"], "output")
        self.assertEqual([a["id"] for a in classify("test_full_rerank_workflow")["areas"]],
                         ["ingress", "output"])
        response_path = "sgl-model-gateway/tests/spec/responses.rs"
        request_validation = test_classification(response_path, {
            "name": "test_validate_conversation_previous_response_mutual_exclusion",
            "kind": "rust_test", "source_calls": [], "source_call_count": 0})
        self.assertEqual(request_validation["primary_area"], "ingress")

    def test_speculative_directory_does_not_hide_cache_pool_test(self):
        item = {"name": "TestResolveSwaKvPool::test_active_pool_is_swa_returns_it",
                "source_calls": [], "source_call_count": 0}
        actual = test_classification("test/registered/unit/spec/test_resolve_swa_kv_pool.py", item)
        self.assertEqual(actual["primary_area"], "cache")

    def test_cuda_device_context_and_pcp_sharding_have_distinct_objects(self):
        item = {"name": "test_get_device_capability_uses_visible_device_ordinal",
                "source_calls": [], "source_call_count": 0}
        self.assertEqual(test_classification("tests/cuda/test_cuda_context.py", item)["primary_area"],
                         "configuration")
        item["name"] = "test_sharded_decode_piecewise_graph_padding"
        self.assertEqual(test_classification("tests/v1/worker/test_gpu_pcp_manager.py", item)["primary_area"],
                         "distributed")


if __name__ == "__main__":
    unittest.main()
