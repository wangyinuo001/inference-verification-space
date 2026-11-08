import json
import unittest
from pathlib import Path


CATALOG = Path(__file__).resolve().parents[1] / "test_catalog.json"


class TestPinnedExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        cls.files = {
            (snapshot["id"], file["path"]): file
            for snapshot in catalog["snapshots"]
            for file in snapshot["files"]
        }

    def test_representative_local_placements(self):
        examples = [
            ("vllm", "tests/kernels/attention/test_flash_attn.py", "model_execution"),
            ("vllm", "tests/entrypoints/openai/responses/test_serving_responses.py", "output"),
            ("vllm", "tests/watermarking/test_watermarking.py", "sampling"),
            ("sglang", "test/registered/disaggregation/test_disaggregation_optimistic_prefill.py", "transfer"),
            ("sglang", "test/registered/jit/test_rmsnorm_hf.py", "model_execution"),
        ]
        for snapshot, path, expected_phase in examples:
            with self.subTest(path=path):
                phases = {area["phase"] for area in self.files[(snapshot, path)]["areas"]}
                self.assertIn(expected_phase, phases)

    def test_tests_outside_request_flow_remain_separate(self):
        examples = [
            ("sglang", "python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_flux_pipeline.py", "image_generation"),
            ("sglang", "test/registered/unit/test_no_bare_pytest_main.py", "diagnostics"),
        ]
        for snapshot, path, expected_area in examples:
            with self.subTest(path=path):
                areas = self.files[(snapshot, path)]["areas"]
                self.assertIn(expected_area, {area["id"] for area in areas})
                self.assertTrue(all(area["phase"] is None for area in areas))

    def test_operator_and_pooler_checks_are_model_computation(self):
        examples = [
            ("sglang", "test/registered/cpu/test_norm.py", "TestLayerNorm::test_norm_input_2d"),
            ("vllm", "tests/model_executor/layers/test_pooler_heads.py", "TestTokenEmbeddingPoolerHead::test_head_dtype"),
        ]
        for snapshot, path, name in examples:
            with self.subTest(path=path, name=name):
                item = next(item for item in self.files[(snapshot, path)]["items"] if item["name"] == name)
                self.assertEqual(item["classification"]["primary_area"], "model")

    def test_auth_check_in_general_utils_is_access_control(self):
        source = self.files[("sglang", "test/registered/unit/utils/test_auth.py")]
        item = next(item for item in source["items"]
                    if item["name"] == "TestDecideRequestAuth::test_normal_with_api_key_correct")
        self.assertEqual(item["classification"]["primary_area"], "security")
        self.assertEqual(item["classification"]["areas"][0]["phase"], "ingress")


if __name__ == "__main__":
    unittest.main()
