import ast
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from third_pass_full_scan import (  # noqa: E402
    definitions, helper_clues, make_record, oracle_form,
)


class ThirdPassTests(unittest.TestCase):
    def test_oracle_form_does_not_call_shape_a_numerical_reference(self):
        self.assertEqual(oracle_form("assert output.shape == expected.shape"), "shape_or_size")
        self.assertEqual(oracle_form("torch.testing.assert_close(got, ref)"),
                         "numerical_closeness")
        self.assertEqual(oracle_form("with pytest.raises(ValueError):", "expected_exception"),
                         "expected_exception")

    def test_transitive_same_file_helper_is_only_a_clue(self):
        tree = ast.parse("""
from vllm.config import EngineArgs
def prepare():
    return EngineArgs()
def run():
    return prepare()
def test_case():
    x = run()
    assert x is not None
""")
        tests, globals_, _ = definitions(tree)
        helpers, targets = helper_clues(tests[("test_case", 7)], globals_, {},
                                        {"EngineArgs": "vllm.config.EngineArgs"})
        self.assertEqual(helpers, ["prepare", "run"])
        self.assertEqual(targets, ["vllm.config.EngineArgs"])

    def test_nested_unused_definition_does_not_create_path(self):
        tree = ast.parse("""
from vllm.config import EngineArgs
def prepare():
    return EngineArgs()
def test_case():
    def unused():
        prepare()
    assert True
""")
        tests, globals_, _ = definitions(tree)
        helpers, targets = helper_clues(tests[("test_case", 5)], globals_, {},
                                        {"EngineArgs": "vllm.config.EngineArgs"})
        self.assertEqual((helpers, targets), ([], []))

    def test_unbound_assertion_form_is_not_promoted_to_confirmed_property(self):
        row = {"snapshot": "vllm", "path": "tests/x.py", "test_name": "test_x",
               "line": "1", "candidate_area": "model", "source_review_verdict": "",
               "reviewed_area": "", "source_link_status": "no_direct_link"}
        item = {"source_assertions": [{"expression": "assert got.dtype == torch.float16",
                                       "kind": "assert"}]}
        result = make_record(row, item, None, (["prepare"], ["vllm.config.EngineArgs"]),
                             "source_parsed")
        self.assertEqual(result["oracle_forms"], "dtype")
        self.assertEqual(result["oracle_link_status"], "unbound_source_assertions")
        self.assertEqual(result["helper_relation"], "same_file_helper_to_product_candidate")
        self.assertEqual(json.loads(result["helper_product_targets"]),
                         ["vllm.config.EngineArgs"])


if __name__ == "__main__":
    unittest.main()
