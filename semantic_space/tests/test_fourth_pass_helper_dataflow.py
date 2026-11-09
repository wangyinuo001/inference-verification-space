import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fourth_pass_helper_dataflow import links_for_test, returned_product  # noqa: E402
from third_pass_full_scan import definitions  # noqa: E402


def analyze(source):
    tree = ast.parse(source)
    tests, globals_, classes = definitions(tree)
    test = next(node for (name, _), node in tests.items() if name == "test_case")
    imports = {"compute": "vllm.module.compute"}
    return links_for_test(test, globals_, classes.get("", {}), imports)


class FourthPassTests(unittest.TestCase):
    def test_returned_product_result_reaches_caller_assertion(self):
        links = analyze("""
def helper(x):
    got = compute(x)
    return got
def test_case():
    value = helper(1)
    assert value == 2
""")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["product_target"], "vllm.module.compute")
        self.assertEqual(links[0]["helper_chain"], ["helper"])

    def test_two_helper_returns_are_followed(self):
        links = analyze("""
def inner(x):
    return compute(x)
def outer(x):
    return inner(x)
def test_case():
    assert outer(1) == 2
""")
        self.assertEqual(links[0]["helper_chain"], ["outer", "inner"])

    def test_reference_named_return_is_not_promoted(self):
        self.assertEqual(analyze("""
def helper(x):
    ref = compute(x)
    return ref
def test_case():
    assert helper(1) == 2
"""), [])

    def test_setup_helper_without_asserted_result_is_not_linked(self):
        self.assertEqual(analyze("""
def helper(x):
    return compute(x)
def test_case():
    helper(1)
    assert True
"""), [])

    def test_reassigned_caller_result_is_not_linked(self):
        self.assertEqual(analyze("""
def helper(x):
    return compute(x)
def test_case():
    value = helper(1)
    value = 3
    assert value == 3
"""), [])

    def test_ambiguous_returns_are_rejected(self):
        tree = ast.parse("""
def helper(x):
    if x:
        return compute(x)
    return 0
""")
        node = tree.body[0]
        self.assertIsNone(returned_product(node, {"compute": "vllm.module.compute"},
                                           {"helper": node}, {}))

    def test_assignment_after_return_is_not_a_result(self):
        self.assertEqual(analyze("""
def helper(x):
    return value
    value = compute(x)
def test_case():
    assert helper(1) == 2
"""), [])

    def test_mutated_result_is_not_a_direct_link(self):
        self.assertEqual(analyze("""
def helper(x):
    return compute(x)
def test_case():
    value = helper(1)
    value.clear()
    assert value == []
"""), [])

    def test_mutated_result_inside_helper_is_not_a_direct_link(self):
        self.assertEqual(analyze("""
def helper(x):
    value = compute(x)
    value.clear()
    return value
def test_case():
    assert helper(1) == []
"""), [])

    def test_helper_result_used_as_input_to_other_operation_is_not_linked(self):
        self.assertEqual(analyze("""
def helper(x):
    return compute(x)
def test_case():
    value = helper(1)
    assert transform(value) == 2
"""), [])

    def test_inline_nested_operation_is_not_linked(self):
        self.assertEqual(analyze("""
def helper(x):
    return compute(x)
def test_case():
    assert transform(helper(1)) == 2
"""), [])

    def test_comparator_may_directly_check_returned_result(self):
        links = analyze("""
def helper(x):
    return compute(x)
def test_case():
    value = helper(1)
    assert torch.equal(value, expected)
""")
        self.assertEqual(len(links), 1)


if __name__ == "__main__":
    unittest.main()
