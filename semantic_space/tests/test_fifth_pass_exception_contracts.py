import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fifth_pass_exception_contracts import exception_links  # noqa: E402


def scan(source):
    tree = ast.parse(source)
    test = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "test_case")
    return exception_links(test, {"compute": "vllm.module.compute",
                                  "other": "vllm.module.other"})


class ExceptionContractTests(unittest.TestCase):
    def test_single_product_call_in_raises(self):
        links = scan("""
def test_case():
    with pytest.raises(ValueError):
        compute(-1)
""")
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["status"], "direct_product_exception_link")
        self.assertEqual(links[0]["expected_exception"], "ValueError")

    def test_two_product_calls_are_ambiguous(self):
        links = scan("""
def test_case():
    with pytest.raises(ValueError):
        compute(-1)
        other(-1)
""")
        self.assertEqual(links[0]["status"], "ambiguous_exception_context")

    def test_setup_outside_raises_is_not_linked(self):
        self.assertEqual(scan("""
def test_case():
    compute(1)
    with pytest.raises(ValueError):
        unrelated(-1)
"""), [])

    def test_nested_exception_context_is_ambiguous(self):
        links = scan("""
def test_case():
    with pytest.raises(ValueError):
        with pytest.raises(TypeError):
            compute(-1)
""")
        self.assertEqual(links[0]["status"], "ambiguous_exception_context")

    def test_assert_raises_method(self):
        links = scan("""
def test_case(self):
    with self.assertRaises(RuntimeError):
        compute(-1)
""")
        self.assertEqual(links[0]["expected_exception"], "RuntimeError")

    def test_product_argument_to_other_operation_is_ambiguous(self):
        links = scan("""
def test_case():
    with pytest.raises(ValueError):
        other_operation(compute(-1))
""")
        self.assertEqual(links[0]["status"], "ambiguous_exception_context")

    def test_multiple_statements_in_context_are_ambiguous(self):
        links = scan("""
def test_case():
    with pytest.raises(ValueError):
        compute(-1)
        unrelated()
""")
        self.assertEqual(links[0]["status"], "ambiguous_exception_context")


if __name__ == "__main__":
    unittest.main()
