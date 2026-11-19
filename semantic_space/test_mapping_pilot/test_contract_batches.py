import ast
import tempfile
import unittest
from pathlib import Path

from build_contract_batches import extract_function, make_card, parse_source


class ContractBatchTests(unittest.TestCase):
    def extract(self, text):
        return extract_function(ast.parse(text).body[0], 'test.py')

    def test_uncalled_nested_assertions_are_not_direct_checks(self):
        result = self.extract('def test_a():\n def helper():\n  assert False\n x = 1\n')
        self.assertEqual(result['checks'], [])

    def test_decorator_and_default_are_not_body_calls(self):
        result = self.extract('@wrap(assert_ready())\ndef test_a(x=assert_default()):\n assert x\n')
        self.assertEqual(len(result['checks']), 1)
        self.assertEqual(result['calls'], [])
        self.assertEqual(result['decorators'], ['wrap(assert_ready())'])

    def test_comparison_is_not_claimed_to_prove_integrity(self):
        result = self.extract('def test_a():\n torch.testing.assert_close(actual, ref, rtol=0.02)\n')
        self.assertEqual(result['checks'][0]['kind'], 'numerical_comparison')
        self.assertIn('rtol=0.02', result['checks'][0]['expression'])
        self.assertNotIn('unit_id', result)

    def test_exception_condition_and_branch_are_preserved(self):
        result = self.extract('def test_a(x):\n if x < 0:\n  with pytest.raises(ValueError):\n   f(x)\n')
        self.assertEqual(result['checks'][0]['kind'], 'expected_exception')
        self.assertEqual(result['guards'][0]['expression'], 'x < 0')

    def make_source(self, text):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name)/'test.py'
        p.write_text(text)
        return parse_source(p)

    def row(self, name, line):
        return dict(snapshot='vllm', path='test.py', test_name=name, line=str(line),
                    definition_path='test.py', definition_line=str(line),
                    review_status='review_call_inputs_and_property', unit_ids='',
                    property='', object_candidates='compute')

    def test_same_shape_different_constants_do_not_gain_review_status(self):
        source = self.make_source('def test_a():\n assert f(1) == 2\ndef test_b():\n assert f(3) == 8\n')
        a, b = [make_card(self.row(n, l), source) for n, l in [('test_a', 1), ('test_b', 3)]]
        self.assertEqual(a['batch_id'], b['batch_id'])
        self.assertNotEqual(a['evidence']['checks'][0]['expression'], b['evidence']['checks'][0]['expression'])
        self.assertEqual(b['existing_review_status'], 'review_call_inputs_and_property')
        self.assertEqual(b['reviewed_units'], '')

    def test_same_named_assertions_from_different_calls_separate(self):
        source = self.make_source('def test_a():\n assert a(1) == 2\ndef test_b():\n assert b(1) == 2\n')
        a, b = [make_card(self.row(n, l), source) for n, l in [('test_a', 1), ('test_b', 3)]]
        self.assertNotEqual(a['batch_id'], b['batch_id'])

    def test_fixture_shadowing_does_not_follow_module_helper(self):
        source = self.make_source('def helper():\n assert False\ndef test_a(helper):\n helper()\n')
        result = make_card(self.row('test_a', 3), source)
        self.assertEqual(result['evidence']['local_helpers'], [])

    def test_local_helper_checks_keep_their_own_location(self):
        source = self.make_source('def helper():\n assert result == 1\ndef test_a():\n helper()\n')
        result = make_card(self.row('test_a', 3), source)
        self.assertEqual(result['extraction_status'], 'local_helper_checks')
        self.assertEqual(result['evidence']['local_helpers'][0]['body']['checks'][0]['source'], 'test.py:2')


if __name__ == '__main__':
    unittest.main()
