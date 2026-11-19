import unittest
from classify_all_contracts import classify, relation_kinds


class ClassificationTests(unittest.TestCase):
    def relation(self, expression, kind='predicate'):
        return relation_kinds({'expression': expression, 'kind': kind})

    def test_numeric_assertion_does_not_imply_system_integrity(self):
        self.assertEqual(self.relation('torch.testing.assert_close(a,b)', 'numerical_comparison'), {'numeric_tolerance'})

    def test_equality_with_message_does_not_check_message_shape(self):
        self.assertEqual(self.relation('assert value == 2, f"{unrelated.shape}"'), {'equality'})
        self.assertEqual(self.relation('self.assertEqual(value, 2, f"{unrelated.dtype}")', 'assertion_method'), {'equality'})

    def test_shapes_lengths_and_dtype_are_distinct(self):
        self.assertEqual(self.relation('assert output.shape == expected.shape'), {'equality','shape'})
        self.assertEqual(self.relation('assert len(result) >= 1'), {'cardinality','order_bound'})
        self.assertEqual(self.relation('assert result.dtype == torch.float16'), {'equality','dtype'})

    def test_assert_array_equal_is_not_tolerance(self):
        self.assertEqual(self.relation('np.testing.assert_array_equal(x, y)', 'numerical_comparison'), {'equality'})

    def test_mock_call_does_not_imply_real_execution(self):
        self.assertEqual(self.relation('mock.assert_called_once_with(x)', 'assertion_helper_call'), {'mock_interaction'})

    def test_custom_assertion_retains_uncertainty(self):
        self.assertEqual(self.relation('assert_close_enough(x,y)', 'assertion_helper_call'), {'helper_check'})

    def test_exception_regex_not_treated_as_observation(self):
        self.assertEqual(self.relation('self.assertRaisesRegex(ValueError, str(x.shape))', 'expected_exception'), {'exception'})

    def test_simple_predicate_never_promotes_review_status(self):
        card={'entry':['vllm','test.py','test_a','1'], 'source_sha256':'x',
              'definition_path':'test.py','definition_line':'1', 'extraction_status':'direct_checks',
              'evidence':{'checks':[{'source':'test.py:2','expression':'assert cache.size == 0','kind':'predicate'}]}}
        reviewed={'source_sha256':'x','review_status':'review_call_inputs_and_property','unit_ids':'','property':'','object_candidates':'cache'}
        result=classify(card,reviewed,{})
        self.assertEqual(result['disposition'],'source_relation_classified')
        self.assertEqual(result['reviewed_units'],'')


if __name__ == '__main__': unittest.main()
