"""Regression checks for misleading test names and file-only source records."""
import csv
import json
import unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent

class ReviewContractTests(unittest.TestCase):
    def test_placeholder_tests_do_not_become_property_evidence(self):
        rows=json.loads((HERE/'round9_work/rust/reviewed-contracts.json').read_text())
        placeholders={'test_worker_selection','test_payload_too_large','test_pd_mode_routing','test_policy_registry_with_router_manager'}
        selected=[r for r in rows if r['test_name'] in placeholders and r['decision']=='incomplete_test_reviewed']
        self.assertEqual(len(selected),4)
        with (HERE/'reviewed-round9-crosswalk.tsv').open() as f:
            mapped={tuple(r[k] for k in ('snapshot','path','test_name','line')) for r in csv.DictReader(f,delimiter='\t')}
        for r in selected:
            self.assertNotIn(tuple(r[k] for k in ('snapshot','path','test_name','line')),mapped)

    def test_function_name_does_not_define_the_checked_property(self):
        rows=json.loads((HERE/'round9_work/rust/reviewed-contracts.json').read_text())
        cases={r['test_name']:r for r in rows}
        expected={
            'streaming_first_chunk_before_completion':'没有测量首块返回时间',
            'test_api_key_timing_attack_resistance':'没有计时',
            'test_tls_config_available':'普通 HTTP',
            'test_max_tool_calls_limit':'未超过限制',
            'test_cache_aware_consistent_routing':'没有检查节点 ID 相等',
        }
        for name,fragment in expected.items():
            self.assertIn(fragment,cases[name]['review_note'])

    def test_auxiliary_artifacts_are_not_counted_as_mapped_tests(self):
        rows=json.loads((HERE/'round9_work/file-record-roles.json').read_text())
        self.assertEqual(len(rows),93)
        self.assertTrue(all(not r['counted_as_completed_test_mapping'] for r in rows))
        self.assertEqual(sum(r['role']=='fixture_or_configuration' for r in rows),40)
        self.assertEqual(sum(r['role']=='generated_test_methods' for r in rows),2)
        self.assertFalse(any(r['role']=='file_purpose_pending' for r in rows))

if __name__=='__main__':unittest.main()
