"""Checks that broad placement never silently becomes property verification."""

import unittest

from build_stage_placement import route, stage_from_target


class StagePlacementTest(unittest.TestCase):
    def row(self, objects="cache"):
        return {
            "snapshot": "sglang", "path": "test/example.py", "test_name": "test_example",
            "line": "12", "object_candidates": objects, "oracle_forms": "comparison",
        }

    def review(self, status="", units=""):
        return {"source_review_status": status, "source_reviewed_units": units}

    def test_direct_link_needs_a_known_source_namespace(self):
        linked = {
            "bulk_status": "source_linked", "product_calls_json":
                '[{"target":"sglang.srt.mem_cache.radix_cache.RadixCache.evict"}]',
            "input_binding": "symbolic",
        }
        item = route(self.row("cache"), self.review(), linked)
        self.assertEqual(item["stage"], "cache")
        self.assertEqual(item["placement_status"], "linked_stage")
        self.assertEqual(item["property_review_status"], "")

    def test_multiple_domains_do_not_force_one_stage(self):
        item = route(self.row("cache,model"), self.review(), None)
        self.assertEqual(item["stage"], "")
        self.assertEqual(item["placement_status"], "multiple_stage_candidates")

    def test_conflicting_target_and_domain_waits_for_review(self):
        linked = {
            "bulk_status": "source_linked", "product_calls_json":
                '[{"target":"sglang.srt.mem_cache.radix_cache.RadixCache.evict"}]',
            "input_binding": "symbolic",
        }
        item = route(self.row("model"), self.review(), linked)
        self.assertEqual(item["stage"], "")
        self.assertEqual(item["placement_status"], "cross_stage_review")

    def test_direct_link_with_unknown_namespace_keeps_domain_as_candidate(self):
        linked = {
            "bulk_status": "source_linked", "product_calls_json":
                '[{"target":"sglang.lang.choices.token_length_normalized"}]',
            "input_binding": "symbolic",
        }
        item = route(self.row("sampling"), self.review(), linked)
        self.assertEqual(item["stage"], "sampling")
        self.assertEqual(item["placement_status"], "linked_domain_stage")
        self.assertEqual(item["stage_source"], "domain_with_direct_call_oracle_link")

    def test_adjacent_domain_is_not_a_request_stage(self):
        item = route(self.row("diagnostics"), self.review(), None)
        self.assertEqual(item["stage"], "")
        self.assertEqual(item["placement_status"], "adjacent_domain")

    def test_configuration_is_preflight_not_request_execution(self):
        item = route(self.row("configuration"), self.review(), None)
        self.assertEqual(item["stage"], "deployment_preflight")
        self.assertEqual(item["placement_status"], "domain_stage_candidate")

    def test_support_target_not_turned_into_direct_stage(self):
        self.assertEqual(stage_from_target(
            "sglang.srt.debug_utils.comparator.aligner.compute_plan"), "")


if __name__ == "__main__":
    unittest.main()
