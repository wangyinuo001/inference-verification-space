"""Check the reviewed direct properties against the fixed source snapshots."""

import csv
import unittest

from build_direct_property_crosswalk import CASES, OUTPUT, ROOTS, main
from build_core_direct_cases import unit_for_test


class DirectPropertyCrosswalkTest(unittest.TestCase):
    def test_reviewed_properties_have_real_oracle_sites(self):
        if not all((ROOTS[snapshot] / path).is_file() for snapshot, path in CASES):
            self.skipTest("Pinned source snapshots are outside the public preview repository")
        before = OUTPUT.read_bytes()
        self.assertEqual(main()["reviewed_direct"], 3713)
        self.assertEqual(OUTPUT.read_bytes(), before)
        with OUTPUT.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 3713)
        for row in rows:
            lines = (ROOTS[row["snapshot"]] / row["path"]).read_text(encoding="utf-8").splitlines()
            self.assertTrue(row["property"] and row["conditions"])
            self.assertTrue(row["oracle_source"])
            for site in row["oracle_source"].split(";"):
                path, line = site.rsplit(":", 1)
                self.assertEqual(path, row["path"])
                self.assertLessEqual(int(line), len(lines))

    def test_core_unit_routing_avoids_name_fragment_collisions(self):
        self.assertEqual(
            unit_for_test("tests/v1/kv_offload/tiering/p2p/test_manager.py", "TestHasPendingWork", "test_idle_manager_reports_no_pending_work"),
            "disagg.transfer_gate",
        )
        self.assertEqual(
            unit_for_test("tests/v1/kv_offload/tiering/p2p/test_sessions.py", "TestLookupFlow", "test_lookup_progress_keeps_overlapping_batches_alive"),
            "disagg.transfer_gate",
        )
        self.assertEqual(
            unit_for_test("tests/v1/kv_connector/unit/test_mooncake_store_worker.py", "", "test_config_from_file_rejects_non_string_tenant_id"),
            "deployment.configuration_admissibility",
        )
        self.assertEqual(
            unit_for_test("tests/v1/kv_connector/unit/test_nixl_connector.py", "TestNixlHandshake", "test_handshake_fails_on_kv_cache_layout_mismatch"),
            "disagg.transfer_gate",
        )


if __name__ == "__main__":
    unittest.main()
