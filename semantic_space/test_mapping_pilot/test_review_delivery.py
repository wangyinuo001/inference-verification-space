"""Check that the delivered pages agree with the reviewed and observed records."""
import csv
import json
import re
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rows(name):
    with (HERE / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def key(row):
    return tuple(row[k] for k in ("snapshot", "path", "test_name", "line"))


class ReviewDeliveryTest(unittest.TestCase):
    def test_every_entry_is_accounted_for_and_source_links_are_consistent(self):
        index = rows("full-scan-index.tsv")
        delivery = rows("full_delivery/all-entries.tsv")
        self.assertEqual({key(r) for r in index}, {key(r) for r in delivery})
        self.assertEqual(len(index), len(delivery))
        self.assertEqual(json.loads((HERE / "full_delivery/consistency-issues.json").read_text()), [])
        self.assertTrue(all(r["source_sha256"] and r["review_status_label"] for r in delivery))

    def test_reviewed_and_pending_are_separate_and_all_units_are_visible(self):
        checked = rows("reviewed-supplement-crosswalk.tsv")
        pending = json.loads((HERE / "reviewed-supplement-pending.json").read_text())
        self.assertFalse({key(r) for r in checked} & {key(r) for r in pending})
        graph = (HERE.parent / "space_view.html").read_text()
        visible = set(re.findall(r'data-unit="([^"]+)"', graph))
        self.assertFalse({r["unit_id"] for r in checked} - visible)

    def test_remote_evidence_does_not_automatically_mark_entries_reviewed(self):
        execution = {key(r): r for r in rows("background_evidence/runtime-entries.tsv")}
        delivery = {key(r): r for r in rows("full_delivery/all-entries.tsv")}
        self.assertTrue(execution)
        self.assertTrue(any(delivery[k]["review_status"] != "property_reviewed" for k in execution))
        self.assertTrue(all(int(delivery[k]["runtime_instances"]) == int(r["instances"])
                            for k, r in execution.items()))


if __name__ == "__main__":
    unittest.main()
