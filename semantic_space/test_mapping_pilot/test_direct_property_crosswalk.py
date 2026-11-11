"""Check the reviewed direct properties against the fixed source snapshots."""

import csv
import unittest

from build_direct_property_crosswalk import CASES, OUTPUT, ROOTS, main


class DirectPropertyCrosswalkTest(unittest.TestCase):
    def test_reviewed_properties_have_real_oracle_sites(self):
        if not all((ROOTS[snapshot] / path).is_file() for snapshot, path in CASES):
            self.skipTest("Pinned source snapshots are outside the public preview repository")
        before = OUTPUT.read_bytes()
        self.assertEqual(main()["reviewed_direct"], 771)
        self.assertEqual(OUTPUT.read_bytes(), before)
        with OUTPUT.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(len(rows), 771)
        for row in rows:
            lines = (ROOTS[row["snapshot"]] / row["path"]).read_text(encoding="utf-8").splitlines()
            self.assertTrue(row["property"] and row["conditions"])
            self.assertTrue(row["oracle_source"])
            for site in row["oracle_source"].split(";"):
                path, line = site.rsplit(":", 1)
                self.assertEqual(path, row["path"])
                self.assertLessEqual(int(line), len(lines))


if __name__ == "__main__":
    unittest.main()
