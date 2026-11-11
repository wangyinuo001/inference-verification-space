"""Keep the on-demand explorer bundle aligned with reviewed rows."""

import json
import unittest

from sync_inherited_space_view import EXTRA, main


class InheritedSpaceViewTest(unittest.TestCase):
    def test_rows_match_explorer_and_sync_is_idempotent(self):
        before = EXTRA.read_bytes()
        self.assertEqual(main()["reviewed_inherited"], 891)
        self.assertEqual(EXTRA.read_bytes(), before)
        bundle = json.loads(before)
        inherited = bundle["inherited"]
        checked = bundle["inheritedProperty"]
        reviewed = bundle["reviewedProperty"]
        keys = {
            (row["snapshot"], row["test_path"],
             row["test_class"] + "::" + row["method"],
             str(row["test_class_line"]))
            for row in inherited
        }
        for row in checked:
            self.assertIn((row["snapshot"], row["path"], row["test_name"], row["line"]), keys)
        self.assertEqual(len(checked), 891)
        self.assertEqual(len(reviewed), 3957)


if __name__ == "__main__":
    unittest.main()
