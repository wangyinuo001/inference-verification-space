"""Keep published inherited explorer details aligned with reviewed rows."""

import json
import re
import unittest

from sync_inherited_space_view import VIEW, main


class InheritedSpaceViewTest(unittest.TestCase):
    def test_embedded_rows_match_explorer_and_sync_is_idempotent(self):
        before = VIEW.read_bytes()
        self.assertEqual(main()["reviewed_inherited"], 891)
        self.assertEqual(VIEW.read_bytes(), before)
        body = before.decode("utf-8")

        def embedded(name):
            match = re.search(
                rf'<script type="application/json" id="{name}">(.*?)</script>',
                body, re.S,
            )
            self.assertIsNotNone(match)
            return json.loads(match.group(1))

        inherited = embedded("inherited-catalog")
        checked = embedded("inherited-property-crosswalk")
        reviewed = embedded("reviewed-property-crosswalk")
        keys = {
            (row["snapshot"], row["test_path"],
             row["test_class"] + "::" + row["method"],
             str(row["test_class_line"]))
            for row in inherited
        }
        for row in checked:
            self.assertIn((row["snapshot"], row["path"], row["test_name"], row["line"]), keys)
        self.assertEqual(len(checked), 891)
        self.assertEqual(len(reviewed), 2891)
        self.assertEqual(body.count('id="inherited-property-crosswalk"'), 1)
        self.assertEqual(body.count('id="reviewed-property-crosswalk"'), 1)


if __name__ == "__main__":
    unittest.main()
