"""Check the reviewed shared-method mappings and their source locations."""

import csv
import unittest

from build_inherited_property_crosswalk import HERE, RULES, main
from resolve_inherited_methods import ROOTS


class InheritedPropertyCrosswalkTest(unittest.TestCase):
    def test_reviewed_methods(self):
        self.assertEqual(main()["mapped"], 677)
        with (HERE / "inherited-property-crosswalk.tsv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(len(rows), 677)
        self.assertEqual(sum(row["rule_id"] == "gsm8k_score" for row in rows), 68)
        self.assertEqual(sum(row["rule_id"] == "ascend_gsm8k_score_throughput" for row in rows), 52)
        self.assertEqual(sum(row["rule_id"] == "capital_answer" for row in rows), 18)
        self.assertEqual(sum(row["rule_id"].startswith("json_") for row in rows), 33)
        self.assertEqual(sum(row["rule_id"].startswith("regex_") for row in rows), 72)
        self.assertEqual(sum(row["rule_id"].startswith("ebnf_") for row in rows), 63)
        self.assertEqual(sum(row["rule_id"].startswith("dsv4_") for row in rows), 74)
        self.assertEqual(sum(row["rule_id"] == "mmlu_score" for row in rows), 11)
        self.assertEqual(sum(row["rule_id"] == "spec_gsm8k_score" for row in rows), 16)
        self.assertEqual(sum(row["rule_id"].startswith("tool_") for row in rows), 64)
        self.assertEqual(sum(row["rule_id"].startswith("vlm_") for row in rows), 54)
        self.assertEqual(sum(row["rule_id"].startswith("stop_") or row["rule_id"] == "length_stop_metadata" for row in rows), 40)
        self.assertEqual(sum(row["rule_id"] in {"prefill_hit_kl", "decode_hit_kl", "prefix_branch_hit_count"} for row in rows), 35)
        self.assertFalse(any(row["test_name"].endswith("::test_json_invalid") for row in rows))
        self.assertFalse(any(row["test_name"].endswith("::test_ascii_ratio") for row in rows))
        self.assertFalse(any(row["test_name"].endswith("::test_malformed_input") for row in rows))
        self.assertFalse(any(row["test_name"].endswith("::test_mixed_batch") for row in rows))
        for row in rows:
            self.assertTrue(row["property"] and row["oracle_source"] and row["conditions"])
            for location in row["oracle_source"].split(";"):
                path, line = location.rsplit(":", 1)
                source = ROOTS[row["snapshot"]] / path
                if (ROOTS[row["snapshot"]] / ("vllm" if row["snapshot"] == "vllm" else "python/sglang")).is_dir():
                    self.assertTrue(source.is_file(), location)
                    self.assertLessEqual(int(line), len(source.read_text(encoding="utf-8").splitlines()))
        self.assertEqual(len(RULES), 56)


if __name__ == "__main__":
    unittest.main()
