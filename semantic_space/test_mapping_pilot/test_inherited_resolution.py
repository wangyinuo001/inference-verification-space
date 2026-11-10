"""Check inherited test methods against their real source definitions."""

import csv
import unittest

from resolve_inherited_methods import HERE, ROOTS, Resolver, main


@unittest.skipUnless(all((root / ("vllm" if name == "vllm" else "python/sglang")).is_dir() for name, root in ROOTS.items()), "full source checkouts required")
class InheritedResolutionTest(unittest.TestCase):
    def test_vllm_parser_base_method(self):
        resolver = Resolver(ROOTS["vllm"])
        matches = resolver.find(ROOTS["vllm"] / "tests/tool_parsers/test_deepseekv3_tool_parser.py",
                                "TestDeepSeekV3ToolParser", "test_empty_arguments")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].relative_to(ROOTS["vllm"]).as_posix(),
                         "tests/tool_parsers/common_tests.py")

    def test_sglang_mixin_method(self):
        resolver = Resolver(ROOTS["sglang"])
        matches = resolver.find(ROOTS["sglang"] / "test/registered/constrained_decoding/test_constrained_decoding.py",
                                "TestXGrammarBackend", "test_json_generate")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].relative_to(ROOTS["sglang"]).as_posix(),
                         "python/sglang/test/kits/json_constrained_kit.py")

    def test_full_batch(self):
        summary = main()
        self.assertEqual(sum(summary.values()), 1366)
        self.assertEqual(summary.get("resolved_unique", 0), 1366)
        with (HERE / "inherited-method-sources.tsv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            if row["status"] == "resolved_unique":
                self.assertTrue((ROOTS[row["snapshot"]] / row["source_path"]).is_file())
                self.assertGreater(int(row["source_line"]), 0)


if __name__ == "__main__":
    unittest.main()
