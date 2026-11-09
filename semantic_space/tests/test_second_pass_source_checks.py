import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from second_pass_source_checks import (  # noqa: E402
    build, inspect, module_hints, reviewed_conflicts, source_calls_by_entry,
)


def row():
    return {"snapshot": "sglang", "path": "test/x.py", "test_name": "test_x",
            "line": "1", "candidate_area": "scheduling",
            "candidate_areas": "scheduling", "claim_level": "static_candidate",
            "next_action": "inspect"}


class SecondPassTests(unittest.TestCase):
    def test_linked_call_adds_alternative_without_overwriting(self):
        bulk = {"bulk_status": "source_linked", "input_binding": "symbolic",
                "product_calls_json": json.dumps([{"target": "sglang.srt.server_args.ServerArgs"}]),
                "oracle_expressions_json": json.dumps([{"link_kind": "assigned_result"}])}
        result = inspect(row(), [], bulk)
        self.assertEqual(result["candidate_area"], "scheduling")
        self.assertEqual(result["area_relation"], "linked_call_crosses_area_candidate")
        self.assertEqual(result["candidate_areas_v2"], "configuration,scheduling")

    def test_unlinked_setup_call_cannot_reclassify(self):
        result = inspect(row(), ["sglang.srt.server_args.ServerArgs"], None)
        self.assertEqual(result["area_relation"], "unlinked_call_crosses_area_candidate")
        self.assertEqual(result["review_action"], "separate_setup_call_from_checked_operation")

    def test_module_role_is_only_a_hint(self):
        self.assertEqual(module_hints(["vllm.distributed.kv_transfer.connector"]), ["transfer"])
        self.assertEqual(module_hints(["unresolved.module"]), [])

    def test_inherited_test_uses_source_calls_from_definition(self):
        inherited = {"snapshot": "vllm", "test_path": "tests/test_child.py",
                     "test_class": "TestChild", "test_class_line": "11",
                     "method": "test_inherited", "source_calls": json.dumps([
                         {"target": "vllm.distributed.kv_transfer.connector"}])}
        calls = source_calls_by_entry({"snapshots": []}, [inherited])
        self.assertEqual(calls[("vllm", "tests/test_child.py", "TestChild::test_inherited", "11")],
                         ["vllm.distributed.kv_transfer.connector"])

    def test_source_review_is_attached_without_rewriting_initial_candidate(self):
        reviewed = reviewed_conflicts([{"snapshot": "sglang", "path": "test/x.py",
                                        "test_name": "test_x", "line": "1",
                                        "review_result": "candidate_area_misaligned",
                                        "reviewed_area": "ingress",
                                        "checked_behavior": "input binding"}], [])
        result, summary = build([row()], {"snapshots": []}, [], reviews=reviewed)
        self.assertEqual(result[0]["candidate_area"], "scheduling")
        self.assertEqual(result[0]["reviewed_area"], "ingress")
        self.assertEqual(summary["entries_with_source_review"], 1)


if __name__ == "__main__":
    unittest.main()
