"""Review smaller direct tool-parser families with pinned source and anchors."""

from __future__ import annotations

import ast
import csv
import hashlib
import json

from review_tool_parser_batch1 import FIELDS, REPOS, test_nodes, rows
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {
    ("vllm", "tests/tool_parsers/test_hermes_tool_parser.py"): (11, "52d967457a4af0487d58bb0e8f7ac739b82fc778a715a20def044553dc026476"),
    ("sglang", "test/registered/unit/function_call/test_hermes_detector.py"): (10, "0c53591f8e2dc46a810eb4394c9685ab0a46c5a5335cfb43c91ce5168cba57fd"),
    ("vllm", "tests/tool_parsers/test_dots_tool_parser.py"): (9, "9b97908c56b66d97c8febd425de4bdfe552ef13bf84a8a736e7bfb1788f02c0f"),
    ("vllm", "tests/tool_parsers/test_poolside_v1_tool_parser.py"): (8, "e2513c844a9e1e089b670e2f79e0b3e425387a0ab38d3f20cd5e146d9a986332"),
    ("vllm", "tests/tool_parsers/test_xlam_tool_parser.py"): (6, "ede55a4385c59b04e41be3605addbcfc5ef5fa91f7b7166be2545659d99f397c"),
    ("vllm", "tests/tool_parsers/test_glm4_moe_tool_parser.py"): (5, "bfa20eb64de279b9224d992a74490aa750d22274054fb95c0e3969b59ef6e264"),
    ("vllm", "tests/tool_parsers/test_jamba_tool_parser.py"): (5, "d125bd1e5529bc1f44981ca446229938afc91677e722437c71fb298a2b5d1192"),
    ("vllm", "tests/tool_parsers/test_minimax_m3_tool_parser.py"): (5, "f966956f1ee99f867cad6b683fa14590c1c0286509bc4700c6a0462b7f3877f2"),
    ("vllm", "tests/tool_parsers/test_llama4_pythonic_tool_parser.py"): (4, "6d98ae643b227a25e817f66693cb5461c8c9e03710cdc688c4cc96bf21bef280"),
    ("vllm", "tests/tool_parsers/test_olmo3_tool_parser.py"): (4, "9cc1da3c780b73d7316d894149f4a13efa02d14efc5e6b13d260386f30879563"),
    ("vllm", "tests/tool_parsers/test_deepseekv31_tool_parser.py"): (3, "e955c20803bf3557a8f205f329caaf5db3740c7b622491890657c033a725dcb2"),
    ("vllm", "tests/tool_parsers/test_ernie45_moe_tool_parser.py"): (3, "43d463a1a602133b605dc47a839dc96fb1f00f9b79045e19b8270d1204d60f3f"),
    ("vllm", "tests/tool_parsers/test_gigachat3_tool_parser.py"): (3, "a1e7d8c141e172444446925d46fd529b26371b80131dfc2d0b19a02e9ab791c5"),
    ("vllm", "tests/tool_parsers/test_granite_tool_parser.py"): (3, "d910e6ebdefbfd2ba3fa700fd835f9db9f302f3f03a3023d63a6153e7e6016ac"),
    ("sglang", "test/registered/unit/function_call/test_parallel_tool_calls.py"): (2, "747fb6cd4650251e4463bfc1734462c5a1a5eeda16f32482909e2981a40d9700"),
    ("sglang", "test/registered/unit/function_call/test_unknown_tool_name.py"): (2, "456b69c020b0e6c2fbfa0d4c3daa35355edf983e5de8bfdecbc89d537bd9038c"),
    ("vllm", "tests/tool_parsers/test_deepseekv3_tool_parser.py"): (2, "013d168ce7b1de5d9c5b3fbc92f9e2c62fcd10683dfbf0c4238a7ed6685583ba"),
    ("vllm", "tests/tool_parsers/test_hunyuan_a13b_tool_parser.py"): (2, "cd8b66b5799ce9a65e05312eefdea58904032514e4612a78a7cf963693a479c3"),
    ("vllm", "tests/tool_parsers/test_internlm2_tool_parser.py"): (2, "ac2d8f9651aa6214d1f3e82d2dd95bf591c661f58b766ed4cd8262982e5c65df"),
    ("vllm", "tests/tool_parsers/test_granite4_tool_parser.py"): (1, "48e0b98ad757b905f6da0a3733f10f82e071d2c50b3659cb7b2b4b03deafc5fe"),
    ("vllm", "tests/tool_parsers/test_schema_types.py"): (1, "c31b311e7f916f90a2d1db8af9b6a9c376f71979ee03868afd6493db82ce9629"),
}
EXCLUDED = {"TestHermesDetector::test_structure_info"}
TYPE_CHECKS = {"test_non_stream_xml_converts_schema_types_and_resolves_ref",
               "test_non_string_arg_still_deserialized", "test_nullable_string_keeps_raw_text",
               "test_hermes_streaming_boolean_args_with_stream_interval", "test_convert_value"}


def main() -> dict[str, int]:
    candidates = {}
    for row in rows(HERE / "candidates.tsv"):
        key = row["snapshot"], row["path"]
        if key in FILES:
            candidates.setdefault(key, []).append(row)
    reviewed = []
    excluded = []
    for (snapshot, path), (count, expected_hash) in FILES.items():
        source = (REPOS[snapshot] / path).read_bytes()
        assert hashlib.sha256(source).hexdigest() == expected_hash, path
        group = candidates[(snapshot, path)]
        assert len(group) == count, path
        nodes = test_nodes(source)
        for row in group:
            name = row["test_name"]
            assert row["source_sha256"] == expected_hash
            if name in EXCLUDED:
                excluded.append(name)
                continue
            node = nodes[name]
            assert node.lineno == int(row["line"]), (path, name)
            sites = {int(x) for x in row["oracle_lines"].split(";")}
            assert sites and all(node.lineno < line <= node.end_lineno for line in sites), (path, name)
            unit = "output.tool_argument_type_binding" if name in TYPE_CHECKS or row["proposed_unit"] == "output.tool_argument_type_binding" else "tool.call_parse_relation"
            assert row["proposed_unit"] in {"tool.call_parse_relation", "output.tool_argument_type_binding"}, name
            doc = ast.get_docstring(node)
            summary = doc.splitlines()[0] if doc else name.split("::")[-1].removeprefix("test_").replace("_", " ")
            reviewed.append({
                "snapshot": snapshot, "path": path, "test_name": name, "line": row["line"],
                "unit_id": unit, "property": summary + ". Direct source checks: " + row["checks"],
                "oracle_source": ";".join(f"{path}:{line}" for line in sorted(sites)),
                "conditions": "Pinned local tool parser or detector input, declared tools and complete/streamed chunk boundaries at linked source; no model/server execution asserted.",
                "rule_id": "round8_tool_parser_batch2_source_review",
            })
    assert excluded == list(EXCLUDED) and len(reviewed) == 90
    keys = [(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in reviewed]
    assert len(keys) == len(set(keys))
    output = HERE / "reviewed-tool-parser-batch2.tsv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(reviewed, key=lambda r: (r["snapshot"], r["path"], int(r["line"]))))
    result = {"reviewed_local_only": len(reviewed), "files": len(FILES),
              "excluded_misclassified": len(excluded), "published_green_added": 0}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
