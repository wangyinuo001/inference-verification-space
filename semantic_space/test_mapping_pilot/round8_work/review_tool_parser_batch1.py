"""Review pinned direct tool-parser assertions from 15 source files.

Registration and structure-info checks are excluded because their candidate
unit is not the asserted behavior. Argument value/type checks get the narrower
type-binding unit. This produces local review rows, not public green entries.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "../../experiments/third_party/reference_repos"
REPOS = {"vllm": ROOT / "vllm-20261105", "sglang": ROOT / "sglang-v0.5.14-complete"}
FIELDS = ("snapshot", "path", "test_name", "line", "unit_id", "property",
          "oracle_source", "conditions", "rule_id")

# Count is the frozen candidate count before the four explicit exclusions.
FILES = {
    ("sglang", "test/registered/unit/function_call/test_poolside_v1_detector.py"): (26, "fe2959ab5f34ee56eeee45fdf5aa15badaffe5023d2ae6eea5a180148dfe5eff"),
    ("vllm", "tests/tool_parsers/test_minicpm5xml_tool_parser.py"): (23, "ee6d8d7fc02972afdf5d5fa00860c734b7aff6156727937a00bd516d1e5e57d5"),
    ("vllm", "tests/tool_parsers/test_functiongemma_tool_parser.py"): (22, "502bdd8f4fc612f0d40bc1ea10aa2fa426d2178c10dd996abcdec3c1f6a8f9e3"),
    ("vllm", "tests/tool_parsers/test_minimax_m2_tool_parser.py"): (21, "825c803660bf998c185ed0251efa668816088571dd73c5d144cb69317ac20efb"),
    ("vllm", "tests/tool_parsers/test_apertus_tool_parser.py"): (20, "29ee1c7f5f74f7502959910e3183e8ed99a4036a4332f3792f6ddb48b9aa9ee9"),
    ("vllm", "tests/tool_parsers/test_kimi_k2_tool_parser.py"): (20, "9ea48e07d5f0d0bd9b5e629345d4834496a4b1bc5c0577e385587ce7197cc5f2"),
    ("vllm", "tests/tool_parsers/test_qwen3coder_tool_parser.py"): (19, "3dc12ac25545ae9d78c596c93ef2beb1d7fc7b440334c3bca1b17c521c83c746"),
    ("vllm", "tests/tool_parsers/test_step3p5_tool_parser.py"): (16, "c9001932e7734cc32c6e447a3852d774c7836da79fe7e42ffb8e698dd4c1edc3"),
    ("sglang", "test/registered/unit/function_call/test_minicpm5_detector.py"): (14, "c3b7f2961e72928e3b8886e7a44363dd7403e2eeb9738aa4fe716cda2111eb68"),
    ("sglang", "test/registered/unit/function_call/test_mistral_detector.py"): (14, "c221bef392ded955d49c89202109fa12a5411320f05a17269c62168d27f19f8e"),
    ("vllm", "tests/tool_parsers/test_hy_v3_tool_parser.py"): (13, "524e65e4252e3811ff2b593e10a191f2d794bdbfd1137a1da4255dac03389985"),
    ("sglang", "test/registered/unit/function_call/test_llama32_detector.py"): (12, "ad8023e02f046d6216e2edb96b17e2f07affa4094ace247cbdc56dc3bf95c4df"),
    ("vllm", "tests/tool_parsers/test_deepseekv4_tool_parser.py"): (12, "eb66ca0f9f9407d972c8905d974c4927b4a8b7da60aa88f651fe9ce3329fb932"),
    ("vllm", "tests/tool_parsers/test_lfm2_tool_parser.py"): (12, "cfbe580f402672171a99e5dec401d1f87551bc4fb1857a3ce3dd428899f2f905"),
    ("vllm", "tests/tool_parsers/test_glm47_moe_tool_parser.py"): (11, "02ed888cbb7ea634345c2ed96532ff323296e7f0122e6bc935d2fd2c46e03d31"),
}
EXCLUDE = {
    "TestPoolsideV1Detector::test_registered_in_function_call_parser",
    "TestPoolsideV1Detector::test_structure_info",
    "TestMistralDetector::test_structure_info",
    "TestLlama32Detector::test_structure_info",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_nodes(source: bytes) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result = {}
    for outer in ast.parse(source.decode("utf-8")).body:
        if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[outer.name] = outer
        elif isinstance(outer, ast.ClassDef):
            for node in outer.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{outer.name}::{node.name}"] = node
    return result


def unit_for(row: dict[str, str]) -> str:
    name = row["test_name"]
    class_name = name.split("::")[0] if "::" in name else ""
    if class_name in {"TestParseArguments", "TestTemplateArgumentTypes",
                      "TestAnyOfNullableParam", "TestNoneStringPreservation"}:
        return "output.tool_argument_type_binding"
    if name in {"test_non_string_types", "test_nullable_string_keeps_raw_text",
                "test_composed_schema_converts_object_and_array_params",
                "test_anyof_parameter_not_double_encoded", "test_no_double_serialization_string_args"}:
        return "output.tool_argument_type_binding"
    if row["proposed_unit"] == "output.tool_argument_type_binding":
        return "output.tool_argument_type_binding"
    assert row["proposed_unit"] == "tool.call_parse_relation", (row["path"], name)
    return "tool.call_parse_relation"


def main() -> dict[str, int]:
    candidates: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows(HERE / "candidates.tsv"):
        key = row["snapshot"], row["path"]
        if key in FILES:
            candidates.setdefault(key, []).append(row)
    prior = set()
    for name in ("confirmed-property-crosswalk.tsv", "direct-property-crosswalk.tsv",
                 "inherited-property-crosswalk.tsv"):
        prior.update((r["snapshot"], r["path"], r["test_name"], r["line"])
                     for r in rows(HERE.parent / name))
    reviewed = []
    excluded = []
    for (snapshot, path), (count, expected_hash) in FILES.items():
        source = (REPOS[snapshot] / path).read_bytes()
        assert hashlib.sha256(source).hexdigest() == expected_hash, path
        group = candidates[(snapshot, path)]
        assert len(group) == count, path
        nodes = test_nodes(source)
        for row in group:
            assert row["source_sha256"] == expected_hash
            name = row["test_name"]
            if name in EXCLUDE:
                excluded.append((snapshot, path, name))
                continue
            node = nodes[name]
            assert node.lineno == int(row["line"]), (path, name)
            key = snapshot, path, name, row["line"]
            assert key not in prior, key
            lines = {int(x) for x in row["oracle_lines"].split(";")}
            assert lines and all(node.lineno < line <= node.end_lineno for line in lines), key
            unit = unit_for(row)
            doc = ast.get_docstring(node)
            summary = doc.splitlines()[0] if doc else name.split("::")[-1].removeprefix("test_").replace("_", " ")
            reviewed.append({
                "snapshot": snapshot, "path": path, "test_name": name, "line": row["line"],
                "unit_id": unit,
                "property": summary + ". Direct source checks: " + row["checks"],
                "oracle_source": ";".join(f"{path}:{line}" for line in sorted(lines)),
                "conditions": "Pinned local parser input text, tool declaration, complete or streamed chunks and expected output at linked source; no model or server execution asserted.",
                "rule_id": "round8_tool_parser_batch1_source_review",
            })
    assert len(excluded) == len(EXCLUDE) and {name for _, _, name in excluded} == EXCLUDE
    assert len(reviewed) == 251
    keys = [(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in reviewed]
    assert len(keys) == len(set(keys))
    output = HERE / "reviewed-tool-parser-batch1.tsv"
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
