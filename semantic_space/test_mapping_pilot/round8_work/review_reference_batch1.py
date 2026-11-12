"""Freeze the first file-by-file reviewed numerical comparison batch locally.

The five file families and their reference assertions were inspected together.
Output stays outside the published crosswalk until the one-release target is met.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
ROOT = HERE.parents[2] / "../../experiments/third_party/reference_repos"
REPOS = {"vllm": ROOT / "vllm-20261105", "sglang": ROOT / "sglang-v0.5.14-complete"}
FIELDS = ("snapshot", "path", "test_name", "line", "unit_id", "property",
          "oracle_source", "conditions", "rule_id")

# Each entry is a reviewed source family with a direct implementation/reference
# comparison in every chosen test body. Counts and hashes fail closed on drift.
FILES = {
    ("sglang", "test/registered/unit/layers/test_conv_layer.py"):
        (12, "19c67346233d7c562893c348081daf56691e2a25ef7d03b6d5cae636042d30c4",
         "Conv2d/Conv3d layer output versus an nn.Conv reference with copied weights."),
    ("vllm", "tests/kernels/attention/test_rocm_aiter_fa.py"):
        (9, "547a16004b7abc3ca1d69ff383c043ee480b4423b51a57d33e0869dbc349c2f1",
         "AITER attention output versus the paged attention reference for the specified ROCm shapes and cache layout."),
    ("vllm", "tests/kernels/quantization/test_cpu_fp8_scaled_mm.py"):
        (6, "d27c30bf8498f850d94d72cdce37ad16bc5e98e67c224dc6629fc6d4dacaf0a6",
         "CPU FP8 scaled matrix multiply output versus the floating reference for the specified quantization and bias settings."),
    ("vllm", "tests/kernels/fla/test_fla_layernorm_guard.py"):
        (9, "984ea550fd53ec24f43a27984cabf4e9221a9e0159c93eaa698f15897561910e",
         "FLA layer or RMS norm output versus the local reference for the specified shape, gate and dtype."),
    ("vllm", "tests/kernels/attention/test_triton_unified_attention.py"):
        (6, "7828b2e81c968148df227e83f42d16a6fb3c9e14a68d3537582fe265c428bb76",
         "Triton unified attention output versus the dense or independently computed reference for the specified cache and score settings."),
    ("vllm", "tests/kernels/moe/test_cpu_quant_fused_moe.py"):
        (15, "b0ee0346d6e7b39b320b24d2254c7256a05ea78a00ccf16332c2a6e2ba5261a7",
         "Quantized CPU fused MoE output versus the source's floating reference at the stated formats and expert settings."),
    ("vllm", "tests/kernels/moe/test_moe.py"):
        (9, "5cdd40a7fba5ee18865d526fb541475e1448bb530cb9443a144e8e7407ace28c",
         "Fused MoE output or expert reduction versus the source's reference for the stated routing and weight inputs."),
    ("vllm", "tests/kernels/quantization/test_marlin_tile_padding.py"):
        (9, "89bc94d5c1dd64419bff0898592d916ceecc2896b1ec9cd02574a7e46c5092c6",
         "Padded Marlin quantized matrix or MoE result versus the corresponding unpadded reference."),
    ("vllm", "tests/kernels/attention/test_rocm_triton_attn_dsv4.py"):
        (21, "0b168bd674d0248aac6a7698a88bdeca5cc4516431aec45c98740b7c838ae91c",
         "ROCm attention, attention index or destination buffer result versus the source's expected reference values."),
    ("vllm", "tests/kernels/quantization/test_fp8_quant.py"):
        (6, "16d495f9c282daacfa8f1d0a0a710acd8ae65371f0eb079cbfc3f672bc89a4e6",
         "FP8 quantization output and scale versus the source's reference values."),
    ("vllm", "tests/kernels/quantization/test_per_token_group_quant.py"):
        (7, "be5e49552e353b54d5d8630649026a01d1b6eb6bc430e2c00cb57fb1779cc839",
         "Per-token grouped quantized values or packed scales versus the source's reference values."),
    ("vllm", "tests/models/qwen4_exp/test_hc_ops.py"):
        (7, "3fc88b1bcd33ca74e263496987a523ee28c9d1aebedadfb96081cce4c27db690",
         "Qwen4 HC local operator output versus the source's independently constructed expected tensor."),
    ("vllm", "tests/models/test_beam_numerics.py"):
        (9, "48b65080878f151909abb93e914aab40ed7f63e29ec0aca9d5759d9401c45ed0",
         "Beam or attention local numerical operation output versus the source's reference values under the specified rounding and row conditions."),
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def test_nodes(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    result = {}
    for outer in tree.body:
        if isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[outer.name] = outer
        elif isinstance(outer, ast.ClassDef):
            for member in outer.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{outer.name}::{member.name}"] = member
    return result


def main() -> dict[str, int]:
    candidates = read(HERE / "reference-candidates.tsv")
    by_file: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in candidates:
        key = row["snapshot"], row["path"]
        if key in FILES:
            by_file.setdefault(key, []).append(row)
    prior = set()
    for name in ("confirmed-property-crosswalk.tsv", "direct-property-crosswalk.tsv",
                 "inherited-property-crosswalk.tsv"):
        prior.update((r["snapshot"], r["path"], r["test_name"], r["line"])
                     for r in read(PILOT / name))
    result = []
    for (snapshot, path), (count, expected_hash, description) in FILES.items():
        source = REPOS[snapshot] / path
        raw = source.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_hash, path
        rows = by_file[(snapshot, path)]
        assert len(rows) == count, (path, len(rows), count)
        nodes = test_nodes(ast.parse(raw.decode("utf-8")))
        for row in rows:
            key = snapshot, path, row["test_name"], row["line"]
            assert key not in prior, key
            node = nodes[row["test_name"]]
            assert node.lineno == int(row["line"]), key
            assertion_lines = {int(line) for line in row["oracle_lines"].split(";")}
            assert assertion_lines and all(node.lineno < line <= node.end_lineno for line in assertion_lines), key
            reference = row["reference_comparisons"]
            assert reference and all(part.strip() in row["checks"] for part in reference.split(" || ")), key
            result.append({
                "snapshot": snapshot, "path": path, "test_name": row["test_name"],
                "line": row["line"], "unit_id": "compute.operator_reference_relation",
                "property": description + " Direct source comparison: " + reference,
                "oracle_source": ";".join(f"{path}:{line}" for line in sorted(assertion_lines)),
                "conditions": "Pinned source test inputs, parameter values, hardware skip gates and numeric tolerances at the linked definition; local operator relation only, no model/server result asserted.",
                "rule_id": "round8_reference_batch1_source_review",
            })
    assert len(result) == 125
    keys = [(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in result]
    assert len(keys) == len(set(keys))
    output = HERE / "reviewed-reference-batch1.tsv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(result, key=lambda r: (r["snapshot"], r["path"], int(r["line"]))))
    summary = {"reviewed_local_only": len(result), "files": len(FILES),
               "published_green_added": 0, "output": str(output)}
    (HERE / "reviewed-reference-batch1-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
