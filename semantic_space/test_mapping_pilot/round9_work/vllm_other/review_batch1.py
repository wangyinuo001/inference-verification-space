"""First source-reviewed vLLM batch: top-k kernel reference checks.

The name groups below are review decisions confined to one pinned source file.
They are not rules for classifying unseen tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATH = "tests/kernels/test_top_k_per_row.py"
REPO_SOURCE = HERE.parents[5] / "experiments/third_party/reference_repos/vllm-20261105"
FIELDS = ["snapshot", "path", "test_name", "line", "unit_id", "property", "oracle_source", "conditions", "rule_id"]

# In the pinned file, these entries invoke a kernel and compare output indices,
# selected values, or cache changes with an independently built expected result.
REFERENCE = {
    "test_top_k_per_row", "test_top_k_per_row_decode",
    "test_top_k_per_row_decode_large_vocab_size",
    "test_top_k_per_row_decode_gfx950_long_c4a",
    "test_top_k_per_row_decode_gfx950_long_c4a_1d_seq_lens",
    "test_top_k_per_row_decode_gfx950_k512_1d_seq_lens",
    "test_top_k_per_row_decode_gfx950_k512_masked_graph_replay",
    "test_aiter_c4a_prefill_topk_returns_sequence_local_indices",
    "test_aiter_c4a_decode_topk_uses_exact_mtp_lengths",
    "test_aiter_decode_topk_matches_per_row_under_mtp",
    "test_aiter_dsa_decode_topk_matches_per_row",
    "test_aiter_dsa_prefill_topk_matches_per_row",
    "test_deepseek_workspace_topk", "test_cooperative_topk_cs2",
    "test_workspace_topk_correctness", "test_workspace_topk_algorithm_paths",
    "test_workspace_topk_stress", "test_deepseek_topk_backends_no_error_and_reference",
    "test_cooperative_topk_rows_shorter_than_stride",
    "test_cooperative_topk_candidate_overflow",
    "test_cooperative_topk_overflow_cluster_boundary",
    "test_cooperative_topk_16bit_grid_overflow",
    "test_cooperative_topk_overflow_negative_halfway",
    "test_cooperative_topk_negative_inf_ties",
    "test_cooperative_topk_streaming_overflow",
    "test_cooperative_topk_overflow_staging_fallback",
    "test_cooperative_topk_overflow_special_bins",
    "test_cooperative_topk_overflow_mixed_lengths",
    "test_persistent_topk_bf16_immediate_cast",
    "test_workspace_topk_candidate_overflow", "test_workspace_topk",
    "test_persistent_topk_reused_group_after_short_row",
    "test_workspace_topk_padded_stride",
    "test_deep_select_topk", "test_deep_select_topk_preallocated_output",
    "test_sparse_indexer_decode_topk_explicit_backends",
    "test_sparse_indexer_resolved_persistent_bf16",
}

DISPATCH = {
    "test_aiter_c4a_topk_kernel_selection",
    "test_aiter_dsa_topk_kernel_selection",
    "test_dsv4_native_top_k_window",
    "test_sparse_indexer_topk_backend_resolution",
    "test_sparse_indexer_topk_aiter_requires_rocm",
    "test_sparse_indexer_topk_backend_resolution_aiter",
    "test_sparse_indexer_topk_aiter_applies_no_shape_gate",
    "test_sparse_indexer_topk_auto_stays_on_per_row",
    "test_sparse_indexer_topk_aiter_expands_mtp_row_ends",
}

OTHER = {
    "test_cooperative_topk_512_tie_workspace_is_per_row": "checks tie-specific selected-row behavior rather than a full output reference",
    "test_cooperative_topk_overflow_nan": "checks index range and uniqueness under NaNs; no complete reference comparison",
    "test_persistent_topk_masked_row_tail": "checks selected index range and uniqueness for masked tails, without an independent top-k result oracle",
    "test_decode_chunk_ranges_cover_rows_with_bounded_balanced_slices": "checks row-slice partition coverage and bounded balance",
    "test_decode_chunk_budget_disabled_fitting_and_invalid": "checks disabled/invalid decode chunk budget behavior",
}


def all_evidence(packet):
    stack = [packet.get("evidence")]
    while stack:
        ev = stack.pop()
        if not ev:
            continue
        yield ev
        stack.extend(ev.get("local_helpers", {}).values())


def main():
    entries = [json.loads(line) for line in (HERE / "source-packets.jsonl").open() if f'"{PATH}"' in line]
    entries = [e for e in entries if e["key"][1] == PATH]
    rows, pending = [], []
    for packet in entries:
        _, path, full_name, line = packet["key"]
        name = full_name.split("::")[-1]
        evs = list(all_evidence(packet))
        assert_refs = []
        for ev in evs:
            for c in ev.get("checks", []):
                assert_refs.append(f"{path}:{c['line']}")
        call_refs = []
        for ev in evs:
            for c in ev.get("calls", []):
                if any(k in c["name"] for k in ("top_k", "topk", "_run_topk_backend")):
                    call_refs.append(f"{path}:{c['line']}")
        if name in REFERENCE:
            if not assert_refs or not call_refs:
                pending.append([*packet["key"], "reference_group_missing_static_link", ";".join(call_refs), ";".join(assert_refs)])
                continue
            rows.append({
                "snapshot": "vllm", "path": path, "test_name": full_name, "line": line,
                "unit_id": "compute.operator_reference_relation",
                "property": "Top-k kernel selected indices or values agree with a per-row PyTorch top-k/reference backend under the test's row lengths, ties, dtype and backend; checks are limited to the asserted relation.",
                "oracle_source": ";".join(dict.fromkeys(assert_refs))[:1200],
                "conditions": "Kernel/backend and parameterized row lengths, top_k, dtype, masks and graph mode are taken from the pinned test and helper; each test's own source determines the checked subset.",
                "rule_id": "vllm_topk_source_review_20261112",
            })
        elif name in DISPATCH:
            pending.append([*packet["key"], "needs_compute_dispatch_unit", ";".join(dict.fromkeys(call_refs))[:1200], ";".join(dict.fromkeys(assert_refs))[:1200]])
        elif name in OTHER:
            pending.append([*packet["key"], OTHER[name], ";".join(dict.fromkeys(call_refs))[:1200], ";".join(dict.fromkeys(assert_refs))[:1200]])
        else:
            pending.append([*packet["key"], "source_review_exception: no batch1 rule applies", ";".join(dict.fromkeys(call_refs))[:1200], ";".join(dict.fromkeys(assert_refs))[:1200]])

    with (HERE / "crosswalk.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t");w.writeheader();w.writerows(rows)
    with (HERE / "pending.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t");w.writerow(["snapshot","path","test_name","line","reason","candidate_call_source","assertion_source"]);w.writerows(pending)
    sha = entries[0]["sha256"]
    evidence = {
        "batch": "top_k_per_row_source_review",
        "file_reviews": [{"snapshot":"vllm","path":PATH,"source_sha256":sha,"review":"Reviewed source-grounded test bodies and shared helpers compare_top_k_results, validate_topk_against_reference, _assert_exact_topk, run_large_context_topk_test, _assert_cooperative_topk_exact, _assert_topk_values_match_reference and _run_topk_backend. Per-entry assertion and top-k call source lines retained. Dispatch and row partition checks separated as pending.","product_source":["vllm/model_executor/layers/indexer_topk.py:141","vllm/model_executor/layers/indexer_topk.py:174","vllm/model_executor/layers/sparse_attn_indexer.py:354"]}],
        "reviewed_entries":len(rows),"pending_in_file":len(pending),
        "exceptions":[{"test_name":x[2],"reason":x[4]} for x in pending],
    }
    (HERE / "evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    needs = [{
        "id":"compute.kernel_dispatch_binding",
        "property":"在声明的硬件能力、算子输入形状、步长、精度和配置下，算子入口选择支持的实现后端，显式指定后端遵守其支持约束，自动后端选择遵守既定优先级与回退规则。",
        "oracle":"对照入口的后端解析规则与预期分支，检查选中的实现或拒绝结果；该检查不蕴含数值输出正确。",
        "vllm_source":["vllm/model_executor/layers/indexer_topk.py:174","tests/kernels/test_top_k_per_row.py:889","tests/kernels/test_top_k_per_row.py:2656"],
        "sglang_analogue":"未在本批源码核查；暂不宣称跨框架已确认。",
        "status":"proposed_needs_dual_framework_review"
    },{
        "id":"compute.chunk_partition_contract",
        "property":"按输入行数和预算生成的连续切片恰好覆盖有效行，切片不重叠且满足声明的容量约束；无效或关闭预算使用定义好的回退。",
        "oracle":"检查切片起止、相邻连接、覆盖与容量上界。",
        "vllm_source":["vllm/model_executor/layers/sparse_attn_indexer.py:354","tests/kernels/test_top_k_per_row.py:2941","tests/kernels/test_top_k_per_row.py:2959"],
        "sglang_analogue":"未在本批源码核查；暂不宣称跨框架已确认。",
        "status":"proposed_needs_dual_framework_review"
    }]
    (HERE / "needs-space.json").write_text(json.dumps(needs,ensure_ascii=False,indent=2))
    print(len(entries),len(rows),len(pending))


if __name__ == "__main__":
    main()
