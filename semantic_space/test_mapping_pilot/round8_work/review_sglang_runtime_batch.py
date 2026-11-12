# -*- coding: utf-8 -*-
"""Reproduce the independent, source-checked SGLang runtime review batch.

Only the explicit decisions below produce confirmed mappings. Every audited
candidate retains its pinned source hash, body, and assertion evidence in JSON.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
PILOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/third_party/reference_repos/sglang-v0.5.14-complete"
FIELDS = ["snapshot", "path", "test_name", "line", "unit_id", "property", "oracle_source", "conditions", "rule_id"]


# Explicit, source-reviewed decisions. The test's particular checked relation
# is reported in property; a unit ID does not assert that one test proves the
# complete unit under every condition.
def decision(row: dict[str, str]) -> tuple[str, str, str]:
    file = Path(row["path"]).name
    name = row["test_name"].split("::")[-1]
    cls = row["test_name"].split("::")[0]

    if file == "test_modelopt_nvfp4_moe_scales.py":
        return ("compute.operator_reference_relation", "NVFP4 MoE scale helper output matches the asserted gate/up reference formula, independence relation, shape or FP32 dtype under the test's scale tensors.", "cpu_scale_helper")

    if file == "test_radix_cache_unit.py":
        if name == "test_extra_key_isolation":
            return ("cache.namespace", "Equal token sequences with distinct extra keys return their own stored indices; an absent extra key produces no match.", "radix_extra_key")
        if name in {"test_insert_and_match_basic", "test_advanced_prefix_match_with_node_splits", "test_page_alignment_boundary"}:
            return ("cache.hit_accounting", "The matched prefix and returned indices agree with inserted token prefixes and the tested page alignment, including splits where present.", "radix_prefix")
        if name in {"test_lock_ref_operations", "test_protect_release_host"}:
            return ("cache.lifetime", "Protect and release operations update protected and reclaimable reference state as asserted for the tested node.", "radix_reference")
        if name in {"test_reset", "test_total_size", "test_counter_increment", "test_insert_with_none_value", "test_all_values_flatten"} and cls != "TestRadixKey":
            return ("cache.state_invariants", "Local radix-tree count, node state, or stored-value relation equals the asserted state after the tested operation sequence.", "radix_state")
        if name in {"test_kv_cache_events", "test_hash_value_repeating_tokens"}:
            return ("cache.event_payload_binding", "Generated cache event or node hash metadata has the asserted block, parent, page, or lifecycle relation for the tested insertion/eviction sequence.", "radix_event")
        if name == "test_kv_cache_events_with_eviction":
            return ("", "", "The test asserts a BlockStored event, but iterates over BlockRemoved without asserting that any removal event exists.")
        return ("", "", "RadixKey representation or local hash accessor is not itself evidence for cache isolation or request-level reuse.")

    if file == "test_embedding_cache_controller.py":
        if cls == "TestContiguousMemoryAllocator":
            return ("cache.state_invariants", "The contiguous pool's allocated/free count and reusable offsets equal the asserted state after allocation and release.", "allocator_state")
        if name in {"test_evict_oldest_first", "test_protect_prevents_eviction", "test_evict_hashes_skips_protected"} or cls == "TestRDMAEvictionRacePrevention" and name != "test_evict_hashes_cleans_up_ref_counts":
            return ("cache.eviction_outcome", "The tested protected or older embedding entry is retained or evicted according to the asserted eviction outcome.", "embedding_eviction")
        if name == "test_access_time_updates_prevent_eviction":
            return ("cache.eviction_candidate_order", "After updating an entry's access time, the older entry is selected as eviction candidate and the touched entry is not.", "embedding_order")
        if name in {"test_ref_count_multiple_protects", "test_prefetch_sets_ref_count", "test_insert_batch_sets_ref_count", "test_get_embeddings_sets_ref_count", "test_evict_hashes_cleans_up_ref_counts"}:
            return ("cache.lifetime", "Embedding entry reference count and protection state follow the tested prefetch, insert, read, release or eviction sequence.", "embedding_reference")
        if name in {"test_io_loop_releases_prefetch_ref", "test_io_loop_releases_insert_ref"}:
            return ("", "", "The test manually invokes _release_hash after a mocked store operation; it does not execute the product _io_loop release path.")
        return ("", "", "The asserted missing-key return or tensor view/shape is narrower than the existing cache reuse, lifetime, and isolation units.")

    if file == "test_hicache_file_lru_unit.py":
        if cls in {"TestParseSize", "TestCPSuffix"}:
            if cls == "TestCPSuffix" and name.startswith("test_distinct"):
                return ("cache.namespace", "The same logical key maps to distinct file keys for distinct context-parallel ranks under the tested configuration.", "cp_key")
            return ("", "", "Size parsing or suffix formatting alone does not establish an existing request/cache verification property.")
        if cls in {"TestEnvDefaults", "TestEvictionDisabledByDefault", "TestMLAOwnerGating"}:
            return ("cache.eviction_policy_configuration", "The tested default, cap, or MLA rank configuration produces the asserted eviction enabled/owner behavior.", "file_cache_config")
        if cls == "TestScanExistingFiles":
            return ("cache.eviction_candidate_order", "Scanning existing files seeds the LRU queue in modification-time order and accounts for their bytes.", "file_cache_order")
        if cls == "TestCapBasedEviction":
            if name == "test_repeated_set_same_key_is_noop":
                return ("cache.state_invariants", "Repeated storage of the same key preserves the tracked count and single LRU entry.", "file_cache_state")
            return ("cache.eviction_outcome", "Under the tested capacity and watermark, the retained keys, removed keys and total bytes match the asserted eviction result.", "file_cache_eviction")
        if cls == "TestTrackOrTouch":
            return ("cache.state_invariants", "Access to an externally created cache file adds it to the tracked LRU and synchronizes the tracked byte count.", "file_cache_adoption")
        if cls == "TestPreReservationConcurrency":
            return ("cache.state_invariants", "Concurrent or pending file-cache writes preserve tracked-byte sum and cap; a pending reservation remains protected in the tested state.", "file_cache_concurrency")
        return ("", "", "No specific seed unit confirmed for this storage helper assertion.")

    if file == "test_hicache_staged_write_back_dispatch.py":
        if "backup_then_load_roundtrip" in name:
            return ("cache.tier_transfer_content", "For the tested host-pool layout, staged backup and load restore the original selected tensor values; transfer calls are asserted.", "staged_roundtrip")
        return ("", "", "This test checks host-index device and dispatch/call counts, not transferred content; a separate transfer-dispatch unit is needed.")

    if file == "test_ngram_corpus.py":
        if name in {"test_multi_sam_candidates", "test_remove_reduces_candidates"}:
            return ("spec.draft_candidate_binding", "The multi-corpus N-gram draft candidates change with corpus registration/removal as asserted for the tested token history.", "ngram_multi_corpus")
        return ("", "", "Corpus registration, deletion, HTTP wrapper, and capacity-management assertions do not check draft candidate binding.")

    if file == "test_reasoning_parser.py":
        if cls == "TestMiniMaxAppendThinkDetector":
            return ("", "", "Marker-prepending parser tests inspect returned text but not structured channel events; the current channel-event unit is too broad for this local parser relation.")
        return ("", "", "Constructor/default-field assertions do not check generated output parsing or channel contents.")

    if file == "test_sampling_batch_info.py":
        if cls in {"TestFilterBatch", "TestMergeBatch"}:
            return ("sampling.per_request_modifier_binding", "Filtering or merging batched sampling state preserves the asserted per-row parameter, bias, seed or custom processor assignment.", "sampling_batch_binding")
        return ("", "", "This assertion checks a local helper or logit-bias transform, but no existing seed unit isolates that transform from final sampling output.")

    if file == "test_penaltylib.py":
        if "filter_keeps_subset" in name or "merge_concatenates" in name:
            return ("sampling.per_request_modifier_binding", "Filtering or merging penalty state retains the asserted per-row penalty parameters and active masks.", "penalty_batch_binding")
        return ("", "", "This test checks a penalty/logit transformation, not the selected token required by sampling.policy; the seed lacks a local logit-transform unit.")

    if file == "test_mem_cache_utils.py":
        if name == "test_bigram_vs_flat_token_equivalent":
            return ("cache.hash_bigram_equivalence", "Flat and bigram encodings of the same token sequence produce identical cache hashes.", "hash_representation")
        return ("", "", "The assertion covers hash formatting, conversion or page splitting; it does not establish cache key isolation or reuse.")

    if file == "test_decode_radix_lock_ref.py":
        return ("cache.lifetime", "Decode-side KV transfer success, failure, budget recheck or repetition updates node locks and protected/reclaimable counts as asserted without leaking references.", "decode_lock_reference")

    if file == "test_swa_lock_release_lifecycle.py":
        if "evict_swa_leaf" in name:
            return ("cache.eviction_outcome", "SWA leaf eviction tombstones the tested node while preserving its full-window lock and expected size counters.", "swa_evict_lock")
        return ("cache.lifetime", "The tested SWA/full-window lock release sequence yields the asserted tombstone, free call and protected/evictable count state.", "swa_lock_lifecycle")

    if file == "test_swa_eviction_boundary.py":
        if name.startswith("test_formula"):
            return ("", "", "This test checks an eviction-boundary arithmetic helper without executing a cache eviction or match.")
        if name == "test_insert_case1_evicted_within_matched":
            return ("cache.hit_accounting", "After SWA eviction and insertion, the matched prefix length and retained indices match the asserted boundary case.", "swa_hit_boundary")
        return ("cache.eviction_outcome", "The tested SWA eviction boundary preserves or tombstones the asserted cache region under its page-size and lock state.", "swa_eviction_boundary")

    if file == "test_radix_force_miss.py":
        return ("cache.hit_accounting", "Forced cache miss reports zero reusable prefix and clears matched indices and node references under the tested request.", "forced_miss")

    if file == "test_streaming_session_unit.py":
        if "abort" in name:
            return ("abort.cleanup", "The tested aborted streaming session detaches or removes its session slot and releases the asserted request/KV resources.", "session_abort_cleanup")
        return ("", "", "Trimmed speculative overshoot is a session/KV boundary postcondition not covered by an existing seed unit at this granularity.")

    if file == "test_session_token_share_unit.py":
        return ("", "", "These assertions concern multi-turn token-history carry and rollback, not cache block ownership or KV lifetime; a session-history unit is needed.")

    if file == "test_asymmetric_mha_pool_host_unit.py":
        return ("", "", "The transfer function is mocked. Assertions inspect K/V pointers, strides, call count, or direct-layout rejection; they do not compare transferred tensor content. A transfer-dispatch/binding unit is needed.")

    if file == "test_unified_radix_hicache_dispatch.py":
        return ("", "", "These tests check implementation registry/factory ordering and sidecar wiring, not a declared cache state or request-level behavior unit.")

    return ("", "", "Not selected for this coherent source-review batch.")


def main() -> None:
    audit_files = {
        "test_modelopt_nvfp4_moe_scales.py", "test_radix_cache_unit.py",
        "test_embedding_cache_controller.py", "test_hicache_file_lru_unit.py",
        "test_hicache_staged_write_back_dispatch.py", "test_ngram_corpus.py",
        "test_reasoning_parser.py", "test_sampling_batch_info.py",
        "test_penaltylib.py", "test_mem_cache_utils.py",
        "test_decode_radix_lock_ref.py", "test_swa_lock_release_lifecycle.py",
        "test_swa_eviction_boundary.py", "test_radix_force_miss.py",
        "test_streaming_session_unit.py", "test_session_token_share_unit.py",
        "test_asymmetric_mha_pool_host_unit.py", "test_unified_radix_hicache_dispatch.py",
    }
    seen = set()
    for p in HERE.glob("reviewed-*.tsv"):
        if p.name in {"reviewed-sglang-runtime-batch.tsv", "reviewed-sglang-logit-transform-proposal.tsv"}:
            continue
        with p.open() as f:
            for x in csv.DictReader(f, delimiter="\t"):
                seen.add(tuple(x.get(k, "") for k in ("snapshot", "path", "test_name", "line")))
    for p in (PILOT / "direct").glob("*property-crosswalk.tsv"):
        with p.open() as f:
            for x in csv.DictReader(f, delimiter="\t"):
                seen.add(tuple(x.get(k, "") for k in ("snapshot", "path", "test_name", "line")))
    runtime = set()
    with (PILOT / "background_evidence/runtime-entries.tsv").open() as f:
        for x in csv.DictReader(f, delimiter="\t"):
            runtime.add(tuple(x[k] for k in ("snapshot", "path", "test_name", "line")))
    seed = json.loads((PILOT.parent / "space_seed.json").read_text())
    unit_ids = {u["id"] for u in seed["units"]}
    output = []
    evidence = []
    with (HERE / "candidates.tsv").open() as f:
        candidates = list(csv.DictReader(f, delimiter="\t"))
    for x in candidates:
        key = tuple(x[k] for k in ("snapshot", "path", "test_name", "line"))
        if x["snapshot"] != "sglang" or Path(x["path"]).name not in audit_files or key in seen or key not in runtime:
            continue
        source_path = SOURCE / x["path"]
        raw = source_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == x["source_sha256"], x["path"]
        lines = raw.decode().splitlines()
        oracle = [int(n) for n in x["oracle_lines"].split(";") if n]
        assert all(1 <= n <= len(lines) for n in oracle)
        tree = ast.parse(raw.decode())
        method = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == x["test_name"].split("::")[-1] and node.lineno == int(x["line"]):
                method = node
                break
        assert method is not None, key
        body = "\n".join(lines[method.lineno - 1:method.end_lineno])
        oracle_source = ";".join(f"{x['path']}:{n}" for n in oracle)
        unit, prop, rule = decision(x)
        if unit:
            assert unit in unit_ids
            assert oracle
            output.append({
                "snapshot": "sglang", "path": x["path"], "test_name": x["test_name"],
                "line": x["line"], "unit_id": unit,
                "property": prop + " Checked assertions: " + " | ".join(lines[n-1].strip()[:130] for n in oracle[:3]),
                "oracle_source": oracle_source,
                "conditions": "Pinned SGLang 0.5.14 source; local test inputs/mocks and asserted fields only. This mapping does not establish the entire unit across untested conditions.",
                "rule_id": "round8_sglang_runtime_source_review_" + rule,
            })
        evidence.append({
            "snapshot": x["snapshot"], "path": x["path"], "test_name": x["test_name"],
            "line": int(x["line"]), "source_sha256": digest,
            "test_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "test_body": body, "oracle_source": oracle_source,
            "oracle_snippets": [{"line": n, "text": lines[n-1].strip()} for n in oracle],
            "candidate_unit": x["proposed_unit"], "decision": "source_confirmed_local" if unit else "needs_refinement_or_unconfirmed",
            "reviewed_unit": unit or None, "review_reason": rule,
        })
    out_tsv = HERE / "reviewed-sglang-runtime-batch.tsv"
    with out_tsv.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(output)
    (HERE / "reviewed-sglang-runtime-batch-evidence.json").write_text(json.dumps({
        "source_root": str(SOURCE),
        "source_snapshot": "sglang-v0.5.14-complete",
        "reviewed": len(evidence), "confirmed_local": len(output),
        "decisions": evidence,
    }, ensure_ascii=False, indent=2) + "\n")
    # The following tests support a proposed, narrower unit. Keep them apart
    # until that unit is explicitly accepted into the public space model.
    transform_names = {
        "test_cumulate_and_apply", "test_cumulate_twice_doubles_penalty",
        "test_apply_when_not_prepared_is_noop", "test_presence_penalty_does_not_scale",
        "test_blocks_eos_before_min_tokens", "test_allows_eos_after_min_tokens",
        "test_blocks_additional_stop_tokens", "test_all_three_penalizers",
        "test_applies_additive_penalties", "test_applies_logit_bias",
        "test_no_bias_no_change",
    }
    proposal_evidence = [x for x in evidence if
        Path(x["path"]).name in {"test_penaltylib.py", "test_sampling_batch_info.py"}
        and x["test_name"].split("::")[-1] in transform_names
        and x["reviewed_unit"] is None]
    proposal_rows = []
    for x in proposal_evidence:
        snippets = [s["text"] for s in x["oracle_snippets"]]
        proposal_rows.append({
            "snapshot": x["snapshot"], "path": x["path"],
            "test_name": x["test_name"], "line": x["line"],
            "unit_id": "sampling.logit_transform_relation",
            "property": "For the pinned local input, the checked penalty, mask or bias transform gives the asserted logit values: " + " | ".join(snippets[:3]),
            "oracle_source": x["oracle_source"],
            "conditions": "Local CPU tensor and declared sampling parameters in pinned test; no claim about full processor ordering or sampled token.",
            "rule_id": "round8_sglang_proposed_local_logit_transform",
        })
    with (HERE / "reviewed-sglang-logit-transform-proposal.tsv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(proposal_rows)
    (HERE / "reviewed-sglang-logit-transform-proposal-evidence.json").write_text(json.dumps({
        "status": "pending_unit_acceptance", "proposed_unit": "sampling.logit_transform_relation",
        "source_snapshot": "sglang-v0.5.14-complete", "reviewed": len(proposal_evidence),
        "decisions": proposal_evidence,
    }, ensure_ascii=False, indent=2) + "\n")
    counts = Counter(x["reviewed_unit"] or "needs_refinement_or_unconfirmed" for x in evidence)
    md = ["# SGLang 运行入口源码审查", "", f"审查 {len(evidence)} 条运行观察到的候选源码入口；其中 {len(output)} 条有源码断言支持的局部性质映射，{len(evidence)-len(output)} 条不强行映射。运行记录只用于选取审查对象，不作为性质证据。", "", "## 按单元统计", ""]
    md += [f"- `{k}`：{n}" for k,n in counts.most_common()]
    md += ["", "## 空间缺口和候选误归类", "", "- `test_penaltylib.py` 的累积频率、presence 和最小新 token 惩罚检查是 logits 变换，现有 `sampling.policy` 描述最终采样 token；应有独立的采样前 logits/penalty 变换单元。", "- `test_hicache_staged_write_back_dispatch.py` 的部分测试只断言 staged/fallback 分派及 host index 所在设备，没有比较跨层缓存内容；需要独立的迁移路径选择单元。", "- `test_ngram_corpus.py` 的多语料增删、重复 ID、错误加载保留已有语料，是草稿语料生命周期/配置管理，不是草稿 token 与历史绑定。", "- `test_reasoning_parser.py` 的构造器默认值不证明输出通道解析；`test_mem_cache_utils.py` 的哈希格式和拆页也不证明请求间隔离。", "- `test/registered/unit/constrained/test_grammar_manager.py` 中语法编译对象的缓存，不是 KV 缓存；候选映射到 `cache.state_invariants` 是领域错位。", "", "## 证据与限制", "", "每条映射保留固定快照文件哈希、测试函数体哈希、断言行和断言文本。TSV 的 `property` 限定为测试实际检查的局部关系。模拟存储和 mocked CUDA transfer 的结论仅适用于模拟路径；本轮未据此推断真机传输或端到端结果。"]
    (HERE / "reviewed-sglang-runtime-batch.md").write_text("\n".join(md) + "\n")
    print(json.dumps({"audited": len(evidence), "confirmed_local": len(output), "counts": counts}, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
