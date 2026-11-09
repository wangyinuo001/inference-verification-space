"""Reconcile three manually reviewed test families into the semantic space.

This is review provenance, not a general test classifier. Input reviews retain
the original source-level findings; the output TSVs record explicit decisions
made after the space was extended.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent

HARMONY_UNITS = {
    "output.harmony_lexical_boundary": "output.structured_marker_boundary",
    "output.harmony_event_sequence": "output.channel_event_relation",
    "output.harmony_incremental_relation": "output.incremental_parse_relation",
    "output.harmony_format_selection": "output.parser_format_selection",
}

EVENT_LIFETIME = {
    "test_partial_tail_lookup_does_not_overwrite_store_metadata",
    "test_store_uses_snapshot_or_request_context_for_removal",
    "test_primary_removal_preserves_metadata_for_same_batch_kvcr_store",
    "test_pending_promotion_captures_payload_before_inventory",
    "test_failed_pending_retry_preserves_replacement_metadata",
    "test_pending_cpu_removal_consumes_hit_backfill_until_next_hit",
    "test_reoffload_preserves_secondary_residency",
    "test_take_events_supports_restore_after_eviction",
}

EVENT_CONFIGURATION = {
    "test_take_events_opt_out_keeps_placeholders",
    "test_tiering_accepts_self_describing_kv_events",
    "test_partial_tier_backpressure_inherits_top_level_defaults",
    "test_tier_override_wins_over_top_level",
    "test_tier_without_backpressure_stays_unconfigured",
}


def reconcile(source: str, target: str, mapping) -> dict[str, int]:
    src = HERE / source
    rows = list(csv.DictReader(src.open(newline="", encoding="utf-8"), delimiter="\t"))
    counts: dict[str, int] = {}
    for row in rows:
        decision = mapping(row)
        if decision:
            row["status"] = "mapped"
            row["unit_ids"] = decision
            row["applicability_note"] += " 经源码审查后映射到新增验证性质；原逐项记录保留在 " + source + "。"
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    with (HERE / target).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    harmony_detail = {
        row["name"]: row["proposed_unit"]
        for row in csv.DictReader(
            (HERE / "agent_sglang_harmony_detail.tsv").open(newline="", encoding="utf-8"),
            delimiter="\t",
        )
    }

    def harmony(row: dict[str, str]) -> str | None:
        if row["status"] != "needs_refinement":
            return None
        # This assertion checks only the return container type, not its event relation.
        if row["test_name"] == "TestAdditionalEdgeCases::test_canonical_return_without_final":
            return None
        return HARMONY_UNITS[harmony_detail[row["test_name"]]]

    def event(row: dict[str, str]) -> str | None:
        if row["status"] != "needs_refinement":
            return None
        name = row["test_name"]
        if name in EVENT_LIFETIME:
            return "cache.event_metadata_lifetime"
        if name in EVENT_CONFIGURATION:
            return "cache.event_configuration"
        return "cache.event_payload_binding"

    def eviction(row: dict[str, str]) -> str | None:
        if row["status"] != "needs_refinement":
            return None
        if row["test_name"].endswith("test_custom_threshold"):
            return "cache.eviction_priority"
        if row["test_name"].endswith("test_default_threshold_is_2"):
            return "cache.eviction_policy_configuration"
        raise ValueError(row["test_name"])

    result = {
        "sglang_harmony": reconcile(
            "agent_sglang_harmony_review.tsv",
            "sglang-harmony-reconciled-source-review.tsv",
            harmony,
        ),
        "vllm_offloading_events": reconcile(
            "agent_vllm_offloading_events_audit.tsv",
            "vllm-offloading-events-reconciled-source-review.tsv",
            event,
        ),
        "sglang_eviction_policy": reconcile(
            "agent_sglang_mem_evict_review.tsv",
            "sglang-eviction-policy-reconciled-source-review.tsv",
            eviction,
        ),
    }
    (HERE / "three-review-family-reconciliation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(result)


if __name__ == "__main__":
    main()
