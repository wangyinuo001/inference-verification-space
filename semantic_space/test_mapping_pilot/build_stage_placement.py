"""Place the complete test index on broad inference stages, with provenance.

This is a routing pass, not a claim that a test verifies a particular property.
Direct call-to-assertion links are distinguished from symbol and domain hints.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPACE = HERE.parent
KEY = ("snapshot", "path", "test_name", "line")
FIELDS = KEY + (
    "stage", "placement_status", "stage_source", "product_target",
    "oracle_form", "input_binding", "property_review_status",
)

# The broad stages describe observable request behavior, not framework modules.
# Multiple possible stages stay unresolved rather than being forced into one.
DOMAIN_STAGE = {
    "configuration": "deployment_preflight",
    "ingress": "request_entry", "scheduling": "scheduling",
    "cache": "cache", "transfer": "transport", "distributed": "distributed_execution",
    "model": "model_compute", "adapter": "model_compute", "sampling": "sampling",
    "output": "response", "service": "service_behavior",
    "cancellation": "cancellation", "memory": "resource_lifecycle",
    "quality": "task_quality",
}
ADJACENT_DOMAINS = {
    "diagnostics", "runtime_support", "data_structure",
    "training", "image_generation",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in KEY)


def stage_from_target(target: str) -> str:
    """Only explicit source namespaces with a clear request role are routed."""
    if target.startswith(("vllm.config.", "vllm.envs.", "sglang.srt.server_args.")):
        return "deployment_preflight"
    if target.startswith(("vllm.v1.core.sched.", "sglang.srt.managers.schedule_policy.",
                          "sglang.srt.managers.scheduler.")):
        return "scheduling"
    if target.startswith(("vllm.v1.core.kv_cache", "vllm.v1.kv_offload.",
                          "sglang.srt.mem_cache.")):
        return "cache"
    if target.startswith(("vllm.distributed.kv_transfer.",
                          "sglang.srt.disaggregation.")):
        return "transport"
    if target.startswith(("vllm.model_executor.", "vllm.v1.attention.",
                          "vllm.v1.worker.gpu.", "sglang.srt.layers.",
                          "sglang.jit_kernel.", "sglang.srt.lora.torch_ops.")):
        return "model_compute"
    if target.startswith(("vllm.multimodal.", "sglang.srt.multimodal.")):
        return "request_entry"
    if target.startswith(("vllm.tool_parsers.", "vllm.parser.",
                          "vllm.renderers.", "sglang.srt.parser.",
                          "sglang.srt.function_call.")):
        return "response"
    if target.startswith(("vllm.v1.sample.", "sglang.srt.sampling.")):
        return "sampling"
    if target.startswith(("sglang.srt.constrained.",)):
        return "response"
    return ""


def route(scan: dict[str, str], review: dict[str, str],
          linked: dict[str, str] | None) -> dict[str, str]:
    domains = {part.strip() for part in scan["object_candidates"].split(",") if part.strip()}
    stages = {DOMAIN_STAGE[part] for part in domains if part in DOMAIN_STAGE}
    if not stages and domains & ADJACENT_DOMAINS:
        fallback_status = "adjacent_domain"
    elif len(stages) == 1:
        fallback_status = "domain_stage_candidate"
    elif len(stages) > 1:
        fallback_status = "multiple_stage_candidates"
    else:
        fallback_status = "unresolved"
    result = {
        **{name: scan[name] for name in KEY},
        "stage": next(iter(stages)) if len(stages) == 1 else "",
        "placement_status": fallback_status,
        "stage_source": "object_candidates" if stages else "",
        "product_target": "", "oracle_form": scan["oracle_forms"],
        "input_binding": "", "property_review_status": review["source_review_status"],
    }
    if linked and linked["bulk_status"] == "source_linked":
        calls = json.loads(linked["product_calls_json"])
        if len(calls) == 1:
            target = calls[0]["target"]
            source_stage = stage_from_target(target)
            result["product_target"] = target
            result["input_binding"] = linked["input_binding"]
            if source_stage:
                if stages and source_stage not in stages:
                    result["stage"] = ""
                    result["placement_status"] = "cross_stage_review"
                    result["stage_source"] = "source_target_disagrees_with_domain"
                else:
                    result["stage"] = source_stage
                    result["placement_status"] = "linked_stage"
                    result["stage_source"] = "direct_product_result_to_assertion"
            elif len(stages) == 1:
                result["placement_status"] = "linked_domain_stage"
                result["stage_source"] = "domain_with_direct_call_oracle_link"
    if review["source_review_status"] == "mapped" and review["source_reviewed_units"]:
        result["property_review_status"] = "mapped"
    return result


def main() -> dict:
    scan = read(HERE / "full-scan-index.tsv")
    review = {key(row): row for row in read(SPACE / "oracle-review-queue.tsv")}
    links = {
        (row["snapshot"], row["path"], row["test_name"], row["test_line"]): row
        for row in read(SPACE / "bulk-call-oracle-links.tsv")
    }
    assert len(scan) == len(review) == 31968
    rows = [route(row, review[key(row)], links.get(key(row))) for row in scan]
    assert len(rows) == len({key(row) for row in rows})
    with (HERE / "stage-placement-all.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "entries": len(rows),
        "placement_status": dict(Counter(row["placement_status"] for row in rows)),
        "stages": dict(Counter(row["stage"] for row in rows if row["stage"])),
        "source_linked_stage": sum(row["placement_status"] in {"linked_stage", "linked_domain_stage"} for row in rows),
        "source_target_stage": sum(row["placement_status"] == "linked_stage" for row in rows),
        "property_mapped": sum(row["property_review_status"] == "mapped" for row in rows),
        "interpretation": "A stage is a coarse routing location. Only property_mapped has a reviewed specific verification property. Both linked statuses have a checked direct source call-to-assertion relation; linked_stage additionally has a matching source namespace, while linked_domain_stage uses a domain hint. Symbolic inputs remain unresolved where indicated.",
    }
    (HERE / "stage-placement-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
