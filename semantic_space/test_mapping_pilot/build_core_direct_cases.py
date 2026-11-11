"""Pinned source-level review of scheduler, cache, and KV transfer tests.

The functions select concrete assertions from the listed source definitions.
Their unit mapping is deliberately local to each subsystem; this file does
not infer a complete test suite or promote call-graph links by themselves.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PATH_HASHES = {
    "tests/v1/core/test_scheduler.py": "e3c5c3ea860b061fb0ae5fe58e0642b33bac27047992e7b0ed1a616c687de859",
    "tests/v1/core/test_prefix_caching.py": "27fe83c2dfc5135e2fbb937181846bce819312de635f3fb014e477b459487dbf",
    "tests/v1/core/test_kv_cache_utils.py": "a9d5b185efeff3fc50fe25905814c6cc0802794e03d91ca107502bee32be3d7d",
    "tests/v1/kv_connector/unit/test_mooncake_store_worker.py": "7f198d59520a4f307b21c74502d3c7af835ae05f5b2afe1898377d9e4170a39e",
    "tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py": "75a4107508c6b36eb921293c10800e32eed77aa830c4b46283c8d931b0ab312b",
    "tests/v1/kv_offload/tiering/p2p/test_sessions.py": "8f33525e3d125c81b2170bab0463aace20f175488f68873483b298b246c0d3da",
    "tests/v1/kv_offload/tiering/p2p/test_manager.py": "ec4379775cc16c2d976cd4f3f72097977a43c4a201609f12f9c450c10387ce66",
    "tests/v1/kv_connector/unit/test_nixl_connector.py": "fda7b1dd48be69109907ce919893527560a9fc6623ff33cb21da05b366cefef5",
    "tests/v1/kv_connector/unit/test_nixl_push_connector.py": "36094175334af6eabb4aaa43cc7c2560b31a94438e02aecb598b0bb4d4f2120b",
    "tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py": "cda446495ca98c302642744154b1a89705a3ad310d3dc2a97188d1c87d3ebd8c",
}
EXPECTED_COUNTS = {
    "tests/v1/core/test_scheduler.py": 131,
    "tests/v1/core/test_prefix_caching.py": 128,
    "tests/v1/core/test_kv_cache_utils.py": 67,
    "tests/v1/kv_connector/unit/test_mooncake_store_worker.py": 170,
    "tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py": 155,
    "tests/v1/kv_offload/tiering/p2p/test_sessions.py": 144,
    "tests/v1/kv_offload/tiering/p2p/test_manager.py": 114,
    "tests/v1/kv_connector/unit/test_nixl_connector.py": 74,
    "tests/v1/kv_connector/unit/test_nixl_push_connector.py": 71,
    "tests/v1/kv_connector/unit/test_mooncake_store_scheduler.py": 52,
}


def _checks(node: ast.FunctionDef) -> list[tuple[int, str]]:
    checks = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assert) or (
            isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
            and child.func.attr.startswith("assert")
        ):
            checks.append((child.lineno, " ".join(ast.unparse(child).split())))
        elif isinstance(child, ast.With):
            for item in child.items:
                expr = item.context_expr
                if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                        and expr.func.attr == "raises"):
                    checks.append((child.lineno, " ".join(ast.unparse(expr).split())))
    return sorted(set(checks))


def unit_for_test(path: str, class_name: str, name: str) -> str | None:
    low = name.lower()
    if path.endswith("core/test_scheduler.py"):
        if any(k in low for k in ("records_per_step", "subtracts_invalid_drafts", "acceptance_disabled_by_default")):
            return None
        if "stats" in low or "metrics" in low:
            return "cache.hit_accounting" if "prefix_cache" in low else None
        if "watermarking" in low:
            return "request.effective_sampling_parameters"
        if "sampling_mask" in low:
            return "request.batch_field_binding"
        if "stop" in low and ("check_stop" in low or "via_update" in low):
            return "output.finalization"
        if low in {"test_diffusion_canvas_width_defaults_to_the_served_canvas", "test_diffusion_scheduler_is_selected_by_default", "test_scheduler_no_ec_connector_by_default"}:
            return "deployment.effective_configuration"
        if "prefix_cache" in low and ("hit" in low or "query" in low):
            return "cache.hit_accounting"
        if "kv_cache_release" in low or "aux_output_reset" in low:
            return "cache.lifetime"
        return "schedule.progress"
    if path.endswith("core/test_prefix_caching.py"):
        if "reject" in low or "invalid" in low:
            return "deployment.configuration_admissibility"
        if "event" in low:
            return "cache.event_payload_binding"
        if "unhashed_block_ids" in low:
            return "cache.state_invariants"
        if any(k in low for k in ("hash", "salt", "namespace", "lora")):
            return "cache.namespace"
        if any(k in low for k in ("failed_host_import", "receive_completion", "inflight_host_import", "async_load")):
            return "disagg.transfer_gate"
        if low.startswith("test_hisparse") or low.startswith("test_host_"):
            return "cache.hit_accounting" if any(k in low for k in ("prefix_hit", "lookup", "core_hit")) else "cache.state_invariants"
        if any(k in low for k in ("free", "evict", "release", "retain", "recycl", "pin", "pool", "capacity", "resident")):
            return "cache.state_invariants"
        if any(k in low for k in ("hit", "lookup", "computed", "prefix", "prefill", "decode", "cache_blocks")):
            return "cache.hit_accounting"
        return "cache.state_invariants"
    if path.endswith("core/test_kv_cache_utils.py"):
        if low == "test_hisparse_steady_state_concurrency_excludes_spilled_resident_pages":
            return "deployment.effective_configuration"
        if "reject" in low:
            return "deployment.configuration_admissibility"
        if low == "test_null_head_len":
            return "cache.state_invariants"
        if "metrics" in low:
            return "cache.hit_accounting"
        if any(k in low for k in ("hash", "salt", "extra_key", "prompt_embed", "lora")):
            return "cache.namespace"
        if any(k in low for k in ("queue", "block", "page", "allocate")) and not any(k in low for k in ("spec", "config", "layout")):
            return "cache.state_invariants"
        if "transfer" in low or "group" in low:
            return "disagg.kv_mapping"
        return "deployment.effective_configuration"
    if path.endswith("test_mooncake_store_worker.py"):
        if "stats" in low or "metrics" in low or "logs_" in low:
            return None
        if low == "test_non_default_tenant_preserves_setup_type_error":
            return "deployment.configuration_admissibility"
        if low == "test_worker_is_hma_required_from_kv_cache_groups":
            return "deployment.effective_configuration"
        if "config" in low and any(k in low for k in ("reject", "invalid", "unknown")):
            return "deployment.configuration_admissibility"
        if low.startswith("test_create_mem_pool_rejects") or "group_id_detection_rejects" in low:
            return "deployment.configuration_admissibility"
        if low.startswith("test_create_mem_pool"):
            return "deployment.effective_configuration"
        if any(k in low for k in ("config", "preferred_segment", "local_buffer_size", "hostname", "device_name")) and "hash" not in low:
            return "deployment.effective_configuration"
        if "event" in low:
            return "cache.event_payload_binding"
        if low.startswith(("test_worker_init_excludes_nonprefix", "test_register_kv_caches_excludes_nonprefix", "test_recv_thread_keys_chunk", "test_hybrid_lookup_uses", "test_lookup_plan_")):
            return "disagg.kv_mapping"
        if low.startswith("test_lookup_key_prefixes"):
            return "cache.namespace"
        if low.startswith("test_blob_block_hashes") or low == "test_shared_prefix_junction_handoff_writes_only_its_mamba_key":
            return "disagg.kv_mapping"
        if low == "test_lookup_requires_all_dcp_rank_namespaces":
            return "cache.hit_accounting"
        if any(k in low for k in ("failure", "failed", "raises", "retry", "exception", "timeout", "wait", "oversized_key", "unsplittable_key")):
            return "disagg.transfer_gate"
        if any(k in low for k in ("namespace", "pool_key", "tenant_id", "blob_block_hashes")):
            return "cache.namespace"
        if any(k in low for k in ("close", "sleep", "release", "reuse", "stale", "pressure")):
            return "cache.lifetime"
        if any(k in low for k in ("lookup", "hit", "prefix")):
            return "cache.hit_accounting"
        return "disagg.kv_mapping"
    if path.endswith("offloading_connector/test_scheduler.py"):
        if low.startswith("test_scheduler_reports_lookup_") and "delay" in low:
            return None
        if class_name in {"TestMaximalPrefixLookup", "TestSlidingWindowLookup"}:
            return "cache.hit_accounting"
        if class_name == "TestEagle" and "store" not in low and "hash" not in low:
            return "cache.hit_accounting"
        if low == "test_max_offload_tokens_validation":
            return "deployment.configuration_admissibility"
        if "event" in low:
            return "cache.event_payload_binding"
        if "key" in low or "hash" in low:
            return "cache.namespace"
        if "fence" in low or "pin" in low or "release" in low:
            return "cache.lifetime"
        if "lookup" in low and any(k in low for k in ("defer", "pending", "delay", "inflight")):
            return "disagg.transfer_gate"
        if low == "test_max_offload_tokens_zero_does_not_record_pending_lookups":
            return "disagg.transfer_gate"
        if any(k in low for k in ("lookup", "hit", "cached")):
            return "cache.hit_accounting"
        if any(k in low for k in ("window", "mask", "replay", "retention")):
            return "cache.safe_reuse"
        return "disagg.transfer_gate"
    if path.endswith("p2p/test_sessions.py"):
        if class_name == "TestInflightPerReqInvariant" or any(k in low for k in ("pinned", "pins", "pin_", "drain_stores")):
            return "cache.lifetime"
        if "lookup" in low and "hit" in low:
            return "cache.hit_accounting"
        if any(k in low for k in ("request_id", "peer_id", "mismatched")):
            return "disagg.request_binding"
        return "disagg.transfer_gate"
    if path.endswith("p2p/test_manager.py"):
        if class_name == "TestInitHashSeed":
            return "cache.namespace"
        if class_name == "TestBackpressureRejected":
            return "deployment.configuration_admissibility"
        if class_name == "TestLookup":
            return "cache.hit_accounting"
        if class_name == "TestOnRequestFinished":
            return "cache.lifetime"
        if class_name == "TestAcceptNewPeers":
            return "distributed.remote_endpoint_binding"
        if class_name == "TestPromotionLatency":
            return None
        if class_name in {"TestPeerIdFromParams", "TestBindHostPortDefaults"}:
            return "distributed.remote_endpoint_binding"
        if class_name == "TestReconnectBackoff":
            return "disagg.transfer_gate"
        if class_name in {"TestDrainJobs", "TestShutdownDrain"}:
            return "distributed.queue_wait_shutdown"
        if class_name in {"TestRetainedTransfers", "TestIdleSessionEviction"} or any(k in low for k in ("pinned", "pins", "pin_")):
            return "cache.lifetime"
        if "lookup" in low and "hit" in low:
            return "cache.hit_accounting"
        return "disagg.transfer_gate"
    if path.endswith("test_nixl_connector.py"):
        if "stats" in low or "metrics" in low or "logging" in low or "warning" in low:
            return None
        if low == "test_basic_interface":
            return "disagg.request_binding"
        if any(k in low for k in ("compatibility_hash", "backend_params", "explicit_kv_role")):
            return "deployment.effective_configuration"
        if any(k in low for k in ("reject", "fails", "failure", "decode_errors")):
            return "disagg.transfer_gate"
        if any(k in low for k in ("layout", "descriptor", "shard", "tp_size", "head", "region", "memory_type", "replica", "topology")):
            return "disagg.kv_mapping"
        if any(k in low for k in ("register", "unregister", "cleanup", "sleep", "evict", "ttl", "resource")):
            return "cache.lifetime"
        return "disagg.transfer_gate"
    if path.endswith("test_nixl_push_connector.py"):
        if low == "test_p_side_mamba_truncates_before_cache_lookup":
            return "cache.hit_accounting"
        if any(k in low for k in ("reject", "failure", "failed")):
            return "disagg.transfer_gate"
        if any(k in low for k in ("layout", "descriptor", "region", "layer", "tp", "pp", "shard", "pairing", "metadata")):
            return "disagg.kv_mapping"
        if any(k in low for k in ("lease", "release", "expiry", "evict", "watchdog", "cleanup")):
            return "cache.lifetime"
        if "prefix_hit" in low:
            return "cache.hit_accounting"
        return "disagg.transfer_gate"
    if path.endswith("test_mooncake_store_scheduler.py"):
        if low == "test_worker_metadata_aggregates_completions_across_ranks":
            return "disagg.transfer_gate"
        if "event" in low:
            return "cache.event_payload_binding"
        if any(k in low for k in ("pin", "release", "preemption", "reuse")):
            return "cache.lifetime"
        if any(k in low for k in ("lookup", "hit")):
            return "cache.hit_accounting"
        if any(k in low for k in ("group", "block_ids", "projection", "boundary", "partial_tail", "rank")):
            return "disagg.kv_mapping"
        return "disagg.transfer_gate"
    raise AssertionError(path)


def reviewed_cases(path: str, source: Path, prior: set[tuple[str, str]]) -> dict[str, tuple[str, str, str]]:
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PATH_HASHES[path], source
    body = raw.decode("utf-8")
    selected = {}
    for outer in ast.parse(body).body:
        if isinstance(outer, ast.ClassDef):
            members, class_name = outer.body, outer.name
        elif isinstance(outer, ast.FunctionDef):
            members, class_name = [outer], ""
        else:
            continue
        for node in members:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
                continue
            name = f"{class_name}::{node.name}" if class_name else node.name
            if (name, str(node.lineno)) in prior:
                continue
            checks = _checks(node)
            unit = unit_for_test(path, class_name, node.name)
            if not checks or unit is None:
                continue
            summary = (ast.get_docstring(node) or node.name.replace("test_", "").replace("_", " ")).splitlines()[0]
            selected[name] = (
                unit,
                f"{summary}. Direct source checks for this test's fixtures: {'; '.join(expr for _, expr in checks)}",
                "core_source_review",
            )
    assert len(selected) == EXPECTED_COUNTS[path], (path, len(selected))
    return selected
