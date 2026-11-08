"""Inventory source-level tests in the two pinned inference-framework snapshots.

This is a source catalog, not pytest collection. Parametrized test instances and
runtime fixtures require execution and are deliberately not counted as items.
Mappings are candidate routes inferred from source identifiers, not proof that a
test checks every verification obligation on a route.
"""

from __future__ import annotations

import ast
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from ast_scope import lexical_body_nodes


HERE = Path(__file__).resolve().parent
REPOS = HERE.parent / "third_party" / "reference_repos"
SNAPSHOTS = {
    "vllm": ("vllm-20261105", "vLLM"),
    "sglang": ("sglang-v0.5.14-complete", "SGLang 0.5.14"),
}

# A route is proposed only when its behavior appears in a test's own name,
# source path or referenced identifiers. These rules do not certify an oracle.
ROUTE_RULES = {
    "video_request": (r"\bvideo\b", r"\bframe(s)?\b"),
    "image_request": (r"\bimage(s)?\b", r"\bvision\b", r"\bmultimodal\b"),
    "disaggregated_generation": (r"\bdisagg(regated)?\b", r"\bremote kv\b", r"\bkv transfer\b", r"\bpd disagg\b"),
    "tensor_parallel_generation": (r"\btensor parallel\b", r"\btp size\b", r"\ball reduce\b", r"\ballreduce\b", r"\bcollective\b"),
    "speculative_greedy": (r"\bspeculative\b", r"\beagle\b", r"\bdraft token(s)?\b"),
    "embedding_request": (r"\bembedding(s)?\b", r"\bpooling\b"),
    "segmented_prefill": (r"\bchunked prefill\b", r"\bsegmented prefill\b", r"\bchunk prefill\b"),
    "prefix_reuse": (r"\bprefix cache\b", r"\bprefix caching\b", r"\bradix cache\b", r"\bprefix reuse\b"),
    "cancel_running_request": (r"\babort\b", r"\bcancel(lation)?\b", r"\binterrupt\b"),
    "structured_json_response": (r"\bstructured output(s)?\b", r"\bjson schema\b", r"\bgrammar\b", r"\bregex\b"),
    "adapter_switch": (r"\blora\b", r"\badapter(s)?\b"),
    "service_load": (r"\bthroughput\b", r"\blatency\b", r"\bbenchmark\b", r"\bperformance\b", r"\bserving bench\b"),
    "text_generation": (r"\btext generation\b", r"\bgenerate\b", r"\bgeneration\b", r"\bcompletion(s)?\b", r"\bdecode\b"),
}
COMPILED_RULES = {key: [re.compile(pattern) for pattern in patterns] for key, patterns in ROUTE_RULES.items()}
ROUTE_QUALITY_MARKERS = (r"\b(gsm8k|mmlu|hellaswag|gpqa|longbench|mmmu)\b",)

# These are source-location candidates for a local test, independent of whether
# it exercises a full request path. A null phase means the current request-flow
# graph has no truthful place for that kind of check.
AREA_RULES = [
    ("image_generation", "图像生成系统", None, r"\b(multimodal gen|diffusion|comfyui|flux pipeline|zimage)\b"),
    ("cancellation", "取消与清理", "abort", r"\b(abort\w*|cancel\w*|request cleanup|resource release)\b"),
    ("transfer", "跨实例传输", "transfer", r"\b(disagg\w*|kv connector|ec connector|remote kv|kv transfer|bootstrap)\b"),
    ("cache", "缓存与状态", "cache", r"\b(cache|caching|radix|kv offload|hicache|hisparse|block table|block free|kv canary|session control)\b"),
    ("distributed", "分布式执行", "model_execution", r"\b(distributed|all reduce|allreduce|expert parallel|context parallel|tensor parallel|parallelism|deep ep|deepep|eplb|rccl|nccl|multi gpu|tp[0-9]+|ep[0-9]+)\b"),
    ("adapter", "适配器执行", "model_execution", r"\b(lora|peft|adapter)\b"),
    ("sampling", "采样与约束", "sampling", r"\b(sampling|sampler|sampled token|greedy sample|temperature|logprobs?|grammar|structured|constrained|speculative|spec decode|eagle|draft|stop token|penalty|watermark|ngram|topk|vocab mask|jump forward|dry)\w*\b"),
    ("output", "输出与协议", "output", r"\b(output|response|stream|tool parser|tool call|function call|renderer|render|detoken|matched stop|parse delta|scoring|score api|vocab boundary finish|reasoning parser|template detection)\w*\b"),
    ("quality", "任务质量", "service", r"\b(gsm8k|mmlu|hellaswag|gpqa|longbench|mmmu|task quality|eval benchmark|lm eval)\w*\b"),
    ("security", "认证与访问控制", "ingress", r"\b(auth|authentication|authorization|api key|api token|jwt|jwks|mtls|security|access control)\w*\b"),
    ("service", "服务与性能", "service", r"\b(benchmark|bench|performance|latency|throughput|speed|observability|metrics|tracing|logging|uvicorn|health|stress|profiling|profile|gauge|histogram|timing stats|sanity)\w*\b"),
    ("scheduling", "调度与路由", "scheduling", r"\b(schedul|router|routing|supervisor|dispatch|load balanc|queue|placement|chunked prefill|prefill adder|batch overlap|pause generation)\w*\b"),
    ("ingress", "请求与输入", "ingress", r"\b(input|request|entrypoint|endpoint|openai|tokeniz|request parser|input parser|conversation|chat template|prompt processing|chat|completion|audio|speech|image|video|vision|media|multimodal|modality|mm positions|encode|binding|pooling params|npu api)\w*\b"),
    ("model", "模型计算与算子", "model_execution", r"\b(kernel|jit|attention|moe|quant|model|layer|model worker|linear|matmul|gemm|rmsnorm|norm|cudagraph|fxgraphcache|compile|cuda|triton|fp8|fp4|mamba|dsv4|qkv|activation|bmm|rope|rotary|flash attn|hidden state|expert|weights|vlm|xpu|gb300|mla|causal conv|marlin|fwd occupancy)\w*\b"),
    ("memory", "资源与内存", "cache", r"\b(allocator|memory|offload|prefetch|buffer sharing|shm|ipc|gc)\b"),
    ("configuration", "配置与部署", None, r"\b(config|arg util|cli|launch|deploy|environment|envs|ray env|install|setup|registry|platform|plugin|port|wheel|cmake|runai|runtime context|dynamic grad mode|log level|wake up|vertex endpoint)\w*\b"),
    ("diagnostics", "调试与测试基础设施", None, r"\b(debug|fixture|comparator|dumper|harness|precision baseline|test infra|bare pytest|embedded commit|release util)\w*\b"),
    ("runtime_support", "通用运行组件", None, r"\b(runtime support|async utility|collection utility)\b"),
    ("data_structure", "核心数据结构", None, r"\b(sequence|scalartype|tensor|event|indexer|hash|dataset|repetition|determinism|deterministic|invariance|batch invariant|inflight window)\w*\b"),
    ("training", "训练与 RL", None, r"\b(training|trainer|reinforcement|rl|reward|gradient|backward pass)\b"),
    ("unresolved", "待源码复核", None, r"(?!)"),
]
COMPILED_AREA_RULES = [(key, label, phase, re.compile(pattern)) for key, label, phase, pattern in AREA_RULES]
PATH_ONLY_RULES = (
    ("sampling", re.compile(r"gumbel_sample")),
    ("configuration", re.compile(r"(^|/)wasm_test\.rs$")),
    ("configuration", re.compile(r"(^|/)e2e_test/k8s_integration/|test_version\.py$|version_pin|test_server_args_backend|runner_init_contract")),
    ("diagnostics", re.compile(r"(^|/)tests/common/|(^|/)scripts/release/")),
    ("service", re.compile(r"(^|/)tests/reliability|(^|/)fault_tolerance/")),
    ("ingress", re.compile(r"(^|/)tests/api/|(^|/)tests/api_tests\.rs$|embedding_shape_validation|(^|/)multimodal/test_parse\.py$|short_mm_context|test_context_length")),
    ("output", re.compile(r"(^|/)e2e_test/embeddings/|test_serving_rerank")),
    ("cache", re.compile(r"kv_sharing_fast_prefill")),
    ("transfer", re.compile(r"mori_transfer_engine")),
    ("distributed", re.compile(r"test_ray_engine|test_torch_tp")),
    ("memory", re.compile(r"load_guard_raii")),
    ("scheduling", re.compile(r"test_engine_child_pids|test_srt_engine\.py$|(^|/)tests/v1/executor/")),
    ("model", re.compile(r"test_weight_version|test_replayssm_decode|(^|/)test/manual/test_(qwen|deepseek|kimi|gpt|llada)|(^|/)test/(manual|registered)/(amd|dllm|core)/test_(qwen|deepseek|kimi|gpt|llada)|(^|/)test/registered/ascend/basic_function/dllm/")),
    ("security", re.compile(r"(^|/)(security|auth)(/|$)")),
    ("ingress", re.compile(r"test_conversation\.py$|test_jinja_template_utils\.py$")),
    ("output", re.compile(r"(^|/)(parser|mcp_test)(/|\.)|(^|/)tests/parser/|test_score_engine")),
    ("output", re.compile(r"^test/registered/reasoning/")),
    ("model", re.compile(r"(^|/)tests/v1/worker/")),
    ("scheduling", re.compile(r"(^|/)tests/v1/engine/|test_multiproc_executor")),
    ("configuration", re.compile(r"test_load_snapshot_backends|test_crusoe_backend")),
    ("service", re.compile(r"(^|/)tests/reliability/")),
    ("model", re.compile(r"(^|/)(ir|ops|kernels)(/|$)|custom_4d_mask|test_fxgraphcache")),
    ("ingress", re.compile(r"(^|/)multimodal/test_processing\.py$|test_connections\.py$|test_pooling_params\.py$|test_npu_api\.py$|embed_overrides")),
    ("output", re.compile(r"(^|/)tool_use/|test_template_manager\.py$|test_multi_item_scoring\.py$|test_score_api\.py$|separate_reasoning|rust/src/llm/tests/generate\.rs$")),
    ("cache", re.compile(r"test_deferred_block_free|test_prefix_replay|test_hisparse_unit|test_session_control")),
    ("distributed", re.compile(r"test_data_parallel_controller|test_pp_single_node|test_comm\.py$")),
    ("scheduling", re.compile(r"test_two_batch_overlap|test_tbo_filter_batch_marker|test_prefill_adder")),
    ("scheduling", re.compile(r"multiproc_executor_timeout")),
    ("sampling", re.compile(r"(^|/)lang_frontend/test_choices\.py$")),
    ("transfer", re.compile(r"test_forward_split_prefill")),
    ("configuration", re.compile(r"test_envs\.py$|test_ray_env|test_runai_utils|test_precompiled_wheel|test_cmake_utils")),
    ("diagnostics", re.compile(r"test_logger\.py$|test_precision_baseline_store")),
    ("service", re.compile(r"test_uvicorn_logging")),
    ("quality", re.compile(r"(^|/)(evals?|accuracy)(/|$)")),
    ("service", re.compile(r"(^|/)perf(/|$)")),
    ("sampling", re.compile(r"(^|/)logits_processors(/|$)")),
    ("scheduling", re.compile(r"(^|/)chunked_prefill(/|$)")),
    ("sampling", re.compile(r"(^|/)(sample|samplers)(/|$)")),
    ("cache", re.compile(r"(^|/)kv_offload(/|$)")),
    ("diagnostics", re.compile(r"(^|/)(pre_commit|scripted_runtime|tools)(/|$)")),
    ("runtime_support", re.compile(r"(^|/)test/registered/(unit/)?utils(/|$)|(^|/)python/sglang/test(/|$)")),
    ("runtime_support", re.compile(r"(^|/)(utils_|transformers_utils)(/|$)")),
)
PRODUCT_PREFIX_AREAS = (
    ("sglang.lang.choices.", "sampling"),
    ("sglang.srt.parser.conversation.", "ingress"),
    ("sglang.srt.parser.harmony_parser.", "output"),
    ("vllm.parser.", "output"),
    ("vllm.engine.arg_utils.", "configuration"),
    ("sglang.srt.server_args.", "configuration"),
    ("vllm.models.", "model"),
    ("vllm.model_executor.", "model"),
    ("sglang.srt.models.", "model"),
    ("sglang.srt.layers.", "model"),
    ("sglang.srt.sampling.", "sampling"),
    ("vllm.v1.sample.", "sampling"),
    ("vllm.v1.kv_offload.", "cache"),
    ("sglang.srt.mem_cache.", "cache"),
)

# These directories are organized by the component under test. Their meaning
# is stronger than an incidental constructor or setup call in a test body.
STRONG_TEST_DIRECTORIES = (
    (r"^test/registered/unit/utils/test_auth\.py$", "security"),
    (r"^test/registered/cpu/test_norm\.py$", "model"),
    (r"^tests/model_executor/layers/test_pooler_heads\.py$", "model"),
    (r"^tests/cuda/test_cuda_context\.py$", "configuration"),
    (r"^tests/v1/worker/test_gpu_pcp_manager\.py$", "distributed"),
    (r"^tests/utils_/test_numa_utils\.py$", "configuration"),
    (r"^tests/utils_/", "runtime_support"),
    (r"^tests/jit_monitor/", "diagnostics"),
    (r"^tests/v1/worker/test_gpu_profiler\.py$", "service"),
    (r"^rust/src/bench/tests/cli_parity\.rs$", "diagnostics"),
    (r"^tests/entrypoints/launchers/api_server/test_api_server_process_manager\.py$", "configuration"),
    (r"^test/registered/unit/utils/test_weight_checker\.py$", "diagnostics"),
    (r"^test/registered/unit/utils/", "runtime_support"),
    (r"^test/registered/unit/test_eval_accuracy_kit_sgl_eval\.py$", "diagnostics"),
    (r"^tests/v1/core/test_contiguous_kv_packing\.py$", "cache"),
    (r"^test/registered/unit/spec/test_resolve_swa_kv_pool\.py$", "cache"),
    (r"^test/registered/unit/spec/test_fast_prefill_plan\.py$", "model"),
    (r"^test/registered/unit/spec/test_decode_bookkeeping_ownership\.py$", "scheduling"),
    (r"^test/registered/spec/", "sampling"),
    (r"^test/registered/unit/spec/", "sampling"),
    (r"^test/manual/chunked_prefill/", "scheduling"),
    (r"^test/registered/observability/", "service"),
    (r"^test/registered/unit/configs/", "configuration"),
    (r"^tests/lora/", "adapter"),
    (r"^tests/renderers/inputs/", "ingress"),
    (r"^tests/entrypoints/openai/completion/test_completion_error\.py$", "ingress"),
    (r"^test/manual/eval/", "quality"),
    (r"^tests/platforms/", "configuration"),
    (r"^tests/test_config\.py$", "configuration"),
    (r"^tests/compile/test_config\.py$", "configuration"),
    (r"^tests/compile/", "model"),
    (r"^tests/entrypoints/rl/", "training"),
    (r"^python/sglang/multimodal_gen/test/layers/", "model"),
    (r"^python/sglang/multimodal_gen/test/", "image_generation"),
    (r"^python/sglang/multimodal_gen/", "image_generation"),
    (r"^test/registered/kv_canary/", "service"),
    (r"^tests/basic_correctness/memory/", "memory"),
    (r"^tests/model_executor/test_sleep_mode_tensor_ownership\.py$", "memory"),
    (r"^tests/v1/spec_decode/", "sampling"),
    (r"^tests/compile/passes/", "model"),
    (r"^tests/v1/cudagraph/", "model"),
    (r"^test/registered/unit/managers/test_data_parallel_controller\.py$", "distributed"),
    (r"^python/sglang/multimodal_gen/test/unit/.*/test_pipeline_config\.py$", "configuration"),
    (r"^sgl-model-gateway/bindings/python/tests/test_startup_sequence\.py$", "configuration"),
    (r"^tests/entrypoints/cohere/test_serving_conversion\.py$", "output"),
    (r"^tests/v1/kv_connector/", "transfer"),
    (r"^tests/v1/ec_connector/", "transfer"),
    (r"^tests/v1/kv_offload/", "cache"),
    (r"^tests/entrypoints/serve/utils/test_sse", "output"),
    (r"^test/registered/unit/managers/test_io_struct\.py$", "ingress"),
    (r"^test/registered/ascend/basic_function/parameter/test_npu_no_overlap_scheduler\.py$", "scheduling"),
    (r"^test/registered/utils/", "runtime_support"),
    (r"^tests/renderers/chat_utils/", "ingress"),
    (r"^tests/entrypoints/scale_out/token_in_token_out/test_generate_stream\.py$", "output"),
    (r"^tests/entrypoints/launchers/test_launch_cli\.py$", "configuration"),
    (r"^tests/models/multimodal/processing/", "ingress"),
    (r"^tests/models/", "model"),
    (r"^tests/kernels/", "model"),
    (r"^tests/quantization/", "model"),
    (r"^tests/distributed/", "distributed"),
    (r"^tests/config/", "configuration"),
    (r"^tests/v1/attention/", "model"),
    (r"^tests/v1/sample/", "sampling"),
    (r"^tests/tool_parsers/", "output"),
    (r"^tests/parser/", "output"),
    (r"^test/registered/debug_utils/", "diagnostics"),
    (r"^test/registered/attention/", "model"),
    (r"^test/registered/jit/", "model"),
    (r"^test/registered/unit/function_call/", "output"),
    (r"^test/registered/unit/parser/test_conversation\.py$", "ingress"),
    (r"^test/registered/unit/parser/", "output"),
    (r"^test/registered/unit/sampling/", "sampling"),
    (r"^test/registered/unit/mem_cache/", "cache"),
    (r"^python/sglang/multimodal_gen/test/unit/test_server_args\.py$", "configuration"),
)
COMPILED_STRONG_DIRECTORIES = [(re.compile(pattern), area) for pattern, area in STRONG_TEST_DIRECTORIES]
NAME_OVERRIDE_RULES = {
    "sampling": re.compile(r"\b(temperature|sampl(e|er|ing)|logprob|grammar)\w*\b"),
}


def infrastructure_area(path: str) -> str | None:
    """Repository CI and helper tests are outside the inference request flow."""
    if path.startswith(".buildkite/lm-eval-harness/"):
        return None
    if path.startswith((".buildkite/", ".github/")) or "/.github/" in path:
        return "diagnostics"
    if path.startswith(("scripts/ci/", "tests/tools/")):
        return "diagnostics"
    return None


def normalized(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return re.sub(r"[^a-zA-Z0-9]+", " ", value).lower()


def semantic_path(path: str) -> str:
    """Drop repository branding before treating a path as semantic evidence."""
    return path.removeprefix("sgl-model-gateway/")


def referenced_identifiers(node: ast.AST) -> str:
    names = []
    for part in lexical_body_nodes(node):
        if isinstance(part, ast.Name):
            names.append(part.id)
        elif isinstance(part, ast.Attribute):
            names.append(part.attr)
    return normalized(" ".join(names))


def assertion_info(node: ast.AST) -> tuple[int, list[str]]:
    calls = set()
    count = 0
    for part in lexical_body_nodes(node):
        if isinstance(part, ast.Assert):
            count += 1
        if isinstance(part, ast.Call):
            try:
                name = ast.unparse(part.func)
            except (ValueError, TypeError):
                continue
            if re.search(r"(?:^|\.)(assert|check|compare|verify)[_a-zA-Z]*", name, re.I):
                count += 1
                calls.add(name)
    return count, sorted(calls)[:12]


def static_test_evidence(node: ast.AST, imports: dict[str, str]) -> tuple[list[dict], int, list[dict], int]:
    """Record bounded source clues without claiming that they executed.

    Only calls whose root name is an explicit vLLM/SGLang import are resolved.
    Method calls through objects and helper bodies need runtime evidence.
    """
    product_calls = []
    assertions = []
    for part in lexical_body_nodes(node):
        if isinstance(part, ast.Call):
            try:
                call_name = ast.unparse(part.func)
            except (ValueError, TypeError):
                continue
            root, separator, suffix = call_name.partition(".")
            target = imports.get(root)
            if target:
                product_calls.append({"line": part.lineno, "target": target + ("." + suffix if separator else "")})
            if re.search(r"(?:^|\.)assert[_a-zA-Z]*$", call_name, re.I):
                assertions.append({"line": part.lineno, "kind": "assertion_call",
                                   "expression": ast.unparse(part)[:180]})
        elif isinstance(part, ast.Assert):
            assertions.append({"line": part.lineno, "kind": "assert",
                               "expression": ast.unparse(part)[:180]})
        elif isinstance(part, (ast.With, ast.AsyncWith)):
            for item in part.items:
                expr = item.context_expr
                if isinstance(expr, ast.Call) and ast.unparse(expr.func).endswith(("raises", "assertRaises")):
                    assertions.append({"line": part.lineno, "kind": "expected_exception",
                                       "expression": ast.unparse(expr)[:180]})
    product_calls.sort(key=lambda item: (item["line"], item["target"]))
    assertions.sort(key=lambda item: (item["line"], item["kind"]))
    return product_calls, len(product_calls), assertions[:4], len(assertions)


def product_imports(tree: ast.Module, *, nested: bool = False) -> dict[str, str]:
    imports = {}
    def is_product(name: str) -> bool:
        return (name == "vllm" or name.startswith("vllm.")) or (
            (name == "sglang" or name.startswith("sglang."))
            and name != "sglang.test" and not name.startswith("sglang.test."))
    for node in ast.walk(tree) if nested else tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if not is_product(node.module):
                continue
            for alias in node.names:
                if alias.name != "*":
                    imports[alias.asname or alias.name] = node.module + "." + alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if is_product(alias.name):
                    imports[alias.asname or alias.name.split(".")[0]] = alias.name if alias.asname else alias.name.split(".")[0]
    return imports


def route_candidates(path: str, test_name: str, references: str) -> list[dict]:
    direct = normalized(semantic_path(path) + " " + test_name)
    found = []
    for route, patterns in COMPILED_RULES.items():
        direct_hits = sorted({m.group() for p in patterns for m in p.finditer(direct)})
        reference_hits = sorted({m.group() for p in patterns for m in p.finditer(references)})
        # Referenced calls alone are insufficient: helper imports often mention
        # unrelated behaviors. A route name in path/test is source evidence.
        if not direct_hits:
            continue
        found.append({"route": route, "evidence": direct_hits[:5], "referenced": reference_hits[:5]})
    return found


def request_route_candidates(item: dict, classification: dict) -> list[dict]:
    """Keep route clues only for objects represented in the request graph."""
    return item["routes"] if any(area.get("phase") for area in classification["areas"]) else []


def file_areas(path: str, items: list[dict]) -> list[dict]:
    location = normalized(semantic_path(path))
    names = normalized(" ".join(item["name"] for item in items))
    path_found = []
    name_found = []
    for key, label, phase, pattern in COMPILED_AREA_RULES:
        path_hits = sorted(set(match.group() for match in pattern.finditer(location)))
        name_hits = sorted(set(match.group() for match in pattern.finditer(names)))
        if path_hits or name_hits:
            record = {
                "id": key, "label": label, "phase": phase,
                "evidence": path_hits[:3] or name_hits[:3],
                "basis": "file_path" if path_hits else "test_name",
            }
            (path_found if path_hits else name_found).append(record)
    found = path_found or name_found
    if path.startswith("python/sglang/multimodal_gen/"):
        # This tree tests image/diffusion generation. The word "multimodal"
        # in its package name is not evidence of an LLM request input path.
        return [area for area in found if area["id"] == "image_generation"]
    return found


def test_classification(path: str, item: dict, file_areas: list[dict] | None = None) -> dict:
    """Give each test a source-grounded *candidate* object category.

    Direct imported product calls outrank a test identifier; a test identifier
    outranks the shared file path. Every candidate records its source. None of
    these signals proves an executed request path or an asserted property.
    """
    matches = []
    infrastructure = infrastructure_area(path)
    if infrastructure:
        label, phase = next((label, phase) for key, label, phase, _ in COMPILED_AREA_RULES
                            if key == infrastructure)
        matches = [{"id": infrastructure, "label": label, "phase": phase,
                    "basis": "repository_infrastructure", "score": 1,
                    "evidence": [path]}]
    if not matches:
        quality = next((label for key, label, _, _ in COMPILED_AREA_RULES if key == "quality"), "任务质量")
        if any(re.search(pattern, normalized(item.get("name", ""))) for pattern in ROUTE_QUALITY_MARKERS):
            matches = [{"id": "quality", "label": quality, "phase": "service",
                        "basis": "task_quality_marker", "score": 1,
                        "evidence": [item.get("name", "")]}]
    if not matches and item.get("kind") == "source_test_file_unresolved":
        if path.startswith("sgl-model-gateway/tests/spec/") or path == "sgl-model-gateway/tests/spec_test.rs":
            matches = [{"id": "diagnostics", "label": "调试与测试基础设施", "phase": None,
                        "basis": "protocol_test_module", "score": 1,
                        "evidence": [path]}]
        for key, pattern in PATH_ONLY_RULES:
            if matches:
                break
            if pattern.search(path):
                label, phase = next((label, phase) for area_key, label, phase, _ in COMPILED_AREA_RULES
                                    if area_key == key)
                matches = [{"id": key, "label": label, "phase": phase,
                            "basis": "file_only_path", "score": 1,
                            "evidence": [pattern.pattern]}]
                break
    if not matches and path.startswith("sgl-model-gateway/tests/spec/"):
        # Rust protocol tests contain both request and response contracts. The
        # declaration module above is a file record, not an executable test.
        name = normalized(item.get("name", ""))
        protocol_areas = []
        request_validation = bool(re.search(r"\bvalidate conversation\b", name))
        if re.search(r"\b(request|req input|workflow)\b", name) or request_validation:
            protocol_areas.append("ingress")
        if re.search(r"\b(result|workflow)\b", name) or (
            re.search(r"\bresponse\b", name) and not request_validation
        ):
            protocol_areas.append("output")
        matches = [{"id": key,
                    "label": next(label for area_key, label, _, _ in COMPILED_AREA_RULES
                                  if area_key == key),
                    "phase": next(phase for area_key, _, phase, _ in COMPILED_AREA_RULES
                                  if area_key == key),
                    "basis": "protocol_test_name", "score": 1,
                    "evidence": [item.get("name", "")]} for key in protocol_areas]
    directory_match = None
    if not matches:
        for pattern, key in COMPILED_STRONG_DIRECTORIES:
            if pattern.search(path):
                label, phase = next((label, phase) for area_key, label, phase, _ in COMPILED_AREA_RULES
                                    if area_key == key)
                directory_match = {"id": key, "label": label, "phase": phase,
                                   "basis": "component_test_directory", "score": 1,
                                   "evidence": [pattern.pattern]}
                break
    calls = item.get("source_calls", [])
    votes = Counter()
    call_evidence: dict[str, set[str]] = {}
    for call in ([] if matches else calls):
        prefix_area = next((area for prefix, area in PRODUCT_PREFIX_AREAS
                            if call["target"].startswith(prefix)), None)
        if prefix_area:
            votes[prefix_area] += 1
            call_evidence.setdefault(prefix_area, set()).add(call["target"].split(".")[0] + " module")
            continue
        target = normalized(call["target"])
        # A concrete call contributes to its most specific matching category;
        # broad terms elsewhere in the same name do not create extra votes.
        for key, label, phase, pattern in COMPILED_AREA_RULES:
            found = [match.group() for match in pattern.finditer(target)]
            if found:
                votes[key] += 1
                call_evidence.setdefault(key, set()).update(found)
                break
    if directory_match and not matches:
        name_text = normalized(item.get("name", ""))
        name_areas = {key for key, _, _, pattern in COMPILED_AREA_RULES
                      if pattern.search(name_text)}
        alternatives = {key: score for key, score in votes.items()
                        if directory_match["id"] == "cache" and key == "sampling"
                        and key in name_areas
                        and key in NAME_OVERRIDE_RULES
                        and NAME_OVERRIDE_RULES[key].search(name_text)}
        if alternatives:
            key = max(alternatives, key=alternatives.get)
            label, phase = next((label, phase) for area_key, label, phase, _ in COMPILED_AREA_RULES
                                if area_key == key)
            matches = [{"id": key, "label": label, "phase": phase,
                        "basis": "named_direct_product_call", "score": alternatives[key],
                        "evidence": sorted(call_evidence[key])[:3]}]
        else:
            matches = [directory_match]
    if votes and not matches:
        matches = [{"id": key, "label": label, "phase": phase,
                    "basis": "direct_product_call", "score": votes[key],
                    "evidence": sorted(call_evidence[key])[:3]}
                   for key, label, phase, _ in COMPILED_AREA_RULES if key in votes]
        matches.sort(key=lambda area: -area["score"])
    if not matches:
        for basis, raw in (("test_name", item["name"]),):
            normalized_text = normalized(raw)
            matches = [{"id": key, "label": label, "phase": phase,
                        "basis": basis, "score": 1,
                        "evidence": sorted({m.group() for m in pattern.finditer(normalized_text)})[:3]}
                       for key, label, phase, pattern in COMPILED_AREA_RULES
                       if pattern.search(normalized_text)]
            if matches:
                break
    if not matches:
        for key, pattern in PATH_ONLY_RULES:
            if key == "runtime_support":
                continue
            if pattern.search(path):
                label, phase = next((label, phase) for area_key, label, phase, _ in COMPILED_AREA_RULES
                                    if area_key == key)
                matches = [{"id": key, "label": label, "phase": phase,
                            "basis": "specific_path", "score": 1,
                            "evidence": [pattern.pattern]}]
                break
    if not matches:
        normalized_text = normalized(semantic_path(path))
        matches = [{"id": key, "label": label, "phase": phase,
                    "basis": "file_path", "score": 1,
                    "evidence": sorted({m.group() for m in pattern.finditer(normalized_text)})[:3]}
                   for key, label, phase, pattern in COMPILED_AREA_RULES
                   if pattern.search(normalized_text)]
    if not matches:
        for key, pattern in PATH_ONLY_RULES:
            if key != "runtime_support" or not pattern.search(path):
                continue
            label, phase = next((label, phase) for area_key, label, phase, _ in COMPILED_AREA_RULES
                                if area_key == key)
            matches = [{"id": key, "label": label, "phase": phase,
                        "basis": "specific_path", "score": 1,
                        "evidence": [pattern.pattern]}]
            break
    if not matches:
        # A file-level theme may combine unrelated tests. For a function with
        # no local evidence, use only its own source path; never inherit the
        # other functions' test names as if they were this function's evidence.
        path_areas = file_areas_for_path(path)
        if path_areas:
            matches = [{**area, "basis": "path_fallback", "score": 1}
                       for area in path_areas]
    if not matches:
        matches = [{"id": "unresolved", "label": "待源码复核", "phase": None,
                    "basis": "unresolved", "score": 0,
                    "evidence": [path, item.get("name", "")]}]
    return {
        "areas": matches,
        "primary_area": matches[0]["id"] if matches else None,
        "basis": matches[0]["basis"] if matches else "unclassified",
        "ambiguous": len(matches) > 1 and matches[0]["score"] == matches[1]["score"],
        "source_calls_sampled": len(calls),
        "source_calls_total": item.get("source_call_count", 0),
        "status": "needs_source_review" if matches[0]["id"] == "unresolved"
                  else "source_candidate",
    }


def file_areas_for_path(path: str) -> list[dict]:
    """Infer a file theme using the path alone, excluding sibling test names."""
    location = normalized(semantic_path(path))
    return [{"id": key, "label": label, "phase": phase,
             "evidence": sorted({m.group() for m in pattern.finditer(location)})[:3],
             "basis": "file_path"}
            for key, label, phase, pattern in COMPILED_AREA_RULES
            if key != "unresolved" and pattern.search(location)]


def py_test_items(path: str, text: str) -> tuple[list[dict], str | None]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        return [], f"Python syntax could not be parsed: {type(exc).__name__}"
    items = []
    imports = product_imports(tree)

    def visit(nodes: list[ast.stmt], parents: list[str]) -> None:
        for node in nodes:
            if isinstance(node, ast.ClassDef):
                visit(node.body, parents + [node.name])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_") or node.name == "test":
                    name = "::".join(parents + [node.name])
                    refs = referenced_identifiers(node)
                    assertions, assertion_calls = assertion_info(node)
                    local_imports = {**imports, **product_imports(ast.Module(body=node.body, type_ignores=[]), nested=True)}
                    source_calls, source_call_count, source_assertions, source_assertion_count = static_test_evidence(node, local_imports)
                    items.append({
                        "name": name,
                        "line": node.lineno,
                        "start_line": min([node.lineno, *(decorator.lineno for decorator in node.decorator_list)]),
                        "kind": "python_test_definition",
                        "parameterized": any("parametrize" in ast.unparse(d) for d in node.decorator_list),
                        "assertion_count": assertions,
                        "assertion_calls": assertion_calls,
                        "source_calls": source_calls,
                        "source_call_count": source_call_count,
                        "source_assertions": source_assertions,
                        "source_assertion_count": source_assertion_count,
                        "routes": route_candidates(path, name, refs),
                    })
                # Nested test functions are uncommon but remain discoverable.
                visit(node.body, parents + [node.name])

    visit(tree.body, [])
    return items, None


def rust_test_items(path: str, text: str) -> list[dict]:
    items = []
    pending_test = False
    for line, source in enumerate(text.splitlines(), 1):
        stripped = source.strip()
        if re.match(r"#\[.*(?:test|rstest).*\]", stripped):
            pending_test = True
        match = re.search(r"\bfn\s+(\w+)\s*\(", stripped)
        if match:
            name = match.group(1)
            if pending_test or name.startswith("test_"):
                items.append({
                    "name": name,
                    "line": line,
                    "start_line": line,
                    "kind": "rust_test_definition",
                    "parameterized": "rstest" in stripped,
                    "assertion_count": 0,
                    "assertion_calls": [],
                    "source_calls": [],
                    "source_call_count": 0,
                    "source_assertions": [],
                    "source_assertion_count": 0,
                    "routes": route_candidates(path, name, ""),
                })
            pending_test = False
    return items


def is_source_test(path: str) -> bool:
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py") or ("/tests/" in path and path.endswith((".rs", ".go")))


def build_catalog() -> dict:
    model = json.loads((HERE / "space_seed.json").read_text(encoding="utf-8"))
    for key, (directory, _) in SNAPSHOTS.items():
        root = REPOS / directory
        if not root.is_dir():
            raise FileNotFoundError(f"Missing pinned source checkout: {root}")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        expected = model["snapshots"][key]["commit"]
        if actual != expected:
            raise RuntimeError(f"Wrong {key} source commit: expected {expected}, got {actual}")
    indexed = list(csv.DictReader((REPOS / "test-source-index.tsv").open(newline=""), delimiter="\t"))
    snapshots = []
    counts = Counter()
    for key, (directory, title) in SNAPSHOTS.items():
        files = []
        for row in indexed:
            if row["snapshot"] != directory:
                continue
            path = row["relative_path"]
            source = REPOS / directory / path
            if not source.is_file():
                raise FileNotFoundError(f"Indexed source file is missing: {source}")
            items = []
            parse_error = None
            if path.endswith(".py"):
                items, parse_error = py_test_items(path, source.read_text(encoding="utf-8", errors="replace"))
            elif path.endswith(".rs"):
                items = rust_test_items(path, source.read_text(encoding="utf-8", errors="replace"))
            if not items and is_source_test(path):
                # File-level record: decorators or a custom runner may generate
                # tests dynamically. No executable test case is asserted here.
                items = [{
                    "name": Path(path).name,
                    "line": 1,
                    "start_line": 1,
                    "kind": "source_test_file_unresolved",
                    "parameterized": False,
                    "assertion_count": 0,
                    "assertion_calls": [],
                    "source_calls": [],
                    "source_call_count": 0,
                    "source_assertions": [],
                    "source_assertion_count": 0,
                    "routes": route_candidates(path, "", ""),
                }]
            counts["source_files"] += 1
            counts["test_entries"] += len(items)
            counts["entries_with_source_calls"] += sum(bool(item["source_calls"]) for item in items)
            counts["entries_with_source_assertions"] += sum(bool(item["source_assertions"]) for item in items)
            counts["source_test_file_unresolved"] += sum(item["kind"] == "source_test_file_unresolved" for item in items)
            areas = file_areas(path, items)
            for item in items:
                item["classification"] = test_classification(path, item, areas)
                # A keyword can occur in CI, configuration, training, or image
                # generation tests without referring to an inference request.
                # Retain the source entry, but do not place it on this request
                # graph unless its classified object has a request phase.
                item["routes"] = request_route_candidates(item, item["classification"])
                counts["candidate_mapped_entries"] += bool(item["routes"])
                counts[f"classified_by_{item['classification']['basis']}"] += 1
                counts["ambiguous_test_entries"] += item["classification"]["ambiguous"]
            counts["files_with_local_placement"] += bool(areas) and bool(items)
            files.append({"path": path, "kind": row["kind"], "items": items, "areas": areas, "parse_error": parse_error})
        snapshots.append({"id": key, "title": title, "directory": directory, "files": files})
    return {
        "scope": "source-level test definitions and source-index files in pinned vLLM and SGLang 0.5.14 snapshots",
        "limits": ["pytest parameter combinations are not expanded", "runtime coverage and assertions are not established by route keywords", "non-Python test discovery is partial"],
        "counts": dict(counts),
        "snapshots": snapshots,
    }


def main() -> None:
    catalog = build_catalog()
    target = HERE / "test_catalog.json"
    target.write_text(json.dumps(catalog, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rows = []
    for snapshot in catalog["snapshots"]:
        for file in snapshot["files"]:
            routes = sorted({route["route"] for item in file["items"] for route in item["routes"]})
            rows.append({
                "snapshot": snapshot["id"],
                "path": file["path"],
                "test_entries": len(file["items"]),
                "route_candidates": ",".join(routes),
                "local_areas": ",".join(area["id"] for area in file["areas"]),
                "request_phases": ",".join(sorted({area["phase"] for area in file["areas"] if area["phase"]})),
                "basis": ";".join(f"{area['id']}:{area['basis']}:{','.join(area['evidence'])}" for area in file["areas"]),
                "parse_error": file["parse_error"] or "",
            })
    with (HERE / "test-file-placement.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    test_rows = []
    for snapshot in catalog["snapshots"]:
        for file in snapshot["files"]:
            for item in file["items"]:
                classification = item["classification"]
                test_rows.append({
                    "snapshot": snapshot["id"], "path": file["path"],
                    "test_name": item["name"], "line": item["line"],
                    "kind": item["kind"], "primary_area": classification["primary_area"] or "",
                    "candidate_areas": ",".join(area["id"] for area in classification["areas"]),
                    "basis": classification["basis"],
                    "candidate_evidence": json.dumps(
                        [{"area": area["id"], "basis": area["basis"],
                          "source": area["evidence"]} for area in classification["areas"]],
                        ensure_ascii=False, separators=(",", ":")),
                    "ambiguous": int(classification["ambiguous"]),
                    "source_call_count": item["source_call_count"],
                    "source_assertion_count": item["source_assertion_count"],
                    "request_route_candidates": ",".join(route["route"] for route in item["routes"]),
                })
    with (HERE / "test-entry-classification.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(test_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(test_rows)
    overlaps = Counter(row["candidate_areas"] for row in test_rows if row["ambiguous"])
    with (HERE / "classification-overlap-summary.tsv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["candidate_areas", "test_entries"], delimiter="\t")
        writer.writeheader()
        writer.writerows({"candidate_areas": areas, "test_entries": count}
                         for areas, count in overlaps.most_common())
    test_files = [row for row in rows if row["test_entries"]]
    summary = {
        "indexed_source_files": len(rows),
        "files_with_test_entries": len(test_files),
        "files_with_request_route_candidate": sum(bool(row["route_candidates"]) for row in test_files),
        "files_with_request_phase_candidate": sum(bool(row["request_phases"]) for row in test_files),
        "files_with_local_area_candidate": sum(bool(row["local_areas"]) for row in test_files),
        "files_outside_current_request_graph": sum(bool(row["local_areas"]) and not row["request_phases"] for row in test_files),
        "source_level_test_entries": catalog["counts"]["test_entries"],
        "source_test_file_records_without_function_discovery": catalog["counts"]["source_test_file_unresolved"],
        "discovered_source_test_definitions": (catalog["counts"]["test_entries"]
                                               - catalog["counts"]["source_test_file_unresolved"]),
        "source_level_entries_with_candidate_object": sum(row["primary_area"] not in ("", "unresolved") for row in test_rows),
        "source_level_entries_without_candidate_object": sum(row["primary_area"] == "unresolved" for row in test_rows),
        "source_level_entries_with_ambiguous_object": sum(row["ambiguous"] for row in test_rows),
        "classification_basis": dict(Counter(row["basis"] for row in test_rows)),
        "primary_area_counts": dict(Counter(row["primary_area"] for row in test_rows)),
        "entries_with_route_candidate": catalog["counts"]["candidate_mapped_entries"],
        "interpretation": "Static source placement candidates, not confirmed test execution, assertion coverage, or a 95% semantic-coverage estimate.",
    }
    (HERE / "test-placement-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {target.name}, test-file-placement.tsv, test-placement-summary.json: {summary}")


if __name__ == "__main__":
    main()
