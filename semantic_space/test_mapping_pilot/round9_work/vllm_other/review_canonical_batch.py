"""Source review of the largest strict AST families, with input exceptions.

The family index routes review only. Each member keeps its own test source,
literal conditions and assertion/helper oracle location.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from extract_review import HERE, SOURCE

FIELDS = ["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]

MAPPED = {
    "58f25b607873213e": ("model.parameter_binding", "Checkpoint KV-scale names resolve to the asserted target model parameter name or remain unchanged.", ""),
    "57fe65d2e2038f5e": ("tool.call_parse_relation", "Recipient classifier accepts or rejects the asserted tool recipient syntax.", ""),
    "f14b245e58022565": ("quality.task_score", "MTEB embedding task score differs from the declared HF or recorded reference by less than the model's tolerance.", "tests/models/language/pooling_mteb_test/mteb_embed_utils.py:276"),
    "5d90c4a98a136d47": ("model.parameter_binding", "Checkpoint weight name without a routed expert ID is not assigned an expert index.", ""),
    "7395da991507846c": ("compute.operator_reference_relation", "CPU paged-KV variable-length attention output agrees with the reference output under the test's dtype, head, block, window and ISA parameters.", "tests/kernels/attention/test_cpu_attn.py:706;tests/kernels/attention/test_cpu_attn.py:710"),
    "807eecb34bad0df3": ("request.effective_sampling_parameters", "Explicit client penalty values override configured defaults; omitted penalties receive the configured or neutral default.", ""),
}

PENDING = {
    "ae320c83c7be6415": "needs_runtime_model_backend_smoke_unit",
    "56d07904f01b1952": "needs_scalar_score_reference_unit",
    "178979f1323aa4bc": "needs_weight_layout_axis_interpretation_unit",
    "68845ea80b2dfbae": "needs_kernel_configuration_selection_unit",
    "20baf70894f3be32": "needs_expert_weight_ownership_unit",
    "d0e71c6fdd764d0a": "needs_worker_environment_transfer_selection_unit",
    "32f911a94a1c4865": "needs_kernel_determinism_unit",
    "3b54c4348f791eac": "helper_unconditionally_pytest_skips_before_comparison",
}

def source_checks(packet):
    refs=[]
    ev=packet.get("evidence") or {}
    for c in ev.get("checks",[]): refs.append(f"{packet['key'][1]}:{c['line']}")
    return refs

def condition_source(packet):
    targets=("maybe_remap_kv_scale_name","is_function_recipient","mteb_test_embed_models","parse_expert_id","varlen_with_paged_kv","to_sampling_params","_run_ranks")
    calls=(packet.get("evidence") or {}).get("calls",[])
    relevant=[f"{c['line']}:{c['source']}" for c in calls if any(t in c['name'] for t in targets)]
    parametrization=[f"{c['line']}:{c['source']}" for c in calls if c['name']=="pytest.mark.parametrize"]
    parts=relevant[:2]+parametrization[:2]
    return "; ".join(parts)[:700] if parts else f"source definition {packet['key'][1]}:{packet['key'][3]}"

def main():
    packets={tuple(x["key"]):x for x in (json.loads(line) for line in (HERE/"source-packets.jsonl").open())}
    families=json.load((HERE/"canonical-families.json").open())
    selected=[x for x in families if x["family_id"] in MAPPED or x["family_id"] in PENDING or x["family_id"]=="db8ebef1e534b339"]
    rows=[];pending=[];file_refs={}
    for family in selected:
        fid=family["family_id"]
        for key in family["members"]:
            packet=packets[tuple(key)]
            file_refs[key[1]]=packet["sha256"]
            name=key[2].split("::")[-1]
            checks=source_checks(packet)
            if fid=="db8ebef1e534b339":
                # The exact same AST dispatch call uses different literal
                # check names. Five compare replicated projections; one checks
                # own-shard writes. The literal is a semantic input condition.
                if name=="test_sharded_tail_tp4_writes_only_its_own_shard":
                    unit="distributed.partition_binding"
                    prop="Each TP rank writes the declared local tail shard and leaves other shard ranges unchanged."
                    helper="tests/models/kimi_k3/test_amd_latent_moe_runner.py:229;tests/models/kimi_k3/test_amd_latent_moe_runner.py:232;tests/models/kimi_k3/test_amd_latent_moe_runner.py:233"
                else:
                    unit="distributed.logical_execution"
                    prop="Sharded or pre-sharded projection result agrees with the replicated projection after rank assembly under the test's TP/EP configuration."
                    helper="tests/models/kimi_k3/test_amd_latent_moe_runner.py:155;tests/models/kimi_k3/test_amd_latent_moe_runner.py:185"
            elif fid in MAPPED:
                unit,prop,helper=MAPPED[fid]
            else:
                pending.append([*key,PENDING[fid],fid,"reviewed strict AST family; target property absent or narrower than current seed"])
                continue
            refs=checks+([helper] if helper else [])
            if not refs:
                pending.append([*key,"mapped_family_oracle_not_visible",fid,"review helper chain before confirmation"])
                continue
            rows.append({"snapshot":key[0],"path":key[1],"test_name":key[2],"line":key[3],"unit_id":unit,"property":prop,"oracle_source":";".join(refs),"conditions":condition_source(packet),"rule_id":f"vllm_canonical_family_{fid}_review_20261112"})
    with (HERE/"crosswalk-batch3.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,delimiter="\t");w.writeheader();w.writerows(rows)
    with (HERE/"pending-batch3.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["snapshot","path","test_name","line","reason","family_id","review_note"]);w.writerows(pending)
    evidence={"batch":"largest_canonical_ast_families","family_method":"Full function AST shape, call targets, attribute names, comparison operators and control structure retained; local names and literal values normalized. Family membership routes review but does not assign properties. In the six Kimi K3 distributed tests, the check-dispatch string changes the target property and is split explicitly.","file_reviews":[{"snapshot":"vllm","path":p,"source_sha256":sha,"review":"Reviewed representative function and relevant common helper; each member retains its assertion source or common helper oracle and unique literal conditions. Mappings are local property checks, not full end-to-end proof."} for p,sha in sorted(file_refs.items())],"reviewed_entries":len(rows),"needs_space_entries":len(pending),"family_sizes":{x["family_id"]:x["size"] for x in selected},"shared_helper_refs":["tests/models/language/pooling_mteb_test/mteb_embed_utils.py:276","tests/models/language/pooling/embed_utils.py:40","tests/kernels/attention/test_cpu_attn.py:706","tests/models/kimi_k3/test_amd_latent_moe_runner.py:126"]}
    (HERE/"evidence-batch3.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    needs=[
      {"id":"deployment.serving_backend_smoke","property":"在声明支持的模型、硬件、精度与后端组合下，服务能够启动并对基本请求返回非空结果。","vllm_test":"tests/quantization/test_blackwell_moe.py:43","vllm_source":"vllm/entrypoints/openai/completion/serving.py","scope_note":"仅启动与基本应答；不推断模型数值正确或性能。"},
      {"id":"scoring.reference_relation","property":"同一查询与文档配对上的标量分数，在指定容差内与参考评分实现一致，输出条目数和顺序对应输入配对。","vllm_test":"tests/entrypoints/pooling/scoring/test_bi_encoder_online.py:99","scope_note":"标量 score 与向量 embedding 是不同可观察量。"},
      {"id":"model.weight_layout_axis_interpretation","property":"权重形状、转置与分片维度确定的隐藏维和中间维与声明的检查点布局一致。","vllm_test":"tests/kernels/moe/test_moe_weight_loading_padded.py:31","vllm_source":"vllm/model_executor/layers/fused_moe/routed_experts.py:485"},
      {"id":"compute.kernel_configuration_selection","property":"算子在输入形状、精度和设备能力条件下选择声明支持的内核配置或回退配置。","vllm_test":"tests/kernels/helion/test_fused_qk_norm_rope.py:111","scope_note":"只验证配置选择，不推断计算结果。"},
      {"id":"distributed.expert_weight_ownership","property":"在专家并行布局下，各 rank 识别本地专家权重并跳过远端专家权重，同时保留共享权重和必要的量化尺度。","vllm_test":"tests/model_executor/model_loader/test_ep_weight_filter.py:175","vllm_source":"vllm/model_executor/model_loader/ep_weight_filter.py:68"},
      {"id":"distributed.worker_environment_transfer_selection","property":"按内置及用户指定的环境变量前缀和显式名称，确定需要传给分布式 worker 的变量集合。","vllm_test":"tests/test_ray_env.py:21","vllm_source":"vllm/ray/ray_env.py:56","scope_note":"该源测试只检查待复制变量的选择，不检查真实 worker 环境。"},
      {"id":"compute.repeatability","property":"在声明相同的输入、后端和执行条件下，重复算子执行的输出满足指定精度或位级重复性关系。","vllm_test":"tests/kernels/quantization/test_rdna3_w4a16_determinism.py:220"}
    ]
    for n in needs:n["status"]="candidate_needs_dual_framework_review"
    (HERE/"needs-space-batch3.json").write_text(json.dumps(needs,ensure_ascii=False,indent=2))
    print(len(selected),len(rows),len(pending))

if __name__=="__main__":main()
