"""Second source review of SGLang image/video generation configuration tests."""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
PATH="python/sglang/multimodal_gen/test/unit/test_server_args.py"
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"/PATH
COLS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]
CONFIG_NAMES={
"test_dynamic_component_attention_backend_cli_args",
"test_dit_layerwise_offload_from_kwargs",
"test_dit_layerwise_offload_cli_arg",
"test_layerwise_offload_components_cli_args",
"test_serve_cli_preserves_config_and_dynamic_unknown_args",
"test_serve_cli_defaults_warmup_on",
"test_serve_cli_preserves_explicit_warmup_false",
"test_serve_cli_preserves_config_warmup_false",
"test_pipeline_configs_declare_auto_tune_hints",
"test_cfg_parallel_cli_can_be_disabled_explicitly",
"test_ltx23_snapshot_device_mode_cli_alias_is_accepted",
"test_cli_args_parsed",
"test_resolution_flag_overrides_qwen_image_layered_pipeline_config",
"test_disable_autocast_is_preserved_after_pipeline_config_resolution",
"test_downstream_wait_timeout_cli_arg_is_parsed",
"test_disagg_role_alias_cli_arg_is_accepted",
"test_transfer_backend_cli_arg_is_parsed",
}

def main():
 text=SOURCE.read_text();tree=ast.parse(text)
 tests={n.lineno:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name.startswith("test_")}
 rows=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"));keep=[];out=[]
 for row in rows:
  if row["path"]!=PATH:keep.append(row);continue
  node=tests.get(int(row["line"]))
  if node is None or node.name not in CONFIG_NAMES:keep.append(row);continue
  calls=sorted((n for n in ast.walk(node) if isinstance(n,ast.Call)),key=lambda n:n.lineno)
  product_calls=[n for n in calls if any(s in ast.unparse(n.func) for s in ("ServerArgs.from_", "ServerArgs.add_cli_args", "execute_serve_cmd", "get_model_deployment_config"))]
  assertions=[n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr.startswith("assert")]
  if not product_calls or not assertions:
   row["review_reason"]="No direct configuration producer plus assertion in test body."
   keep.append(row);continue
  parts=[" ".join((ast.get_source_segment(text,n) or "").split())[:220] for n in assertions]
  out.append({"snapshot":"sglang","path":PATH,"test_name":row["test_name"],"line":row["line"],
   "unit_id":"deployment.effective_configuration",
   "property":"Image/video generation configuration resolves asserted fields under the supplied CLI, kwargs, defaults or pipeline config: " + " | ".join(parts),
   "oracle_source":";".join(f"{PATH}:{n.lineno}" for n in assertions),
   "conditions":"Pinned SGLang 0.5.14 multimodal_gen test; configuration source, pipeline type, platform mocks and explicit flags retained in source body; checks resolved fields, not generated images or running service.",
   "rule_id":"sglang_gen_server_args_source_review_2"})
 with (HERE/"crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=COLS).writerows(out)
 with (HERE/"pending.tsv").open("w",newline="") as f:
  w=csv.DictWriter(f,delimiter="\t",fieldnames=rows[0].keys());w.writeheader();w.writerows(keep)
 e=HERE/"evidence.json";j=json.loads(e.read_text());j.setdefault("reviewed_clusters",[]).append({"path":PATH,"sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"mapped_existing_unit":len(out),"exceptions":sum(r["path"]==PATH for r in keep),"review_method":"Second read of non-helper CLI/config tests, including direct configuration producer and actual unittest assertion."});j["total_pending_after_review"]=len(keep);e.write_text(json.dumps(j,ensure_ascii=False,indent=2)+"\n")
 print("mapped",len(out),"remaining in file",sum(r["path"]==PATH for r in keep))
if __name__=="__main__":main()
