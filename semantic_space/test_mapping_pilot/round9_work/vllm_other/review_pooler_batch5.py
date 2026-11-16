"""Manual source-grounded review of vLLM local pooler checks."""

import csv
import json
from pathlib import Path
from review_cpu_gdn_batch6 import exact_entry

HERE=Path(__file__).resolve().parent
PATHS={"tests/model_executor/layers/test_pooler_heads.py","tests/model_executor/layers/test_pooler_methods.py"}


def verdict(path,name):
    cls,_,method=name.partition("::")
    if path.endswith("test_pooler_heads.py"):
        if method=="test_supported_tasks":
            return None,"supported_task_declaration_not_pooling_result"
        if "chunked_prefill" in method:
            return "model.pooling_partial_prefill_state","Unfinished and finished rows retain their asserted None/value positions in a mixed pooler batch."
        if method=="test_forward_empty_batch":
            return "model.pooling_transform_relation","An empty input batch yields the asserted empty pooler output."
        return "model.pooling_transform_relation","Local pooler head output is checked for the asserted value, shape, dtype, list layout or normalization under its supplied projector/classifier and truncation settings."
    if name.startswith("test_build_pooling_cursor_caches_cpu_status"):
        return "model.pooling_partial_prefill_state","Pooling cursor reports the asserted finished mask and partial-prefill status for sliced request ranges."
    if name.startswith("test_pooling_params_update_combines_token_id_requirements"):
        return None,"pooling_configuration_requirement_not_numeric_result"
    if cls in {"TestGetSeqPoolingMethod","TestGetTokPoolingMethod"}:
        return None,"pooling_method_registry_selection_requires_distinct_unit"
    if cls in {"TestCLSPool","TestLastPool","TestMeanPool"}:
        if any(x in method for x in ("partial_prefill","chunked_accumulation")):
            return "model.pooling_partial_prefill_state","Pooler handles an incomplete sequence with the asserted last token, accumulated mean or rejection behavior."
        return "model.pooling_transform_relation","CLS/LAST/MEAN output equals the asserted token selection or reference reduction for these sequence boundaries and dtype."
    if cls=="TestAllPool":
        if "chunked_prefill" in method:
            return "model.pooling_partial_prefill_state","AllPool retains unfinished chunks and returns the asserted concatenated or None result at sequence completion."
        if "owns_output_under_async" in method:
            return None,"output_storage_ownership_under_async_requires_separate_state_oracle"
        return "model.pooling_transform_relation","AllPool output retains the asserted per-sequence hidden states."
    if cls=="TestStepPool":
        if method=="test_requires_token_ids_update":
            return None,"pooling_configuration_requirement_not_numeric_result"
        if "chunked_prefill" in method:
            return "model.pooling_partial_prefill_state","StepPool returns None for unfinished chunks and filtered values on completion."
        return "model.pooling_transform_relation","StepPool output contains the asserted token rows selected by step tag or returned-token IDs."
    if name.startswith("test_dispatch_single_task_keeps_unfinished_token_outputs") or name.startswith("test_bos_eos_filter_waits_for_complete_chunk_vectors"):
        return "model.pooling_partial_prefill_state","Mixed finished/unfinished request handling yields the asserted None and value positions."
    if name.startswith(("test_late_chunk_pool_","test_bos_eos_filter_preserves_chunk_vectors")):
        return "model.pooling_partial_prefill_state","Chunked pooling preserves the asserted cross-chunk reduction or request isolation after completion."
    if name.startswith("test_dispatch_") or name.startswith("test_token_pooler_batches_projection"):
        return "model.pooling_transform_relation","Pooler dispatch produces the asserted per-request rows, offsets, dimensions or projected values for supplied task parameters."
    return None,"unreviewed_pooling_assertion"


def main():
    packets=[]
    for line in (HERE/"source-packets.jsonl").open():
        p=json.loads(line)
        if p["key"][1] in PATHS:packets.append(p)
    rows=[];pending=[];reviews={}
    for p in packets:
        snapshot,path,name,line=p["key"]
        unit,prop=verdict(path,name)
        ev=p["evidence"]
        checks=ev["checks"]+ev["expected_exceptions"]
        refs=";".join(f"{path}:{x['line']}" for x in checks[:5])
        calls=[x for x in ev["calls"] if not x["name"].startswith(("torch.","pytest.","len","isinstance"))]
        condition=";".join(f"{x['line']}:{x['source'][:100]}" for x in calls[:4])
        if unit and checks:
            rows.append([snapshot,path,name,line,unit,prop,refs,condition,"vllm_pooler_local_review_20261112"])
        else:
            pending.append([snapshot,path,name,line,prop if not unit else "assertion_not_located",refs or f"{path}:{line}"])
        reviews[path]={"snapshot":snapshot,"path":path,"source_sha256":p["sha256"],"review":"Reviewed test bodies and local _make_metadata helper; direct assertions kept at their actual granularity. Product classes checked in seqwise/tokwise heads and methods."}
    with (HERE/"crosswalk-batch5.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]);w.writerows(rows)
    with (HERE/"pending-batch5.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["snapshot","path","test_name","line","reason","evidence_source"]);w.writerows(pending)
    (HERE/"evidence-batch5.json").write_text(json.dumps({"batch":"pooler_local", "counts":{"scope":len(packets),"mapped":len(rows),"pending":len(pending)},"file_reviews":list(reviews.values()),"product_refs":["vllm/model_executor/layers/pooler/seqwise/methods.py:37","vllm/model_executor/layers/pooler/seqwise/heads.py:33","vllm/model_executor/layers/pooler/tokwise/methods.py:36","vllm/model_executor/layers/pooler/tokwise/heads.py:66"],"ontology_candidate":"needs-space-pooling.json"},ensure_ascii=False,indent=2)+"\n")
    # Preserve complete per-entry assertion/input evidence on every rebuild.
    payload=json.load((HERE/"evidence-batch5.json").open())
    packet_lookup={tuple(p["key"]):p for p in packets}
    payload["entries"]=[]
    for row in rows:
        e=exact_entry(row[1],row[2])
        payload["entries"].append({"key":row[:4],"source_sha256":packet_lookup[tuple(row[:4])]["sha256"],"unit_id":row[4],"property":row[5],**e})
        row[6]=";".join(f"{row[1]}:{c['line']}" for c in e["checks"])
        row[7]+="; full exact input construction and checks: evidence-batch5.json entry "+row[2]
    (HERE/"evidence-batch5.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    with (HERE/"crosswalk-batch5.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]);w.writerows(rows)
    print(len(packets),len(rows),len(pending))


if __name__=="__main__":main()
