# -*- coding: utf-8 -*-
"""Source-linked SGLang endpoint representation family."""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
PATH="test/registered/utils/test_network_address.py"
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"/PATH
COLS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]
def main():
    text=SOURCE.read_text(encoding="utf-8");tree=ast.parse(text)
    rows=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"));keep=[];out=[]
    for row in rows:
        if row["path"]!=PATH:keep.append(row);continue
        line=int(row["line"])
        nodes=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.lineno<=line<=n.end_lineno]
        n=min(nodes,key=lambda x:x.end_lineno-x.lineno) if nodes else None
        if n is None:keep.append(row);continue
        calls=[x for x in ast.walk(n) if isinstance(x,ast.Call) and ("NetworkAddress" in ast.unparse(x.func) or "is_zmq_endpoint_ipv6" in ast.unparse(x.func))]
        asserts=[x for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr.startswith("assert")]
        asserts=sorted(asserts,key=lambda x:x.lineno)
        if not calls or not asserts:
            row["review_reason"]="No direct NetworkAddress/IPv6 endpoint call with unittest assertion in body."
            keep.append(row);continue
        snippets=[" ".join((ast.get_source_segment(text,x) or "").split())[:160] for x in asserts[:5]]
        out.append({"snapshot":"sglang","path":PATH,"test_name":row["test_name"],"line":row["line"],"unit_id":"deployment.network_endpoint_representation","property":"For specified host, port and endpoint form, parsing, formatting or rejection follows this case's asserted relation: "+" | ".join(snippets),"oracle_source":";".join(f"{PATH}:{x.lineno}" for x in asserts[:8]),"conditions":"Pinned SGLang 0.5.14 NetworkAddress utility; IPv4, IPv6, hostname and malformed text cases as provided; address representation only, not successful network communication.","rule_id":"sglang_network_address_direct_source_review"})
    with (HERE/"proposed-crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,delimiter="\t",fieldnames=COLS).writerows(out)
    with (HERE/"pending.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,delimiter="\t",fieldnames=rows[0].keys());w.writeheader();w.writerows(keep)
    needs=json.loads((HERE/"needs-space.json").read_text());needs.append({"id":"deployment.network_endpoint_representation","kind":"obligation","phase":"preflight","state_scope":"deployment","name":"Configured network endpoint has correct canonical representation","property":"For supported IPv4, IPv6 and hostname forms, host and port parsing and URI formatting preserve the address; malformed forms are rejected before transport setup.","conditions":["deployment_configuration"],"sglang_evidence":[f"{PATH}:22","python/sglang/srt/utils/network.py"],"vllm_evidence":["vllm/utils/network_utils.py:119","vllm/utils/network_utils.py:128","vllm/utils/network_utils.py:141"],"difference":"SGLang uses immutable NetworkAddress with parse/to_url/to_tcp; vLLM provides split_host_port/join_host_port and IPv6 validation functions. Both express address representation, not live reachability."});(HERE/"needs-space.json").write_text(json.dumps(needs,ensure_ascii=False,indent=2)+"\n")
    evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"path":PATH,"sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"proposed_new_unit":len(out),"exceptions":sum(r["path"]==PATH for r in keep),"review_method":"Complete pinned test file read; direct NetworkAddress or IPv6 detector call and direct unittest assertion required."});evidence["total_pending_after_review"]=len(keep);(HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print("reviewed",len(out),"pending",sum(r["path"]==PATH for r in keep))
if __name__=="__main__":main()
