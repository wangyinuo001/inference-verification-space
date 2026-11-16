"""Merge only completed source-review batches; retain per-entry pending ledger."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from extract_review import HERE, SOURCE

FIELDS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]
PENDING_FIELDS=["snapshot","path","test_name","line","reason","source_evidence","next_review"]

def key(row): return tuple(row[x] for x in ("snapshot","path","test_name","line"))

def main():
    reviewed={}
    for name in ("crosswalk-batch1.tsv","crosswalk-batch3.tsv","crosswalk-batch4.tsv","crosswalk-batch5.tsv","crosswalk-batch6.tsv","crosswalk-batch7.tsv","crosswalk-batch8.tsv","crosswalk-batch9.tsv","crosswalk-batch10.tsv"):
        for row in csv.DictReader((HERE/name).open(),delimiter="\t"):
            k=key(row)
            if k in reviewed: raise ValueError(f"duplicate reviewed row {k}")
            reviewed[k]=row
    known_pending={}
    for name in ("pending-batch1.tsv","pending-batch3.tsv","pending-batch4.tsv","pending-batch5.tsv","pending-batch6.tsv","pending-batch7.tsv","pending-batch8.tsv","pending-batch9.tsv","pending-batch10.tsv"):
        for row in csv.DictReader((HERE/name).open(),delimiter="\t"):
            k=key(row)
            known_pending[k]=row["reason"]
    provisional={}
    for row in csv.DictReader((HERE/"crosswalk-batch2.tsv").open(),delimiter="\t"):
        provisional[key(row)]=row["unit_id"]
    for row in csv.DictReader((HERE/"pending-batch2.tsv").open(),delimiter="\t"):
        provisional[key(row)]=row["reason"]
    all_packets=[];pending=[]
    for line in (HERE/"source-packets.jsonl").open():
        p=json.loads(line);k=tuple(p["key"]);all_packets.append(k)
        if k in reviewed:continue
        ev=p.get("evidence") or {}
        calls=ev.get("calls",[]);checks=ev.get("checks",[])
        helper_checks=sum(len(x.get("checks",[])) for x in ev.get("local_helpers",{}).values())
        if k in known_pending:
            reason=known_pending[k]
        elif k in provisional:
            reason=f"derender_provisional_requires_full_helper_fixture_review:{provisional[k]}"
        elif p.get("error"):
            reason=p["error"]
        elif checks:
            reason="visible_assertion_present_but_target_property_not_reviewed"
        elif ev.get("expected_exceptions"):
            reason="expected_exception_requires_raised_call_and_scope_review"
        elif helper_checks or ev.get("local_helpers"):
            reason="oracle_delegated_to_local_helper_requires_relation_review"
        elif calls:
            reason="calls_present_without_direct_assertion_requires_fixture_or_external_oracle_review"
        else:
            reason="no_visible_call_or_assertion_in_test_body"
        refs=[f"{k[1]}:{x['line']}" for x in checks[:4]]
        refs += [f"{k[1]}:{x['line']}:{x['name']}" for x in calls[:3]]
        next_review="inspect full function, shared fixture/helper, product call and concrete assertion before mapping"
        if "syntax" in reason or "definition_not_located" in reason:next_review="parse with pinned Python grammar or locate generated definition"
        pending.append(dict(zip(PENDING_FIELDS,[*k,reason,";".join(refs),next_review])))
    if len(all_packets)!=len(set(all_packets)):raise ValueError("duplicate source packet keys")
    if set(reviewed)-set(all_packets):raise ValueError("reviewed key outside assigned source packet corpus")
    with (HERE/"crosswalk.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,delimiter="\t");w.writeheader();w.writerows(reviewed.values())
    with (HERE/"pending.tsv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=PENDING_FIELDS,delimiter="\t");w.writeheader();w.writerows(pending)
    file_reviews=[]
    for name in ("evidence-batch1.json","evidence-batch3.json","evidence-batch4.json","evidence-batch5.json","evidence-batch6.json","evidence-batch7.json","evidence-batch8.json","evidence-batch9.json","evidence-batch10.json"):
        file_reviews.extend(json.load((HERE/name).open())["file_reviews"])
    by_path={}
    for item in file_reviews:
        path=item["path"]
        actual=hashlib.sha256((SOURCE/path).read_bytes()).hexdigest()
        if actual!=item["source_sha256"]:raise ValueError(f"source changed: {path}")
        by_path[path]=item
    evidence={"snapshot":"vllm-20261105","assigned_entries":len(all_packets),"reviewed_entries":len(reviewed),"pending_entries":len(pending),"file_reviews":list(by_path.values()),"extraction":{"located":10641,"syntax_error":36,"definition_not_located":8,"source_packets":"source-packets.jsonl","canonical_family_index":"canonical-families.tsv"},"exclusion":"vLLM tests/v1, tests/distributed, source_test_file_unresolved and Rust allocated elsewhere","batch_evidence":["evidence-batch1.json","evidence-batch3.json","evidence-batch4.json","evidence-batch5.json","evidence-batch6.json","evidence-batch7.json","evidence-batch8.json","evidence-batch9.json","evidence-batch10.json"],"note":"crosswalk contains only source-reviewed batches 1, 3, 4, 5, 6, 7, 8, 9 and 10; derender batch2 remains provisional and is excluded."}
    evidence["source_reviewed_pending_entries"]=len(set(known_pending)-set(reviewed))
    evidence["review_stage_note"]="The assigned corpus was source-scanned; only completed batch entries are source-reviewed. Mapped counts do not include scan-only pending entries or unverified batch2."
    (HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    needs=[]
    for name in ("needs-space-batch1.json","needs-space-batch3.json","needs-space-batch4.json","needs-space-batch6.json","needs-space-batch8.json"):
        payload=json.load((HERE/name).open())
        needs.extend(payload if isinstance(payload,list) else payload["candidates"])
    seed=json.load((HERE.parents[2]/"space_seed.json").open())
    accepted={u["id"] for u in seed["units"]}
    for candidate in needs:
        if candidate["id"] in accepted:
            candidate["current_disposition"]="accepted_in_seed; original proposal retained for provenance"
    (HERE/"needs-space.json").write_text(json.dumps(needs,ensure_ascii=False,indent=2))
    print(json.dumps({"assigned":len(all_packets),"reviewed":len(reviewed),"pending":len(pending),"reviewed_files":len(by_path),"needs_space":len(needs)}))

if __name__=="__main__":main()
