"""Compact audit ledger from the full source packets; no semantic assignment."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def walk(evidence, depth=0):
    if not evidence or depth > 2:
        return
    yield evidence
    for helper in evidence.get("local_helpers", {}).values():
        yield from walk(helper, depth + 1)


def main():
    profiles = defaultdict(lambda: {"count": 0, "oracles": Counter(), "calls": Counter(), "checks": [], "unlocated": 0, "helpers": 0})
    for line in (HERE / "source-packets.jsonl").open():
        item = json.loads(line)
        path = item["key"][1]
        p = profiles[path]
        p["count"] += 1
        if item.get("error"):
            p["unlocated"] += 1
        p["oracles"][item.get("oracle_forms_index", "")] += 1
        ev = item.get("evidence")
        p["helpers"] += len((ev or {}).get("local_helpers", {}))
        for x in walk(ev):
            for call in x.get("calls", []):
                name = call["name"]
                if name.startswith(("torch.", "np.", "pytest.", "self.", "client.", "requests.")) or "vllm" in name:
                    p["calls"][name] += 1
            for check in x.get("checks", []):
                if len(p["checks"]) < 8:
                    p["checks"].append(f"{check['line']}:{check['source'][:120]}")
    with (HERE / "file-profiles.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["path", "entries", "unlocated", "helper_edges", "oracle_forms", "top_calls", "sample_checks"])
        for path, p in sorted(profiles.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
            w.writerow([path, p["count"], p["unlocated"], p["helpers"], ";".join(f"{k}:{n}" for k,n in p["oracles"].most_common(4)), ";".join(f"{k}:{n}" for k,n in p["calls"].most_common(10)), ";".join(p["checks"])])
    print(len(profiles), sum(x["count"] for x in profiles.values()))


if __name__ == "__main__":
    main()
