"""Re-open every indexed source and reconcile review, space and run records.

Every entry receives an explicit disposition. Source existence, execution and
semantic review are separate fields; this audit never invents a property match.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE = HERE.parent
PROJECT = HERE.parents[3]
ROOTS = {"vllm": PROJECT / "experiments/third_party/reference_repos/vllm-20261105",
         "sglang": PROJECT / "experiments/third_party/reference_repos/sglang-v0.5.14-complete"}
OUT = HERE / "full_delivery"
KEY = ("snapshot", "path", "test_name", "line")
LABELS = {
    "property_reviewed": "已核对具体检查内容并归类",
    "outside_current_scope": "暂不属于推理验证",
    "locate_definition": "仍按文件记录，需定位测试定义",
    "locate_check_or_helper": "需定位断言或辅助检查",
    "refine_property_or_space": "已回看源码，需细分性质或补充空间",
    "review_call_inputs_and_property": "已有源码位置，待核对调用、输入与性质",
}


def read(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def key(row):
    return tuple(row[k] for k in KEY)


def write_tsv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    index = read(HERE / "full-scan-index.tsv")
    review = {key(r): r for r in read(SPACE / "oracle-review-queue.tsv")}
    inherited = {key(r): r for r in read(HERE / "inherited-method-sources.tsv")}
    runtime = {key(r): r for r in read(HERE / "background_evidence/runtime-entries.tsv")}
    pending_path = HERE / "reviewed-supplement-pending.json"
    pending = {key(r): r for r in json.loads(pending_path.read_text())} if pending_path.exists() else {}
    checked = {}
    for name in ("confirmed-property-crosswalk.tsv", "direct-property-crosswalk.tsv",
                 "inherited-property-crosswalk.tsv", "reviewed-supplement-crosswalk.tsv"):
        for row in read(HERE / name):
            assert key(row) not in checked, (name, key(row))
            checked[key(row)] = row
    units = {u["id"] for u in json.loads((SPACE / "space_seed.json").read_text())["units"]}
    cache = {}

    def source(snapshot, path):
        k = snapshot, path
        if k not in cache:
            file = ROOTS[snapshot] / path
            if not file.is_file():
                cache[k] = {"status": "missing_file", "hash": "", "nodes": {}, "lines": []}
            else:
                raw = file.read_bytes(); text = raw.decode("utf-8", errors="replace")
                nodes = {}; status = "non_python"
                if file.suffix == ".py":
                    try:
                        tree = ast.parse(text)
                        nodes = {n.lineno: n for n in ast.walk(tree)
                                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                        status = "parsed"
                    except SyntaxError:
                        status = "syntax_unresolved"
                cache[k] = {"status": status, "hash": hashlib.sha256(raw).hexdigest(),
                            "nodes": nodes, "lines": text.splitlines()}
        return cache[k]

    rows = []; issues = []; groups = defaultdict(list)
    for r in index:
        k = key(r); observed = review[k]; confirmed = checked.get(k)
        src = source(r["snapshot"], r["path"])
        origin = inherited.get(k)
        source_path, source_line = r["path"], r["line"]
        if r["entry_kind"] == "python_inherited_test_candidate" and origin and origin["status"] == "resolved_unique":
            source_path, source_line = origin["source_path"], origin["source_line"]
            definition = source(r["snapshot"], source_path)
            node = definition["nodes"].get(int(source_line))
            location = "inherited_definition_located" if node and node.name == r["test_name"].split("::")[-1] else "definition_needs_review"
        elif r["entry_kind"] == "python_test_definition":
            node = src["nodes"].get(int(r["line"]))
            location = "definition_located" if node and node.name == r["test_name"].split("::")[-1] else "definition_needs_review"
        elif r["entry_kind"] == "rust_test_definition":
            location = "rust_source_line_located" if 0 < int(r["line"]) <= len(src["lines"]) else "definition_needs_review"
        else:
            location = "file_only" if src["hash"] else "missing_file"
        known = confirmed is not None or (observed["source_review_status"] == "mapped" and bool(observed["source_reviewed_units"]))
        unit_text = confirmed["unit_id"] if confirmed else observed["source_reviewed_units"] if known else ""
        property_text = confirmed["property"] if confirmed else observed["source_reviewed_property"] if known else ""
        if known:
            for uid in re.split(r"[;,]", unit_text):
                if uid and uid not in units:
                    issues.append({"key": k, "issue": "unknown_unit", "unit": uid})
            if not property_text:
                issues.append({"key": k, "issue": "empty_property"})
            if location in {"definition_needs_review", "missing_file", "file_only"}:
                issues.append({"key": k, "issue": "reviewed_definition_unresolved"})
            if confirmed:
                for site in confirmed["oracle_source"].split(";"):
                    site = site.strip()
                    path, line = site.rsplit(":", 1)
                    oracle = source(r["snapshot"], path)
                    if not 0 < int(line) <= len(oracle["lines"]):
                        issues.append({"key": k, "issue": "oracle_outside_source", "site": site})
            decision = "property_reviewed"
        elif r["scan_route"] == "reviewed_out_of_request_scope":
            decision = "outside_current_scope"
        elif location in {"definition_needs_review", "missing_file", "file_only"}:
            decision = "locate_definition"
        elif not r["oracle_forms"]:
            decision = "locate_check_or_helper"
        elif k in pending or observed["source_review_status"] in {"needs_refinement", "sample_reviewed", "oracle_unresolved"}:
            decision = "refine_property_or_space"
        else:
            decision = "review_call_inputs_and_property"
        runtime_record = runtime.get(k, {})
        row = {**{f: r[f] for f in KEY}, "entry_kind": r["entry_kind"],
               "source_status": location, "source_sha256": src["hash"],
               "definition_path": source_path, "definition_line": source_line,
               "review_status": decision, "unit_ids": unit_text, "property": property_text,
               "review_status_label": LABELS[decision],
               "runtime_instances": runtime_record.get("instances", "0"),
               "call_and_assertion_instances": runtime_record.get("calls_and_assertions", "0"),
               "oracle_forms": r["oracle_forms"], "object_candidates": r["object_candidates"]}
        rows.append(row)
        if decision not in {"property_reviewed", "outside_current_scope"}:
            group = r["snapshot"], str(Path(r["path"]).parent), decision, r["object_candidates"], r["oracle_forms"]
            groups[group].append(k)
    assert len(rows) == len({key(r) for r in rows}) == len(index) == 31968
    assert set(checked) <= {key(r) for r in index}
    summary = {"source_entries": len(rows), "indexed_source_files": len({(r['snapshot'],r['path']) for r in rows}),
               "review_status": dict(Counter(r["review_status"] for r in rows)),
               "source_status": dict(Counter(r["source_status"] for r in rows)),
               "runtime_entries": sum(int(r["runtime_instances"]) > 0 for r in rows),
               "pending_groups": len(groups), "consistency_issues": len(issues),
               "interpretation": "All source entries re-opened and accounted for; property_reviewed alone records reviewed semantic placement."}
    write_tsv(OUT / "all-entries.tsv", rows, tuple(rows[0]))
    group_rows = [{"snapshot": k[0], "directory": k[1], "next_work": k[2], "object_candidates": k[3],
                   "next_work_label": LABELS[k[2]],
                   "oracle_forms": k[4], "entries": len(v), "example": ":".join(v[0][1:])}
                  for k, v in sorted(groups.items(), key=lambda p: (-len(p[1]), p[0]))]
    write_tsv(OUT / "pending-groups.tsv", group_rows, tuple(group_rows[0]))
    for name, data in (("summary.json", summary), ("consistency-issues.json", issues)):
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(r["review_status"] for r in rows)
    report = "# 全量入口核对\n\n本次重新读取全部 31,968 个入口对应的源码，核对定义位置、继承方法来源、已审性质和远端运行记录。源码定位与语义审查分开记录；这次全量读取不等于全部条目已完成人工语义审查。\n\n| 状态 | 入口数 |\n|---|---:|\n"
    report += "".join(f"| {LABELS[status]} | {counts[status]:,} |\n" for status in LABELS)
    report += "\n[逐项记录](all-entries.tsv)包含源码哈希、测试定义、检查性质及运行实例数；[待核对分组](pending-groups.tsv)按仓库、目录和检查方式组织后续工作。运行记录只表示在本次隔离环境中观察到执行，不自动改变归类状态。\n"
    (OUT / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
