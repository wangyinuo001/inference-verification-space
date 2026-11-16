"""Review the complete SGLang model-gateway router argument test file."""

import ast
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PATH = "sgl-model-gateway/bindings/python/tests/test_arg_parser.py"
SOURCE = ROOT / "experiments/third_party/reference_repos/sglang-v0.5.14-complete" / PATH
COLS = ["snapshot", "path", "test_name", "line", "unit_id", "property", "oracle_source", "conditions", "rule_id"]


def main():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    tests = {n.lineno: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    rows = list(csv.DictReader((HERE / "pending.tsv").open(newline=""), delimiter="\t"))
    retained, mapped = [], []
    for row in rows:
        if row["path"] != PATH:
            retained.append(row)
            continue
        test = tests.get(int(row["line"]))
        if test is None:
            row["review_reason"] = "No matching test body at pinned source line."
            retained.append(row)
            continue
        if test.name in {"test_parse_tokenizer_args", "test_help_output"}:
            row["review_reason"] = ("Test invokes pytest.skip before a product call or oracle; no property is checked."
                                    if test.name == "test_parse_tokenizer_args" else
                                    "Help invocation checks clean exit, not parsed effective configuration; separate CLI help property needed.")
            retained.append(row)
            continue
        assertions = sorted((n for n in ast.walk(test) if isinstance(n, ast.Assert)), key=lambda n: n.lineno)
        exceptions = sorted((n for n in ast.walk(test) if isinstance(n, ast.With) and any("pytest.raises" in ast.unparse(i.context_expr) for i in n.items)), key=lambda n: n.lineno)
        if not assertions and not exceptions:
            row["review_reason"] = "No direct assertion or expected exception in test body."
            retained.append(row)
            continue
        calls = [n for n in ast.walk(test) if isinstance(n, ast.Call) and any(s in ast.unparse(n.func) for s in ("RouterArgs", "parse_router_args", "policy_from_str"))]
        if not calls:
            row["review_reason"] = "No direct router argument parser or policy conversion call."
            retained.append(row)
            continue
        invalid = test.name in {"test_parse_selector_invalid", "test_parse_prefill_urls_invalid", "test_invalid_policy", "test_parse_invalid_args"}
        # The malformed-selector test checks a documented normalization result, not rejection.
        if test.name == "test_parse_selector_invalid":
            invalid = False
        unit = "deployment.configuration_admissibility" if invalid else "deployment.effective_configuration"
        property_bits = [ast.unparse(n.test) for n in assertions[:8]]
        for n in exceptions:
            body = [ast.unparse(x) for x in n.body[:1]]
            property_bits.append("expect " + ast.unparse(n.items[0].context_expr) + " from " + "; ".join(body))
        prop = "Router argument parsing preserves the asserted fields and conversions: " if not invalid else "Malformed router arguments are rejected as asserted: "
        prop += " | ".join(property_bits)
        condition_calls = [ast.unparse(n) for n in sorted(calls, key=lambda n: n.lineno)[:2]]
        mapped.append({
            "snapshot": "sglang", "path": PATH, "test_name": row["test_name"], "line": row["line"],
            "unit_id": unit, "property": prop,
            "oracle_source": ";".join(f"{PATH}:{n.lineno}" for n in (assertions + exceptions)),
            "conditions": "Pinned SGLang 0.5.14 model gateway; " + " | ".join(condition_calls)[:400] + "; checks parser/configuration values, not a running gateway.",
            "rule_id": "sglang_router_args_source_review",
        })
    with (HERE / "crosswalk.tsv").open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=COLS, delimiter="\t").writerows(mapped)
    with (HERE / "pending.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(retained)
    evidence = json.loads((HERE / "evidence.json").read_text())
    evidence.setdefault("reviewed_clusters", []).append({
        "path": PATH, "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "mapped_existing_unit": len(mapped), "exceptions": sum(r["path"] == PATH for r in retained),
        "review_method": "Complete pinned router-argument test file and parser implementation read; direct call plus direct field assertion or expected exception checked per test."
    })
    evidence["total_pending_after_review"] = len(retained)
    (HERE / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print("mapped", len(mapped), "remaining in file", sum(r["path"] == PATH for r in retained))


if __name__ == "__main__":
    main()
