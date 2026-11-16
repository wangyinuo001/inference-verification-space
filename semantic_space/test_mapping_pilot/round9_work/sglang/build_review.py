# -*- coding: utf-8 -*-
"""Source-backed triage of pinned SGLang test definitions.

This pass records what source evidence exists. It deliberately does not turn
the presence of an assertion or an import into a verified property mapping.
"""

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
TABLE = HERE.parents[1] / "full_delivery" / "all-entries.tsv"
SOURCE = ROOT / "experiments/third_party/reference_repos/sglang-v0.5.14-complete"


def import_lines(tree, stem):
    out = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(stem):
            out.append(node.lineno)
        elif isinstance(node, ast.Import):
            if any(item.name.startswith(stem) for item in node.names):
                out.append(node.lineno)
    return out


def node_for_line(tree, line):
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= node.end_lineno:
            hits.append(node)
    return min(hits, key=lambda n: n.end_lineno - n.lineno) if hits else None


def direct_assertions(node):
    if node is None:
        return []
    rows = []
    for item in ast.walk(node):
        if isinstance(item, ast.Assert):
            rows.append((item.lineno, "assert"))
        elif isinstance(item, ast.Call):
            try:
                name = ast.unparse(item.func)
            except Exception:
                continue
            if name.startswith(("self.assert", "torch.testing.assert", "np.testing.assert", "numpy.testing.assert", "pytest.raises")):
                rows.append((item.lineno, name))
    return sorted(set(rows))


def source_snippet(lines, line):
    return lines[line - 1].strip() if 0 < line <= len(lines) else ""


def main():
    entries = []
    with TABLE.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if (row["snapshot"] == "sglang"
                and row["review_status"] not in ("property_reviewed", "outside_current_scope")
                and row["entry_kind"] in ("python_test_definition", "python_inherited_test_candidate")):
                entries.append(row)
    by_file = {}
    for row in entries:
        by_file.setdefault(row["path"], []).append(row)
    pending = []
    file_evidence = {}
    for path, rows in sorted(by_file.items()):
        source_path = SOURCE / path
        data = source_path.read_bytes()
        source_lines = data.decode("utf-8", errors="replace").splitlines()
        digest = hashlib.sha256(data).hexdigest()
        try:
            tree = ast.parse(data.decode("utf-8"), filename=path)
            parse_error = ""
        except (SyntaxError, UnicodeDecodeError) as exc:
            tree = None
            parse_error = str(exc)
        if tree is None:
            gen_imports = []
            debug_imports = []
        else:
            gen_imports = import_lines(tree, "sglang.multimodal_gen")
            debug_imports = import_lines(tree, "sglang.srt.debug_utils")
        if path.startswith("python/sglang/multimodal_gen/") and gen_imports:
            purpose = "image_video_generation_subsystem"
            purpose_line = gen_imports[0]
            purpose_note = "Test imports the multimodal_gen generation subsystem; its behavior requires a separate image/video generation space, not the text-request path."
        elif path.startswith("test/registered/debug_utils/") and debug_imports:
            purpose = "diagnostic_instrumentation"
            purpose_line = debug_imports[0]
            purpose_note = "Test imports debug_utils instrumentation/comparison code; its checks concern diagnostic artifacts or tooling, not the request-path property itself."
        else:
            purpose = "inference_or_unresolved"
            purpose_line = 0
            purpose_note = "Source-level calls, inputs, helper assertions and expected values still need semantic linkage to a verification property."
        file_evidence[path] = {"sha256": digest, "entries": len(rows), "purpose": purpose, "purpose_source": f"{path}:{purpose_line}" if purpose_line else "", "parse_error": parse_error}
        for row in rows:
            definition_path = row["definition_path"] or path
            definition_line = int(row["definition_line"] or row["line"] or 0)
            def_tree = tree if definition_path == path else None
            if def_tree is None and definition_path != path:
                try:
                    def_tree = ast.parse((SOURCE / definition_path).read_text(encoding="utf-8"))
                except (OSError, SyntaxError, UnicodeDecodeError):
                    pass
            node = node_for_line(def_tree, definition_line) if def_tree else None
            asserts = direct_assertions(node)
            evidence = ";".join(f"{definition_path}:{ln}:{kind}" for ln, kind in asserts[:8])
            if definition_path == path:
                assertion_excerpt = " | ".join(source_snippet(source_lines, ln) for ln, _ in asserts[:5])
            else:
                try:
                    other_lines = (SOURCE / definition_path).read_text(encoding="utf-8").splitlines()
                except OSError:
                    other_lines = []
                assertion_excerpt = " | ".join(source_snippet(other_lines, ln) for ln, _ in asserts[:5])
            if purpose != "inference_or_unresolved":
                status = "separate_domain"
                reason = purpose_note
            elif row["entry_kind"] == "python_inherited_test_candidate":
                status = "needs_fixture_and_inheritance_review"
                reason = "Inherited test body and subclass fixtures/overrides must be jointly checked before assigning a property."
            elif not asserts:
                status = "needs_helper_or_runtime_oracle"
                reason = "No direct assertion found in the definition; inspect called helpers, exception context, or external server/client responses."
            else:
                status = "needs_call_oracle_link"
                reason = purpose_note
            pending.append({
                "snapshot": row["snapshot"], "path": path, "test_name": row["test_name"], "line": row["line"],
                "definition_path": definition_path, "definition_line": definition_line,
                "triage_status": status, "purpose": purpose,
                "source_evidence": evidence or (f"{path}:{purpose_line}" if purpose_line else f"{definition_path}:{definition_line}"),
                "assertion_excerpt": assertion_excerpt,
                "review_reason": reason, "source_sha256": digest,
            })
    columns = ["snapshot", "path", "test_name", "line", "definition_path", "definition_line", "triage_status", "purpose", "source_evidence", "assertion_excerpt", "review_reason", "source_sha256"]
    with (HERE / "pending.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=columns)
        writer.writeheader(); writer.writerows(pending)
    with (HERE / "scope-dispositions.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=columns)
        writer.writeheader(); writer.writerows(row for row in pending if row["triage_status"] == "separate_domain")
    summary = Counter(row["triage_status"] for row in pending)
    (HERE / "evidence.json").write_text(json.dumps({"source_snapshot": "sglang-v0.5.14-complete", "total_entries": len(entries), "triage_counts": dict(summary), "files": file_evidence, "rule_limit": "No property is confirmed by this source-structure triage; call-result-oracle and conditions require separate review."}, ensure_ascii=False, indent=2) + "\n")
    print(dict(summary))


if __name__ == "__main__":
    main()
