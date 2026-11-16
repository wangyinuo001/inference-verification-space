# -*- coding: utf-8 -*-
"""Human-reviewed SGLang embedding-override source cluster."""

import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PATH = "test/registered/prefill_only/test_embed_overrides.py"
SOURCE = ROOT / "experiments/third_party/reference_repos/sglang-v0.5.14-complete" / PATH
PENDING = HERE / "pending.tsv"
cols = ["snapshot", "path", "test_name", "line", "unit_id", "property", "oracle_source", "conditions", "rule_id"]


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    rows = list(csv.DictReader(PENDING.open(newline=""), delimiter="\t"))
    mappings, proposed = [], []
    keep = []
    for row in rows:
        if row["path"] != PATH:
            keep.append(row)
            continue
        name = row["test_name"]
        test_class = name.split("::")[0]
        oracle = row["source_evidence"].split(";")
        oracle = ";".join(item.split(":assert")[0].rsplit(":", 1)[0] if item.endswith(":assert") else item.rsplit(":", 1)[0] for item in oracle if item)
        # All rows were read against the complete pinned source, including the
        # helper implementations called by the tests. The assertion excerpt is
        # retained in the property for case-specific review.
        if test_class in ("TestGenerateReqInputEmbedOverride", "TestEmbeddingReqInputEmbedOverride"):
            unit = "request.batch_field_binding"
            basis = "Batched request slicing preserves per-item embedding override fields and positions."
        elif test_class == "TestScoreRequestValidation":
            unit = "request.admission"
            basis = "Score request validation accepts or rejects the specified label, item, and embedding-override combination."
        elif test_class in ("TestPositionalEmbeds", "TestConvertEmbedsToTensors", "TestResolveEmbedOverrides", "TestResolveOverridesForSequence", "TestResolveEmbedOverridesForRequest", "TestBuildTokenIdInputs"):
            unit = "input.embedding_override_binding"
            basis = {
                "TestPositionalEmbeds": "Positional embedding representation preserves the asserted tensor shape, values, identity or rejects mismatched position count, according to this case.",
                "TestConvertEmbedsToTensors": "Input embedding lists are converted to tensors with the asserted nesting, value and dtype, according to this case.",
                "TestResolveEmbedOverrides": "Embedding override positions correspond to the specified placeholder token occurrences, or invalid occurrence counts are rejected.",
                "TestResolveOverridesForSequence": "Embedding override positions correspond to the specified token sequence and offset, or invalid counts are rejected.",
                "TestResolveEmbedOverridesForRequest": "Query and item override positions correspond to their specified sequence offsets.",
                "TestBuildTokenIdInputs": "Scoring request construction preserves the asserted query/item token order and positional override indices.",
            }[test_class]
        else:
            row["triage_status"] = "needs_individual_review"
            row["review_reason"] = "Unrecognized test class in embedding-override source cluster."
            keep.append(row)
            continue
        if not row["assertion_excerpt"]:
            row["triage_status"] = "needs_helper_or_runtime_oracle"
            row["review_reason"] = "No direct assertion excerpt to ground this source mapping."
            keep.append(row)
            continue
        out = {"snapshot": "sglang", "path": PATH, "test_name": name, "line": row["line"],
               "unit_id": unit,
               "property": basis + " Observed check: " + row["assertion_excerpt"],
               "oracle_source": oracle,
               "conditions": "Pinned SGLang 0.5.14; local test inputs and mocked scoring request only; checks do not establish full model output.",
               "rule_id": "sglang_embed_override_source_review_" + test_class}
        (proposed if unit == "input.embedding_override_binding" else mappings).append(out)
    with (HERE / "crosswalk.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=cols);w.writeheader();w.writerows(mappings)
    with (HERE / "proposed-crosswalk.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=cols);w.writeheader();w.writerows(proposed)
    with PENDING.open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=rows[0].keys());w.writeheader();w.writerows(keep)
    needs = [{"id": "input.embedding_override_binding", "kind": "obligation", "phase": "admission", "state_scope": "request", "name": "Embedding override vectors bind to their declared token positions", "property": "Provided embedding vectors replace exactly the intended placeholder positions in each query/item sequence; batching and ordering preserve this binding, while inconsistent counts are rejected.", "conditions": ["input_form", "request_kind"], "sglang_evidence": [f"{PATH}:114", "python/sglang/srt/managers/tokenizer_manager.py", "python/sglang/srt/managers/tokenizer_manager_score_mixin.py"], "vllm_evidence": ["vllm/inputs/llm.py:133", "tests/test_request_input_bounds.py:84", "tests/renderers/test_hf.py:92"], "difference": "SGLang uses a placeholder token and PositionalEmbeds for scoring query/items; vLLM accepts prompt_embeds and validates mask/placeholder positions in its input processor/renderer."}]
    (HERE / "needs-space.json").write_text(json.dumps(needs, ensure_ascii=False, indent=2) + "\n")
    evidence = json.loads((HERE / "evidence.json").read_text())
    evidence["reviewed_clusters"] = [{"path": PATH, "sha256": digest, "mapped_existing": len(mappings), "proposed_new_unit": len(proposed), "review_method": "Read full pinned test file, its test helpers, source imports, concrete assertions, and matched vLLM prompt embedding source/test behavior."}]
    evidence["total_pending_after_review"] = len(keep)
    (HERE / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print("mapped",len(mappings),"proposed",len(proposed),"pending",len(keep))


if __name__ == "__main__":
    main()
