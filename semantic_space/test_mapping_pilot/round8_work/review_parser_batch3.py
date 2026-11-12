"""Source-grounded local review of pinned vLLM parser and rendering candidates.

This writes only independent round-eight artifacts.  A reviewed row records
the *asserted subproperty*, not proof of its whole semantic-space obligation.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from review_tool_parser_batch1 import FIELDS, REPOS, rows, test_nodes

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
SPACE = PILOT.parent / "space_seed.json"
PARSER_FILES = {
    "tests/parser/test_harmony.py": 39,
    "tests/parser/test_streaming.py": 21,
    "tests/parser/test_parse.py": 19,
    "tests/reasoning/test_minimax_m3_reasoning_parser.py": 15,
    "tests/parser/engine/test_llama_json.py": 23,
    "tests/parser/engine/test_plamo3.py": 22,
    "tests/parser/engine/test_minicpmv_reasoning.py": 22,
}
RENDER_FILES = {
    "tests/renderers/test_inkling.py": 21,
    "tests/tokenizers_/test_deepseek_v4.py": 21,
    "tests/tokenizers_/test_deepseek_v41.py": 22,
}
FILES = PARSER_FILES | RENDER_FILES
FILE_REVIEWS = {
    "tests/parser/test_harmony.py": "Complete and chunked Harmony parsing checks reasoning/content/tool separation; four token-count checks need a distinct counting unit.",
    "tests/parser/test_streaming.py": "Parser-composition tests assert emitted reasoning, content, tool calls and tool-choice fallback under fixed chunks; they do not inspect constraint schemas.",
    "tests/parser/test_parse.py": "Complete parser and prompt-boundary checks assert reasoning/content/tool fields; one history-count helper has a different target.",
    "tests/reasoning/test_minimax_m3_reasoning_parser.py": "Fixed marker and token-ID cases assert reasoning/content split and end-state transitions, including split markers.",
    "tests/parser/engine/test_llama_json.py": "JSON tool-parser tests assert call name, arguments, fallback and chunk invariance; registry alias check is not parser behavior.",
    "tests/parser/engine/test_plamo3.py": "Plamo parser cases compare complete versus streamed reasoning/content/tool fields and partial marker behavior.",
    "tests/parser/engine/test_minicpmv_reasoning.py": "Non-thinking and thinking cases assert extracted content and tool calls; standalone newline normalizer and registry checks need different units.",
    "tests/renderers/test_inkling.py": "One entry compares complete token IDs; other entries compare rendered prompt text or configuration without token-ID oracle.",
    "tests/tokenizers_/test_deepseek_v4.py": "All sampled entries inspect rendered text, configuration, or exceptions rather than a reference token-ID sequence.",
    "tests/tokenizers_/test_deepseek_v41.py": "Two entries check actual media UUID/payload order or absence; remaining entries predominantly inspect prompt text and need a rendering unit.",
}
EXCLUDE = {
    ("tests/parser/test_harmony.py", "TestCountReasoningTokens::test_matches_process_chunk"): "reasoning token count needs a narrower unit than response usage",
    ("tests/parser/test_harmony.py", "TestCountReasoningTokens::test_excludes_harmony_control_tokens"): "reasoning token count needs a narrower unit than response usage",
    ("tests/parser/test_harmony.py", "TestCountReasoningTokens::test_final_only_has_no_reasoning"): "reasoning token count needs a narrower unit than response usage",
    ("tests/parser/test_harmony.py", "TestCountReasoningTokens::test_streaming_reuses_running_total"): "reasoning token count and caching need a narrower unit",
    ("tests/parser/test_parse.py", "test_count_history_tool_calls_responses_request"): "history tool-call count, not parsed output call relation",
    ("tests/parser/engine/test_llama_json.py", "test_registered_tool_parser_aliases"): "registry class-name check, not format-selection behavior",
    ("tests/parser/engine/test_minicpmv_reasoning.py", "test_reasoning_parser_is_registered"): "registry class-name check, not parser behavior",
    ("tests/parser/engine/test_minicpmv_reasoning.py", "TestNewlineRecovery::test_recover_newlines"): "text newline normalization needs a distinct unit",
}
RENDER_INCLUDE = {
    ("tests/renderers/test_inkling.py", "test_render_inkling_messages_direct_protocol"): "input.chat_template_tokenization",
    ("tests/tokenizers_/test_deepseek_v41.py", "test_tool_images_follow_rendered_order"): "input.multimodal_binding",
    ("tests/tokenizers_/test_deepseek_v41.py", "test_dropped_developer_only_image_leaves_text_prompt"): "input.multimodal_binding",
}


def source_calls(node: ast.AST) -> list[str]:
    names = []
    for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
        try:
            name = ast.unparse(call.func)
        except Exception:
            continue
        if name not in names:
            names.append(name)
    return names


def unit_for(row: dict[str, str]) -> str:
    path, name = row["path"], row["test_name"]
    if (path, name) in RENDER_INCLUDE:
        return RENDER_INCLUDE[path, name]
    proposed = row["proposed_unit"]
    if path == "tests/parser/test_harmony.py":
        if name.startswith("TestParse::") and proposed == "output.incremental_parse_relation":
            return "output.channel_event_relation"
        if name == "TestCountReasoningTokens::test_does_not_disturb_stream_state":
            return "output.incremental_parse_relation"
    if path == "tests/parser/test_parse.py" and proposed == "output.incremental_parse_relation":
        if name != "test_prompt_reasoning_streaming_matches_non_streaming":
            return "output.channel_event_relation"
    if path == "tests/parser/engine/test_minicpmv_reasoning.py":
        if name.startswith("TestNonThinking::") and "streaming" not in name and proposed == "output.incremental_parse_relation":
            return "output.channel_event_relation"
        if name.startswith("TestThinking::") and "streaming" not in name and proposed == "output.incremental_parse_relation":
            return "output.channel_event_relation"
        if name == "TestNewlineRecovery::test_parse_recovers_content":
            return "output.channel_event_relation"
    if path == "tests/parser/engine/test_plamo3.py" and name in {
        "test_duplicate_think_end_does_not_leak_into_content",
        "test_reusing_parser_drops_pending_names",
    }:
        return "output.channel_event_relation" if "duplicate" in name else "tool.call_parse_relation"
    if proposed == "tool.constraint_schema_construction":
        # These source tests feed an existing parser tool_choice and check
        # returned calls/content. They never inspect a generated schema.
        return "tool.call_parse_relation"
    if path == "tests/parser/engine/test_llama_json.py":
        if name in {"test_parse_invalid_json_falls_back_to_content",
                    "test_adapter_invalid_json_falls_back_to_content"}:
            return "tool.call_parse_relation"
        if name == "test_arg_converter_accepts_arguments_and_parameters":
            return "tool.call_parse_relation"
    if path == "tests/parser/engine/test_minicpmv_reasoning.py":
        if name == "TestNewlineRecovery::test_streaming_does_not_modify_tool_arguments":
            return "tool.call_parse_relation"
    return proposed


def main() -> dict[str, object]:
    candidates = [r for r in rows(HERE / "candidates.tsv")
                  if r["snapshot"] == "vllm" and r["path"] in FILES]
    groups = {path: [r for r in candidates if r["path"] == path] for path in FILES}
    assert len(candidates) == 225
    assert all(len(groups[path]) == count for path, count in FILES.items())
    known = set()
    for filename in ("direct-property-crosswalk.tsv", "confirmed-property-crosswalk.tsv",
                     "inherited-property-crosswalk.tsv"):
        known.update((r["snapshot"], r["path"], r["test_name"], r["line"])
                     for r in rows(PILOT / filename))
    for filename in ("reviewed-reference-batch1.tsv", "reviewed-sampling-batch1.tsv",
                     "reviewed-tool-parser-batch1.tsv", "reviewed-tool-parser-batch2.tsv",
                     "reviewed-ngram-draft-batch1.tsv"):
        known.update((r["snapshot"], r["path"], r["test_name"], r["line"])
                     for r in rows(HERE / filename))
    ontology = {u["id"] for u in json.loads(SPACE.read_text(encoding="utf-8"))["units"]}
    reviewed, held, files = [], [], {}
    for path in FILES:
        source = (REPOS["vllm"] / path).read_bytes()
        sha = hashlib.sha256(source).hexdigest()
        lines = source.decode("utf-8").splitlines()
        nodes = test_nodes(source)
        files[path] = {"sha256": sha, "candidate_entries": len(groups[path])}
        for row in groups[path]:
            key = (row["snapshot"], path, row["test_name"], row["line"])
            assert key not in known, key
            assert row["source_sha256"] == sha, key
            node = nodes[row["test_name"]]
            assert node.lineno == int(row["line"]), key
            sites = sorted({int(n) for n in row["oracle_lines"].split(";")})
            assert sites and all(node.lineno < n <= node.end_lineno for n in sites), key
            assert all(any(isinstance(x, (ast.Assert, ast.With, ast.AsyncWith))
                           and x.lineno <= n <= x.end_lineno for x in ast.walk(node))
                       for n in sites), key
            calls = source_calls(node)
            evidence = {
                "snapshot": "vllm", "path": path, "test_name": row["test_name"],
                "line": int(row["line"]), "source_sha256": sha,
                "oracle_lines": sites, "oracle_text": [lines[n - 1].strip() for n in sites],
                "source_calls": calls,
            }
            if (path, row["test_name"]) in EXCLUDE:
                evidence.update(status="needs_refinement", reason=EXCLUDE[path, row["test_name"]])
                held.append(evidence)
                continue
            if path in RENDER_FILES and (path, row["test_name"]) not in RENDER_INCLUDE:
                evidence.update(status="needs_refinement", reason=(
                    "asserts rendered prompt text, configuration, or exception, but existing "
                    "chat-template-tokenization unit requires token-ID correspondence; "
                    "multimodal binding requires actual item/position relation"))
                held.append(evidence)
                continue
            unit = unit_for(row)
            assert unit in ontology, (key, unit)
            doc = ast.get_docstring(node)
            summary = (doc.splitlines()[0] if doc else
                       row["test_name"].split("::")[-1].removeprefix("test_").replace("_", " "))
            condition = ("Pinned local parser input, parser configuration and chunk boundaries "
                         "in cited source; only the listed assertions are claimed; no server or "
                         "model execution is inferred.")
            if path in RENDER_FILES:
                condition = ("Pinned local renderer input and tokenizer/media stub in cited source; "
                             "only the listed token IDs or media order/absence are claimed.")
            reviewed.append({
                "snapshot": "vllm", "path": path, "test_name": row["test_name"],
                "line": row["line"], "unit_id": unit,
                "property": summary + ". Source asserts: " + row["checks"],
                "oracle_source": ";".join(f"{path}:{n}" for n in sites),
                "conditions": condition,
                "rule_id": "round8_parser_batch3_source_review",
            })
            evidence.update(status="locally_reviewed", unit_id=unit)
            held.append(evidence)
    assert len(reviewed) + sum(e["status"] == "needs_refinement" for e in held) == 225
    assert len(reviewed) >= 150
    assert len({(r["snapshot"], r["path"], r["test_name"], r["line"])
                for r in reviewed}) == len(reviewed)
    out = HERE / "reviewed-parser-batch3.tsv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(reviewed, key=lambda r: (r["path"], int(r["line"]))))
    record = {"source_files": files,
              "file_reviews": [{"snapshot": "vllm", "path": path,
                                "source_sha256": files[path]["sha256"],
                                "candidate_entries": files[path]["candidate_entries"],
                                "review": FILE_REVIEWS[path]}
                               for path in FILES],
              "entries": held,
              "summary": {"candidates": 225, "locally_reviewed": len(reviewed),
                          "needs_refinement": 225 - len(reviewed),
                          "reviewed_units": dict(sorted(Counter(r["unit_id"] for r in reviewed).items())),
                          "published_green_added": 0}}
    (HERE / "reviewed-parser-batch3-evidence.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], ensure_ascii=False))
    return record["summary"]


if __name__ == "__main__":
    main()
