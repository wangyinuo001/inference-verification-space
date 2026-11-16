"""Source-linked inherited end-to-end tool-choice checks across four SGLang models."""
import ast
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
PATH="test/manual/openai_server/function_call/test_tool_choice.py"
SOURCE=ROOT/"experiments/third_party/reference_repos/sglang-v0.5.14-complete"/PATH
COLS=["snapshot","path","test_name","line","unit_id","property","oracle_source","conditions","rule_id"]
MODELS={
"TestToolChoiceLlama32":("meta-llama/Llama-3.2-1B-Instruct","llama3"),
"TestToolChoiceQwen25":("Qwen/Qwen2.5-7B-Instruct","qwen25"),
"TestToolChoiceMistral":("mistralai/Mistral-7B-Instruct-v0.3","mistral"),
"TestToolChoiceLfm2":("LiquidAI/LFM2.5-1.2B-Instruct","lfm2"),
}
WEAK={"test_tool_choice_auto_non_streaming","test_tool_choice_auto_streaming"}
INVALID={"test_error_handling_invalid_tool_choice","test_invalid_tool_missing_name","test_invalid_json_schema_in_tool","test_conflicting_defs_required_tool_choice"}
SKIPPED_MISTRAL={"test_tool_choice_required_non_streaming","test_multi_tool_scenario_required","test_complex_parameters_required_non_streaming"}

def main():
 text=SOURCE.read_text();tree=ast.parse(text)
 tests={n.lineno:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name.startswith("test_")}
 rows=list(csv.DictReader((HERE/"pending.tsv").open(newline=""),delimiter="\t"));keep=[];out=[]
 for row in rows:
  if row["path"]!=PATH:keep.append(row);continue
  cls,name=row["test_name"].split("::",1)
  node=tests.get(int(row["definition_line"]))
  if node is None or name!=node.name or cls not in MODELS:keep.append(row);continue
  if cls=="TestToolChoiceMistral" and name in SKIPPED_MISTRAL:
   row["review_reason"]="Mistral subclass marks this inherited test with unittest.skip; no runtime check is executed."
   keep.append(row);continue
  if name in WEAK:
   row["review_reason"]="Auto-choice test only checks message non-None or locally constructed Python lists; it does not assert any tool call, name or argument relation."
   keep.append(row);continue
  if name=="test_multi_tool_scenario_auto" and cls!="TestToolChoiceQwen25":
   row["review_reason"]="Flaky-mode branch allows no tool call and checks names only conditionally; source does not establish a stable tool-choice oracle."
   keep.append(row);continue
  # The inherited base method provides the request and oracle. The model subclass
  # supplies a separate pinned parser/model condition; never assume equal outcomes.
  calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and "chat.completions.create" in ast.unparse(n.func)]
  assertions=sorted((n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith("assert")),key=lambda n:n.lineno)
  if not calls or not assertions:
   row["review_reason"]="Inherited base test lacks linked API call plus direct semantic assertion."
   keep.append(row);continue
  unit="tool.request_admissibility" if name in INVALID else "tool.choice_output_relation"
  snippets=[" ".join((ast.get_source_segment(text,n) or "").split())[:180] for n in assertions]
  model,parser=MODELS[cls]
  notes=[]
  if name=="test_multi_tool_scenario_auto":notes.append("non-flaky Qwen branch requires exactly two named tool calls")
  if name=="test_required_streaming_arguments_chunks_json":notes.append("reassembles argument fragments before JSON-object assertion")
  if name in INVALID:notes.append("requires BadRequestError plus asserted message fragment")
  prop=("Request rejects malformed tool declaration/choice as asserted: " if name in INVALID else "Service tool-choice result satisfies the asserted call/name/argument relation: ")+" | ".join(snippets)
  out.append({"snapshot":"sglang","path":PATH,"test_name":row["test_name"],"line":row["line"],
   "unit_id":unit,"property":prop,
   "oracle_source":";".join(f"{PATH}:{n.lineno}" for n in assertions),
   "conditions":f"Pinned SGLang 0.5.14 service; model={model}; tool-call-parser={parser}; inherited base test at {PATH}:{node.lineno}; request body, stream flag and tool_choice in source; "+"; ".join(notes),
   "rule_id":"sglang_tool_choice_inherited_source_review"})
 with (HERE/"proposed-crosswalk.tsv").open("a",newline="") as f:csv.DictWriter(f,fieldnames=COLS,delimiter="\t").writerows(out)
 with (HERE/"pending.tsv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys(),delimiter="\t");w.writeheader();w.writerows(keep)
 evidence=json.loads((HERE/"evidence.json").read_text());evidence.setdefault("reviewed_clusters",[]).append({"path":PATH,"sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"proposed_new_unit":len(out),"exceptions":sum(r["path"]==PATH for r in keep),"review_method":"Inherited base test API call and actual assertion inspected; subclass model/parser and unittest.skip/flaky branches accounted separately."});evidence["total_pending_after_review"]=len(keep);(HERE/"evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
 print("proposed",len(out),"remaining in file",sum(r["path"]==PATH for r in keep))
if __name__=="__main__":main()
