"""Conservative HTTP/SDK request-to-oracle source links for review.

The extractor never asserts an API's semantic correctness. It records only
source-located route calls and explicit value flow into test assertions.
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path

from ast_scope import lexical_body_nodes
from audit_all_test_bodies import python_nodes
from audit_call_oracle_links import call_name, check_sites, enclosing, lexical_parents


HERE = Path(__file__).resolve().parent
REPOS = HERE.parent / "third_party/reference_repos"
REPO_NAMES = {"vllm": "vllm-20261105", "sglang": "sglang-v0.5.14-complete"}
OUT = HERE / "agent_http_boundary.tsv"
REPORT = HERE / "agent_http_boundary.md"
FIELDS = [
    "snapshot", "path", "test_name", "line", "client_call_name",
    "route_calls_json", "nested_client_calls_json", "request_oracle_links_json", "link_status",
    "candidate_object", "source_route_evidence", "review_next", "verified_property",
]


def read(name: str) -> list[dict[str, str]]:
    with (HERE / name).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def assigned_names(node: ast.AST | None) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(assigned_names(x) for x in node.elts))
    return set()


def route_label(call: ast.Call) -> tuple[str, str]:
    name = call_name(call)
    if any(name.endswith("." + method) for method in ("get", "post", "put", "patch", "delete", "request", "stream")):
        first = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg in {"url", "path"}), None)
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return "literal_endpoint", first.value
        if isinstance(first, ast.JoinedStr):
            return "templated_endpoint", ast.unparse(first)[:200]
        return "dynamic_endpoint", ast.unparse(first)[:200] if first is not None else ""
    return "sdk_method", name


def main_value_assign(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> tuple[set[str], str]:
    assignment = enclosing(call, parents, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
    if assignment is not None and any(x is call for x in ast.walk(assignment.value)):
        targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
        names = set().union(*(assigned_names(x) for x in targets))
        return names, "assigned_response"
    with_node = enclosing(call, parents, (ast.With, ast.AsyncWith))
    if with_node is not None:
        for item in with_node.items:
            if item.optional_vars is not None and any(x is call for x in ast.walk(item.context_expr)):
                return assigned_names(item.optional_vars), "context_response"
    return set(), "unassigned_call"


def assigned_after(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    return [p for p in lexical_body_nodes(node)
            if isinstance(p, (ast.Assign, ast.AnnAssign, ast.NamedExpr))]


def names_written(part: ast.AST) -> set[str]:
    targets = part.targets if isinstance(part, ast.Assign) else [part.target]
    return set().union(*(assigned_names(x) for x in targets))


VALUE_GETTERS = {"json", "model_dump", "dict", "read", "text"}
VALUE_CONVERSIONS = {"len", "list", "tuple", "str", "int", "float", "bool"}


def is_projection(value: ast.AST, known: set[str]) -> bool:
    """A narrow value projection; passing a response to another request abstains."""
    if isinstance(value, ast.Name):
        return value.id in known
    if isinstance(value, ast.Attribute):
        return is_projection(value.value, known)
    if isinstance(value, ast.Subscript):
        return is_projection(value.value, known)
    if isinstance(value, ast.Await):
        return is_projection(value.value, known)
    if isinstance(value, ast.Call) and not value.keywords:
        if isinstance(value.func, ast.Attribute) and value.func.attr in VALUE_GETTERS:
            return is_projection(value.func.value, known)
        if (isinstance(value.func, ast.Name) and value.func.id in VALUE_CONVERSIONS
                and len(value.args) == 1):
            return is_projection(value.args[0], known)
    return False


def alias_names(original: set[str], assignments: list[ast.AST],
                begin_line: int, end_line: int) -> set[str]:
    known = set(original)
    for part in sorted(assignments, key=lambda x: x.lineno):
        if not begin_line < part.lineno < end_line:
            continue
        names = names_written(part)
        value = getattr(part, "value", None)
        if value is None:
            continue
        if is_projection(value, known):
            known.update(names)
        else:
            known.difference_update(names)
    return known


def relation_for_call(call: ast.Call, node: ast.FunctionDef | ast.AsyncFunctionDef,
                      parents: dict[ast.AST, ast.AST], checks: list[tuple[ast.AST, str]]) -> list[dict]:
    names, form = main_value_assign(call, parents)
    assignments = assigned_after(node)
    links = []
    with_node = enclosing(call, parents, (ast.With, ast.AsyncWith))
    if with_node is not None:
        for item in with_node.items:
            context = item.context_expr
            if (isinstance(context, ast.Call)
                    and call_name(context).split(".")[-1] in {"raises", "assertRaises", "assertRaisesRegex"}):
                links.append({"oracle_line": with_node.lineno,
                              "oracle_expression": ast.unparse(context)[:240],
                              "relation": "expected_exception", "via": ""})
                break
    for check, check_form in checks:
        if check.lineno < call.lineno:
            continue
        cursor = call
        while cursor in parents and cursor is not check:
            cursor = parents[cursor]
        if cursor is check:
            links.append({"oracle_line": check.lineno, "oracle_expression": ast.unparse(check)[:240],
                          "relation": "inline_assertion", "via": ""})
            continue
        if not names or check.lineno <= call.lineno:
            continue
        referenced = {x.id for x in ast.walk(check) if isinstance(x, ast.Name)}
        propagated = alias_names(names, assignments, call.lineno, check.lineno)
        used = sorted(propagated & referenced)
        if not used:
            continue
        # A response ID can feed a second request, then appear beside that
        # request's result in an assertion.  This is a cross-request check,
        # not evidence that the first response alone satisfied the oracle.
        if any(
            call.lineno < part.lineno < check.lineno
            and isinstance(getattr(part, "value", None), ast.Call)
            and not is_projection(part.value, propagated)
            and names_written(part) & referenced
            and {x.id for x in ast.walk(part.value) if isinstance(x, ast.Name)} & propagated
            for part in assignments
        ):
            continue
        # No other assignment to the original response name may intervene.
        if any(call.lineno < p.lineno <= check.lineno and names_written(p) & names
               for p in assignments if p.lineno != call.lineno):
            continue
        relation = "response_value" if names & referenced else "derived_response_value"
        links.append({"oracle_line": check.lineno, "oracle_expression": ast.unparse(check)[:240],
                      "relation": relation, "via": ",".join(used), "response_form": form})
    return links


def nested_client_calls(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict]:
    lexical = set(lexical_body_nodes(node))
    found = []
    for part in ast.walk(node):
        if not isinstance(part, ast.Call) or part in lexical:
            continue
        name = call_name(part)
        if (name.startswith(("requests.", "httpx.", "client.", "self.client."))
                and name.split(".")[-1] in {"get", "post", "put", "patch", "delete", "create", "stream"}):
            found.append({"line": part.lineno, "call": name})
    return sorted(found, key=lambda x: x["line"])


def analyze_test(node: ast.FunctionDef | ast.AsyncFunctionDef, wanted: str) -> tuple[list[dict], list[dict]]:
    parents = lexical_parents(node)
    checks = check_sites(node)
    calls = []
    links = []
    for part in lexical_body_nodes(node):
        if not isinstance(part, ast.Call):
            continue
        actual_name = call_name(part)
        if actual_name != wanted and not actual_name.endswith("." + wanted):
            continue
        route_kind, route = route_label(part)
        call_record = {"line": part.lineno, "method": actual_name,
                       "match": "exact" if actual_name == wanted else "suffix_candidate",
                       "route_kind": route_kind, "route": route}
        calls.append(call_record)
        for link in relation_for_call(part, node, parents, checks):
            links.append({"request_line": part.lineno, "method": actual_name,
                          "route_kind": route_kind, "route": route, **link})
    return calls, links


def main() -> Counter[str]:
    source = [r for r in read("master-review-index.tsv")
              if r["source_route_class"] == "HTTP/SDK 客户端调用"
              and r["agent_review_status"] == "not_reviewed"]
    files: dict[tuple[str, str], dict[int, ast.AST]] = {}
    output = []
    for row in source:
        key = row["snapshot"], row["path"]
        if key not in files:
            text = (REPOS / REPO_NAMES[key[0]] / key[1]).read_text(errors="replace")
            files[key] = python_nodes(text)
        node = files[key].get(int(row["line"]))
        wanted = row["source_route_evidence"].split(";", 1)[0].removeprefix("call=")
        calls, links = analyze_test(node, wanted) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else ([], [])
        nested = nested_client_calls(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
        if not calls:
            status, next_check = "route_call_not_relocated", "核对源码入口及路由识别所用调用"
        elif links:
            status, next_check = "request_to_oracle_source_candidate", "核对请求条件、响应字段或预期异常；按实际运行实例确认"
        else:
            status, next_check = "request_call_without_response_value_link", "追踪客户端回调、状态捕获或辅助断言；必要时运行追踪"
        output.append({"snapshot": row["snapshot"], "path": row["path"],
                       "test_name": row["test_name"], "line": row["line"],
                       "client_call_name": wanted,
                       "route_calls_json": json.dumps(calls, ensure_ascii=False),
                       "nested_client_calls_json": json.dumps(nested, ensure_ascii=False),
                       "request_oracle_links_json": json.dumps(links, ensure_ascii=False),
                       "link_status": status, "candidate_object": row["object_candidates"],
                       "source_route_evidence": row["source_route_evidence"],
                       "review_next": next_check, "verified_property": ""})
    assert len(output) == len(source)
    assert len({(x["snapshot"],x["path"],x["test_name"],x["line"]) for x in output}) == len(source)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader(); writer.writerows(output)
    counts = Counter(x["link_status"] for x in output)
    exact_routes = sum(any(c["route_kind"] == "literal_endpoint" for c in json.loads(x["route_calls_json"])) for x in output)
    multi_calls = sum(len(json.loads(x["route_calls_json"])) > 1 for x in output)
    nested_calls = sum(bool(json.loads(x["nested_client_calls_json"])) for x in output)
    simple_links = sum(len(json.loads(x["route_calls_json"])) == 1
                       and len(json.loads(x["request_oracle_links_json"])) == 1 for x in output)
    lines = [
        "# HTTP／SDK 客户端入口的请求与断言线索",
        "",
        f"主索引中 {len(output):,} 条尚未源码审阅的 HTTP／SDK 入口已逐项处理。脚本按原路由识别记录的客户端调用名，回到固定测试源码定位调用、直接端点参数、响应变量与显式断言；{exact_routes:,} 条至少包含一个字面量端点。输出只表示静态值流候选，不表示请求实际成功、断言一定执行或某个系统性质已获验证。",
        "",
        "| 状态 | 入口数 |",
        "| --- | ---: |",
    ]
    for status, count in counts.most_common():
        lines.append(f"| `{status}` | {count:,} |")
    lines += [
        "",
        "从同一测试体的调用和断言中建立关系时，响应变量重写、数据经辅助函数变换、请求通过 mock 客户端、一个测试发多次请求，都会影响可解释范围。脚本对直接变量重写中止旧链接；对状态捕获及无法从响应值追到断言的入口保持待核查。HTTP URL 与 SDK 方法只标记请求目标，不把它当作验证性质。",
        "",
        f"审查队列可先处理 {simple_links:,} 条单一调用、单一断言候选，再看 {multi_calls:,} 条包含多个匹配客户端调用的入口和 {nested_calls:,} 条含嵌套函数客户端调用的入口。无响应值直达断言的入口优先按流式累积、状态副作用、辅助函数和请求准备操作分组。多个计数可重叠。",
        "",
        "独立源码核查见 `agent_http_holdout.tsv` 与 `agent_http_holdout.md`；抽样只评估静态关联是否有源码依据，不评估运行覆盖或验证性质准确率。",
        "",
        "重建：`experiments/.venv/bin/python experiments/semantic_space/agent_http_boundary.py`。",
        "",
    ]
    REPORT.write_text("\n".join(lines))
    return counts


if __name__ == "__main__":
    for status, count in main().most_common():
        print(status, count)
