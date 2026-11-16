"""Canonical test-body family index for scalable, source-grounded review.

Call targets, attribute names, assertion operators and control structure are
retained. Local variable names and literal values are normalized. Family
membership is only a review routing signal, never a property label.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from extract_review import HERE, INDEX, SOURCE, assigned, call_name, locate


class Canonical(ast.NodeTransformer):
    def visit_Call(self, node):
        fn = call_name(node.func) or "dynamic_callee"
        return ast.Call(
            func=ast.Name(id="CALL_" + fn.replace(".", "__"), ctx=ast.Load()),
            args=[self.visit(x) for x in node.args],
            keywords=[ast.keyword(arg=x.arg, value=self.visit(x.value)) for x in node.keywords],
        )

    def visit_Name(self, node):
        return ast.Name(id="VAR", ctx=type(node.ctx)())

    def visit_Constant(self, node):
        value = node.value
        kind = "None" if value is None else type(value).__name__
        return ast.Constant(value=kind)

    def visit_JoinedStr(self, node):
        return ast.Constant(value="fstr")

    def visit_FunctionDef(self, node):
        return ast.FunctionDef(name="fn", args=ast.arguments(posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]), body=self.clean_body(node.body), decorator_list=[], returns=None, type_comment=None)

    visit_AsyncFunctionDef = visit_FunctionDef

    def clean_body(self, body):
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        return [self.visit(x) for x in body]


def main():
    rows = [x for x in csv.DictReader(INDEX.open(), delimiter="\t") if assigned(x)]
    grouped = defaultdict(list)
    for row in rows: grouped[row["path"]].append(row)
    families = defaultdict(list)
    errors=[]
    for path, members in sorted(grouped.items()):
        file = SOURCE / path
        if not file.is_file():
            errors.extend([("source_missing", row) for row in members]);continue
        try: tree=ast.parse(file.read_text(errors="replace"))
        except SyntaxError:
            errors.extend([("syntax_error", row) for row in members]);continue
        for row in members:
            node=locate(tree,row)
            if node is None:
                errors.append(("definition_not_located",row));continue
            canonical=Canonical().visit(node)
            sig=ast.dump(canonical, include_attributes=False)
            fid=hashlib.sha256(sig.encode()).hexdigest()[:16]
            first = not families[fid]
            families[fid].append({"key":[row[k] for k in ("snapshot","path","test_name","line")],"canonical_ast":sig if first else None})
    out=[]
    for fid,members in sorted(families.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        out.append({"family_id":fid,"size":len(members),"paths":len({x["key"][1] for x in members}),"representative":members[0]["key"],"members":[x["key"] for x in members],"canonical_ast":members[0]["canonical_ast"]})
    (HERE/"canonical-families.json").write_text(json.dumps(out,ensure_ascii=False,indent=2))
    with (HERE/"canonical-families.tsv").open("w",newline="") as f:
        w=csv.writer(f,delimiter="\t");w.writerow(["family_id","size","files","representative_path","representative_test","representative_line","member_paths"])
        for x in out:w.writerow([x["family_id"],x["size"],x["paths"],*x["representative"][1:],";".join(sorted({m[1] for m in x["members"]}))[:500]])
    (HERE/"canonical-errors.json").write_text(json.dumps([{"reason":r,"key":[x[k] for k in ("snapshot","path","test_name","line")]} for r,x in errors],ensure_ascii=False,indent=2))
    print(json.dumps({"entries":len(rows),"families":len(out),"largest":[(x["size"],x["representative"][1],x["representative"][2]) for x in out[:10]],"errors":len(errors)},ensure_ascii=False))


if __name__=="__main__": main()
