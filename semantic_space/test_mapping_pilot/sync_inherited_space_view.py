"""Place reviewed inherited-method properties in the legacy space explorer.

This is a bounded post-processing step for the generated, self-contained
space_view.html.  It does not infer properties from a method name or change
the source-derived 76-unit model.  Fixed anchors make generator drift fail
instead of silently producing an inconsistent published page.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
VIEW = HERE.parent / "space_view.html"
DATA = HERE / "inherited-property-crosswalk.tsv"
TAG = '<script type="application/json" id="inherited-property-crosswalk">'
NEXT_TAG = '<script type="application/json" id="helper-chain-catalog">'

INSERTIONS = (
    (
        "  const inheritedRows = JSON.parse(document.getElementById('inherited-catalog').textContent);",
        "  const inheritedRows = JSON.parse(document.getElementById('inherited-catalog').textContent);\n"
        "  const inheritedPropertyRows = JSON.parse(document.getElementById('inherited-property-crosswalk').textContent);\n"
        "  const inheritedPropertyByTest = new Map(inheritedPropertyRows.map(row =>\n"
        "    [`${row.snapshot}:${row.path}:${row.test_name}:${row.line}`, row]));",
    ),
    (
        "    const reviewedUnits = review\n"
        "      ? (review.status === 'skipped_by_class' ? [] : review.unit_ids.split(';').filter(Boolean))\n"
        "      : null;",
        "    const checked = inheritedPropertyByTest.get(\n"
        "      `${record.snapshot}:${record.path}:${item.name}:${item.line}`);\n"
        "    const reviewedUnits = review?.status === 'skipped_by_class' ? []\n"
        "      : checked ? checked.unit_id.split(';').filter(Boolean)\n"
        "      : review ? review.unit_ids.split(';').filter(Boolean) : null;",
    ),
    (
        "    const reviewedEvidence = review\n"
        "      ? `<p>已核对的检查：${escapeHtml(review.checked_property)}</p>` +",
        "    const checkedEvidence = checked\n"
        "      ? `<p>公共方法中已核对的局部检查：${escapeHtml(checked.property)}</p>` +\n"
        "        `<p>候选单元：${escapeHtml(checked.unit_id)}；适用条件：${escapeHtml(checked.conditions)}</p>` +\n"
        "        `<p>断言或辅助判定源码：${checked.oracle_source.split(';').map(location => {\n"
        "          const cut = location.lastIndexOf(':');\n"
        "          return sourceLink(record.snapshot, location.slice(0, cut), Number(location.slice(cut + 1)));\n"
        "        }).join('、')}。</p>` +\n"
        "        (review?.status === 'skipped_by_class' ? '<p>具体测试类被显式跳过；这条源码检查未执行。</p>' : '')\n"
        "      : '';\n"
        "    const reviewedEvidence = review && !checked\n"
        "      ? `<p>已核对的检查：${escapeHtml(review.checked_property)}</p>` +",
    ),
    (
        "      reviewedEvidence +\n"
        "      (assertions ? `<p>方法体的断言源码：${assertions}</p>` : helperEvidence(record)) +\n"
        "      '<p>这是继承方法的源码候选；具体 pytest 收集、执行路径和判定关系尚待核对。</p>';",
        "      checkedEvidence + reviewedEvidence +\n"
        "      (assertions ? `<p>方法体的断言源码：${assertions}</p>` : helperEvidence(record)) +\n"
        "      (checked ? '<p>这是源码级局部断言记录；具体 pytest 收集、执行路径和运行判定另行核对。</p>'\n"
        "        : '<p>这是继承方法的源码候选；具体 pytest 收集、执行路径和判定关系尚待核对。</p>');",
    ),
)


def replace_once(body: str, before: str, after: str) -> str:
    if after in body:
        return body
    count = body.count(before)
    if count != 1:
        raise ValueError(f"Expected one explorer anchor, found {count}: {before[:65]}")
    return body.replace(before, after, 1)


def main() -> dict[str, int]:
    with DATA.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    keys = [(r["snapshot"], r["path"], r["test_name"], r["line"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate inherited crosswalk keys")
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    body = VIEW.read_text(encoding="utf-8")
    block = TAG + payload + "</script>\n"
    pattern = re.compile(re.escape(TAG) + r".*?</script>\n", re.S)
    if pattern.search(body):
        body = pattern.sub(lambda _: block, body, count=1)
    else:
        body = replace_once(body, NEXT_TAG, block + NEXT_TAG)
    for before, after in INSERTIONS:
        body = replace_once(body, before, after)
    VIEW.write_text(body, encoding="utf-8")
    result = {"reviewed_inherited": len(rows), "bytes": VIEW.stat().st_size}
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
