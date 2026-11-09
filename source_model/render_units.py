"""Render space_model.json as units.md with GitHub permalinks to pinned sources."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def link(model, site):
    prefix, path, lines = site.split(":")
    snap = model["snapshots"]["vllm" if prefix == "v" else "sglang"]
    a, _, b = lines.partition("-")
    url = (f"https://github.com/{snap['repo']}/blob/{snap['commit']}/"
           f"{snap['repo_prefix']}/{path}#L{a}" + (f"-L{b}" if b else ""))
    return f"[{path}:{lines}]({url})"


def main():
    model = json.loads((HERE / "space_model.json").read_text())
    axes = model["axes"]
    units = model["units"]
    out = ["# 验证单元表", "",
           "由 `render_units.py` 从 `space_model.json` 生成，请勿手工编辑。"
           "源码链接指向固定 commit：vLLM `76a0ccc`、SGLang `49e384c`。", ""]
    out += ["| 层 | 名称 | 需要真实运行的部分 | 单元数 |", "|---|---|---|---|"]
    for b, d in axes["boundary"].items():
        out.append(f"| {b} | {d['name']} | {d['real']} | {sum(u['boundary'] == b for u in units)} |")
    out.append("")
    for b, d in axes["boundary"].items():
        group = [u for u in units if u["boundary"] == b]
        if not group:
            continue
        out += [f"## {b} {d['name']}", ""]
        for u in sorted(group, key=lambda u: (list(axes["stage"]).index(u["stage"]), u["id"])):
            tag = "" if u["status"] == "common" else "（扩展）"
            out += [f"### `{u['id']}`{tag}", "",
                    f"{u['property']}", "",
                    f"- 阶段：{axes['stage'][u['stage']]}；类型：{u['kind']}",
                    f"- 判定：{u['oracle']}"]
            if u["conditions"]:
                out.append(f"- 条件：{', '.join('`' + c + '`' for c in u['conditions'])}")
            if u["perturbation"]:
                out.append(f"- 执行扰动：{model['perturbations'][u['perturbation']]}")
            if u["depends_on"]:
                out.append(f"- 依赖：{', '.join('`' + d + '`' for d in u['depends_on'])}")
            if u["composition_gap"]:
                out.append(f"- 组合缺口：{u['composition_gap']}")
            if u["divergence"]:
                out.append(f"- 框架差异：{u['divergence']}")
            for fw in ("vllm", "sglang"):
                sites = u["sites"].get(fw)
                if sites:
                    name = "vLLM" if fw == "vllm" else "SGLang"
                    out.append(f"- {name}：" + "；".join(link(model, s) for s in sites))
            out.append(f"- 调研记录：{', '.join(u['trace'])}")
            out.append("")
    out += ["## 条件约束", ""]
    for c in model["constraints"]:
        out.append(f"- {c['id']}：{c['rule']}（{link(model, c['site'])}）")
    (HERE / "units.md").write_text("\n".join(out) + "\n")
    print("units.md", len(units))


if __name__ == "__main__":
    main()
