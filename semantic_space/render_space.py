"""Render the trace-grounded verification-unit graph without inferred distances."""

from __future__ import annotations

import html
import csv
import json
import shutil
import subprocess
import unicodedata
from pathlib import Path

from validate_space import validate


HERE = Path(__file__).resolve().parent
PHASES = [
    ("ingress", "请求进入"),
    ("scheduling", "调度"),
    ("cache", "缓存"),
    ("transfer", "远程传输"),
    ("model_execution", "模型计算"),
    ("sampling", "采样"),
    ("output", "结果返回"),
    ("service", "服务运行"),
    ("abort", "取消"),
]
PHASE_X = {phase: 85 + i * 140 for i, (phase, _) in enumerate(PHASES)}
DISPLAY_NAMES = {
    "shared_resource.lifetime": "共享资源生命周期",
    "shared_resource.buffer_slot_integrity": "共享槽位隔离",
    "shared_resource.object_handle_identity": "共享句柄绑定",
    "distributed.message_delivery": "消息交付保持",
    "distributed.queue_wait_shutdown": "等待与关闭",
    "distributed.resource_fallback_agreement": "资源回退一致",
    "deployment.runtime_mode_binding": "进程模式绑定",
    "deployment.effective_configuration": "配置解析结果",
    "deployment.configuration_admissibility": "部署配置约束",
    "request.access_control": "请求鉴权",
    "model.parameter_binding": "模型参数绑定",
    "adapter.residency_policy": "适配器驻留与淘汰",
    "model.execution_variant_relation": "执行模式结果对照",
    "response.valid": "有效响应",
    "response.logprob_identity": "概率 token 对应",
    "response.logprob_value": "概率数值对照",
    "response.logprob_text_decoding": "概率 token 文本",
    "embedding.valid": "有效向量响应",
    "spec.target_semantics": "推测解码目标行为",
    "distributed.logical_execution": "分布式逻辑执行",
    "disagg.logical_continuation": "分离式请求延续",
    "input.multimodal_binding": "多模态输入对应",
    "input.text_tokenization": "文本编码对应",
    "input.chat_template_tokenization": "聊天模板编码",
    "input.error_response_classification": "输入错误分类",
    "input.context_length_admissibility": "请求长度准入",
    "input.modality_count_admissibility": "媒体数量准入",
    "input.logprob_token_id_admissibility": "概率 ID 准入",
    "input.video_frame_selection": "视频帧选择",
    "request.prefill_token_view": "请求 token 对应",
    "request.admission": "输入绑定",
    "schedule.progress": "调度进度",
    "schedule.segment_continuity": "分段进度连续",
    "schedule.work_distribution": "工作分配",
    "cache.safe_reuse": "缓存安全复用",
    "cache.namespace": "缓存键隔离",
    "cache.hash_order_sensitivity": "token 顺序哈希",
    "cache.hash_bigram_equivalence": "bigram 哈希对应",
    "cache.hit_accounting": "命中数量",
    "cache.lifetime": "缓存生命周期",
    "cache.eviction_outcome": "淘汰结果与保护",
    "cache.state_invariants": "资源计数一致",
    "cache.tier_transfer_content": "跨层数据保持",
    "cache.result_relation": "复用结果对照",
    "model.reference_relation": "模型计算对照",
    "compute.operator_reference_relation": "算子结果对照",
    "model.adapter_binding": "适配器绑定",
    "distributed.partition_binding": "分片对应",
    "distributed.reduce_relation": "跨卡归约",
    "disagg.request_binding": "跨节点请求绑定",
    "disagg.kv_mapping": "KV 分片对应",
    "disagg.transfer_gate": "传输就绪门禁",
    "embedding.reference_relation": "向量数值对照",
    "sampling.policy": "采样参数",
    "sampling.per_request_modifier_binding": "修饰参数绑定",
    "sampling.temperature_finiteness": "非有限温度拒绝",
    "sampling.top_p_range": "top_p 越界拒绝",
    "sampling.numeric_parameter_bounds": "数值参数越界拒绝",
    "sampling.zero_temperature_mode": "零温度贪心",
    "sampling.structured_constraint": "格式约束",
    "spec.acceptance": "草稿接受与恢复",
    "output.finalization": "输出与停止",
    "output.token_buffer_search": "TokenBuffer 搜索",
    "output.marker_sequence_search": "标记序列搜索",
    "tool.call_parse_relation": "工具调用解析",
    "response.delivery": "接口结果交付",
    "response.stream_event_lifecycle": "流式事件开闭",
    "response.stream_payload_consistency": "流式内容对应",
    "response.usage_accounting": "token 用量对应",
    "embedding.delivery": "向量结果交付",
    "spec.output_relation": "最终输出关系",
    "service.performance": "吞吐与延迟",
    "service.concurrent_request_isolation": "并发请求隔离",
    "quality.task_score": "任务质量得分",
    "quality.sample_semantic_expectation": "单样本语义预期",
    "abort.safe": "安全取消",
    "abort.no_late_output": "迟到输出",
    "abort.cleanup": "状态清理",
    "abort.inflight_release": "在途资源释放",
}
TRACE_NAMES = {
    "text_generation": "文本生成",
    "streamed_response": "流式响应",
    "segmented_prefill": "分段 Prefill",
    "prefix_reuse": "前缀复用",
    "cancel_running_request": "运行中取消",
    "structured_json_response": "约束生成",
    "image_request": "图像输入",
    "adapter_switch": "适配器切换",
    "service_load": "服务负载",
    "task_quality_eval": "任务质量",
    "embedding_request": "向量请求",
    "speculative_greedy": "推测解码",
    "tensor_parallel_generation": "张量并行生成",
    "disaggregated_generation": "分离式生成",
    "video_request": "视频请求",
}


def e(value: object) -> str:
    return html.escape(str(value), quote=True)


def label_lines(label: str, max_width: float = 116) -> list[str]:
    """Fit a short SVG label inside a verification-unit box."""
    lines: list[str] = []
    current = ""
    width = 0.0
    for char in label:
        char_width = 13 if unicodedata.east_asian_width(char) in "FW" else 7 if char != " " else 4
        if current and width + char_width > max_width:
            lines.append(current.rstrip())
            current, width = "", 0.0
            if char == " ":
                continue
        current += char
        width += char_width
    if current.strip():
        lines.append(current.strip())
    return lines or [label]


def render_svg(data: dict) -> str:
    units = data["units"]
    by_id = {unit["id"]: unit for unit in units}
    group_counts: dict[str, int] = {}
    positions: dict[str, tuple[float, float]] = {}
    for unit in units:
        if unit["kind"] == "composite":
            continue
        phase = unit["phase"]
        index = group_counts.get(phase, 0)
        group_counts[phase] = index + 1
        positions[unit["id"]] = (85, 145 + index * 56) if phase == "preflight" else (PHASE_X[phase], 350 + index * 58)
    positions["response.valid"] = (560, 130)
    positions["embedding.valid"] = (860, 130)
    positions["spec.target_semantics"] = (PHASE_X["sampling"], 220)
    positions["distributed.logical_execution"] = (PHASE_X["model_execution"], 220)
    positions["disagg.logical_continuation"] = (PHASE_X["transfer"], 220)
    positions["cache.safe_reuse"] = (PHASE_X["cache"], 220)
    positions["abort.safe"] = (PHASE_X["abort"], 130)
    lowest_unit = max(y for x, y in positions.values() if y >= 350)
    guide_bottom = max(612, lowest_unit + 26)
    legend_y = max(670, lowest_unit + 76)
    svg_height = legend_y + 28

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 {svg_height}" role="img" aria-label="推理系统的验证单元、层次和依赖">',
        '<style>text{font-family:Arial,"PingFang SC","Noto Sans CJK SC",sans-serif}.heading{font-size:23px;font-weight:700;fill:#222}.sub{font-size:12px;fill:#62616a}.phase{font-size:13px;font-weight:700;fill:#34313a}.node-label{font-size:14px;font-weight:600;fill:#27222d}.composite-label{font-size:14px;font-weight:700;fill:#632b77}.legend{font-size:12px;fill:#4b4752}.edge{fill:none;stroke-width:1.5}.unit{cursor:pointer}</style>',
        f'<rect width="1280" height="{svg_height}" fill="#fff"/>',
        '<text x="32" y="38" class="heading">推理系统的验证单元</text>',
        '<text x="32" y="60" class="sub">横向为请求流程；部署条件位于请求进入前。</text>',
        '<text x="32" y="110" class="phase">部署条件与组合性质</text>',
        '<line x1="32" y1="290" x2="1245" y2="290" stroke="#ddd9e0"/>',
    ]
    for phase, label in PHASES:
        x = PHASE_X[phase]
        parts.append(f'<text x="{x}" y="317" text-anchor="middle" class="phase phase-heading" data-phase="{phase}">{e(label)}</text>')
        parts.append(f'<line x1="{x}" y1="328" x2="{x}" y2="{guide_bottom}" stroke="#eeeaf0"/>')

    for relation in data["relations"]:
        x1, y1 = positions[relation["parent"]]
        x2, y2 = positions[relation["child"]]
        kind = relation["type"]
        color = "#8c6b9a" if kind == "refines" else "#70988c" if kind == "requires" else "#a1a0a5"
        dash = "" if kind == "refines" else ' stroke-dasharray="4 4"'
        if y2 > y1 + 45:
            start_y, end_y = y1 + 22, y2 - 21
            bend = (start_y + end_y) / 2
            path = f"M {x1:.1f} {start_y:.1f} C {x1:.1f} {bend:.1f}, {x2:.1f} {bend:.1f}, {x2:.1f} {end_y:.1f}"
        else:
            path = f"M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}"
        parts.append(
            f'<path class="edge" data-parent="{e(relation["parent"])}" data-child="{e(relation["child"])}" '
            f'd="{path}" stroke="{color}" stroke-opacity="0.28"{dash}>'
            f'<title>{e(relation["type"] + ": " + relation["when"])}</title></path>'
        )

    for unit in units:
        x, y = positions[unit["id"]]
        composite = unit["kind"] == "composite"
        scope_colors = {
            "request": ("#f3edf5", "#bca6c7"),
            "shared": ("#e9f2ee", "#8bb6a6"),
            "local": ("#f0f0f2", "#a9a9ae"),
            "deployment": ("#f0f0f2", "#a9a9ae"),
        }
        fill, stroke = ("#fff", "#75348b") if composite else scope_colors[unit["state_scope"]]
        label_class = "composite-label" if composite else "node-label"
        level = "goal" if y == 130 else "unit"
        parts.append(f'<g class="unit" data-unit="{e(unit["id"])}" data-phase="{e(unit.get("phase", ""))}" data-level="{level}" tabindex="0">')
        parts.append(f'<rect x="{x-68:.1f}" y="{y-22:.1f}" width="136" height="44" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{2 if composite else 1.2}"/>')
        label = DISPLAY_NAMES.get(unit["id"], unit["name"])
        lines = label_lines(label)
        font_size = 12 if len(lines) > 2 else 13 if len(lines) > 1 else 14
        line_height = 13 if len(lines) > 2 else 16
        first_y = y - (len(lines) - 1) * line_height / 2 + font_size * 0.34
        parts.append(f'<text x="{x:.1f}" y="{first_y:.1f}" text-anchor="middle" class="{label_class}" style="font-size:{font_size}px">')
        for index, line in enumerate(lines):
            parts.append(f'<tspan x="{x:.1f}" dy="{0 if index == 0 else line_height}">{e(line)}</tspan>')
        parts.append('</text>')
        parts.append(f'<title>{e(unit["property"])}</title></g>')

    parts.extend(
        [
            f'<line x1="33" y1="{legend_y}" x2="67" y2="{legend_y}" stroke="#8c6b9a" stroke-width="2"/><text x="74" y="{legend_y+4}" class="legend">分解</text>',
            f'<line x1="157" y1="{legend_y}" x2="191" y2="{legend_y}" stroke="#70988c" stroke-width="2" stroke-dasharray="4 4"/><text x="198" y="{legend_y+4}" class="legend">前提依赖</text>',
            f'<rect x="302" y="{legend_y-12}" width="30" height="22" rx="5" fill="#f3edf5" stroke="#bca6c7"/><text x="339" y="{legend_y+4}" class="legend">单请求状态</text>',
            f'<rect x="482" y="{legend_y-12}" width="30" height="22" rx="5" fill="#e9f2ee" stroke="#8bb6a6"/><text x="519" y="{legend_y+4}" class="legend">跨请求／进程状态</text>',
            f'<rect x="662" y="{legend_y-12}" width="30" height="22" rx="5" fill="#f0f0f2" stroke="#a9a9ae"/><text x="699" y="{legend_y+4}" class="legend">局部计算／部署条件</text>',
            '</svg>',
        ]
    )
    return "\n".join(parts)


def render_html(data: dict, svg: str) -> str:
    model = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    catalog_path = HERE / "test_catalog.json"
    catalog = catalog_path.read_text(encoding="utf-8").replace("</", "<\\/") if catalog_path.exists() else "null"
    runtime_path = HERE / "runtime_index.json"
    runtime = runtime_path.read_text(encoding="utf-8").replace("</", "<\\/") if runtime_path.exists() else '{"entries":[]}'
    inherited_path = HERE / "inherited-test-audit.tsv"
    with inherited_path.open(encoding="utf-8", newline="") as source:
        inherited_rows = [row for row in csv.DictReader(source, delimiter="\t") if row["method"]]
    inherited = json.dumps(inherited_rows, ensure_ascii=False).replace("</", "<\\/")
    helper_path = HERE / "helper-chain-audit.tsv"
    with helper_path.open(encoding="utf-8", newline="") as source:
        helper_rows = [
            {
                "snapshot": row["snapshot"], "test_path": row["test_path"],
                "test_name": row["test_name"], "test_line": int(row["test_line"]),
                "status": row["status"], "truncated": row["truncated"] == "1",
                "oracle_sites": json.loads(row["oracle_sites"]),
                "unresolved_calls": row["unresolved_calls"],
            }
            for row in csv.DictReader(source, delimiter="\t")
        ]
    helper_catalog = json.dumps(helper_rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with (HERE / "call-oracle-link-audit.tsv").open(encoding="utf-8", newline="") as source:
        call_oracle_rows = [
            {
                "snapshot": row["snapshot"], "path": row["path"],
                "test_name": row["test_name"], "test_line": int(row["test_line"]),
                "product_call_line": int(row["product_call_line"]),
                "product_target": row["product_target"],
                "link_kind": row["link_kind"], "oracle_line": int(row["oracle_line"]),
                "oracle_expression": row["oracle_expression"],
                "via_variable": row["via_variable"],
            }
            for row in csv.DictReader(source, delimiter="\t")
            if row["status"] == "static_oracle_link_candidate"
        ]
    call_oracle_catalog = json.dumps(call_oracle_rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    with (HERE / "python-collectability-audit.tsv").open(encoding="utf-8", newline="") as source:
        collection_exceptions = [
            {"snapshot": row["snapshot"], "path": row["path"],
             "test_name": row["test_name"], "line": int(row["line"]),
             "status": row["status"], "source_role": row["source_role"]}
            for row in csv.DictReader(source, delimiter="\t")
            if row["source_role"] != "test_candidate"
        ]
    collectability = json.dumps(collection_exceptions, ensure_ascii=False,
                                separators=(",", ":")).replace("</", "<\\/")
    with (HERE / "dynamic-test-method-audit.tsv").open(encoding="utf-8", newline="") as source:
        dynamic_rows = [dict(row) for row in csv.DictReader(source, delimiter="\t")]
    dynamic_catalog = json.dumps(dynamic_rows, ensure_ascii=False,
                                 separators=(",", ":")).replace("</", "<\\/")
    with (HERE / "nonstandard-class-audit.tsv").open(encoding="utf-8", newline="") as source:
        nonstandard_rows = [dict(row) for row in csv.DictReader(source, delimiter="\t")]
    nonstandard_catalog = json.dumps(nonstandard_rows, ensure_ascii=False,
                                     separators=(",", ":")).replace("</", "<\\/")
    source_review = (HERE / "classification-source-reviewed-sample.json").read_text(
        encoding="utf-8"
    ).replace("</", "<\\/")
    space_assessment = (HERE / "reviewed-space-assessment.json").read_text(
        encoding="utf-8"
    ).replace("</", "<\\/")
    batch_review_rows = []
    for filename in json.loads((HERE / "source-review-batches.json").read_text(encoding="utf-8")):
        with (HERE / filename).open(encoding="utf-8", newline="") as source:
            batch_review_rows.extend(dict(row) for row in csv.DictReader(source, delimiter="\t")
                                     if row["status"] != "pending_source_review")
    batch_source_review = json.dumps(batch_review_rows, ensure_ascii=False,
                                     separators=(",", ":")).replace("</", "<\\/")
    explorer_script = (HERE / "explorer.js").read_text(encoding="utf-8")
    filter_buttons = '<button type="button" data-trace="all" aria-pressed="true">全部单元</button>'
    filter_buttons += "".join(
        f'<button type="button" data-trace="{e(trace["id"])}" aria-pressed="false">{e(TRACE_NAMES.get(trace["id"], trace["id"]))}</button>'
        for trace in data["traces"]
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Verification Space：源码对照</title>
<style>
body{{margin:0;background:#fbfafb;color:#26212a;font-family:Arial,'PingFang SC','Noto Sans CJK SC',sans-serif}}
.workspace{{display:grid;grid-template-columns:290px minmax(0,1fr);min-height:100dvh}}
.explorer{{position:sticky;top:0;height:100dvh;display:flex;flex-direction:column;box-sizing:border-box;min-width:0;padding:12px 8px;border-right:1px solid #e1dce3;background:#f6f4f7}}
.explorer h2{{margin:2px 8px 10px;font-size:16px}}
.explorer input{{box-sizing:border-box;width:100%;margin-bottom:8px;padding:8px;border:1px solid #cfc6d4;border-radius:6px;background:#fff;font:inherit;font-size:12px}}
.tree{{flex:1;min-height:0;overflow:auto;font-size:12px}}
.tree summary{{cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:4px 5px;border-radius:4px}}
.tree summary:hover,.test-entry:hover,.file-entry:hover{{background:#ebe4ef}}
.tree details details{{margin-left:13px}}
.tree .test-entry,.tree .file-entry{{box-sizing:border-box;display:block;width:100%;border:0;background:transparent;text-align:left;padding:4px 5px 4px 18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;cursor:pointer;font-size:12px}}
.tree .test-entry[aria-selected=true],.tree .file-entry[aria-selected=true],.tree summary[aria-selected=true]{{background:#e2d2e8;color:#532069}}
.tree .support{{color:#837d86}}
.tree .search-count{{padding:5px 8px;color:#625c65}}
main{{min-width:0;max-width:1480px;width:100%;box-sizing:border-box;margin:8px auto;padding:0 12px}}
.toolbar{{display:flex;gap:6px;align-items:center;margin-bottom:6px;overflow-x:auto;white-space:nowrap;scrollbar-width:thin}}
button{{border:1px solid #cfc6d4;border-radius:7px;padding:7px 12px;background:white;color:#34303a;cursor:pointer;font:inherit;font-size:13px}}
.toolbar button{{flex:none}}
button[aria-pressed=true]{{background:#75348b;color:white;border-color:#75348b}}
.layout{{display:grid;grid-template-columns:minmax(0,1fr);gap:14px}}
.plot,.details{{background:white;border:1px solid #e1dce3;border-radius:8px}}
.plot{{display:flex;justify-content:center}}
.plot svg{{width:100%;height:auto;display:block}}
.details{{padding:16px;line-height:1.5}}
.details h2{{font-size:17px;margin:0 0 9px}}
.details h3{{font-size:13px;margin:13px 0 4px}}
.details p,.details li{{font-size:13px;margin:5px 0;overflow-wrap:anywhere}}
.details ul{{padding-left:18px;margin:4px 0}}
.details a{{color:#653078;word-break:break-word}}
.match-result{{display:none}}
.match-result.visible{{display:block}}
.match-result .route-pill{{display:inline-block;margin:2px 5px 2px 0;padding:3px 7px;border-radius:5px;background:#f0e7f3;color:#5b216e;font-size:12px}}
.aux-scopes{{display:none;gap:6px;align-items:center;flex-wrap:wrap;margin:0 0 6px;font-size:12px}}
.aux-scopes.visible{{display:flex}}
.aux-scopes span{{color:#625c65;margin-right:4px}}
.aux-scopes .aux-scope{{padding:4px 8px;border:1px solid #d8cedd;border-radius:5px;background:#fff}}
.aux-scopes .aux-scope.active{{background:#ead9f0;border-color:#75348b;color:#522066;font-weight:600}}
.selection-banner{{display:none;margin:0 0 6px;padding:7px 10px;border-left:3px solid #75348b;background:#f4edf6;font-size:12px}}
.selection-banner.visible{{display:block}}
.phase-heading.candidate{{fill:#5b1b70;text-decoration:underline}}
.unit:hover rect{{stroke:#75348b;stroke-width:1.8;stroke-dasharray:3 2}}
.unit.candidate rect{{stroke:#75348b;stroke-width:2.4;stroke-dasharray:5 3}}
.unit.active rect{{stroke:#302b33;stroke-width:2.8;stroke-dasharray:none}}
.unit.confirmed rect{{fill:#d7ede4;stroke:#27765d;stroke-width:2.8}}
.unit.dimmed{{opacity:.21}}
.unit.dimmed[data-level=goal]{{opacity:.035}}
.unit.dimmed[data-level=goal] rect{{stroke:#d8d2da}}
.edge.dimmed{{opacity:.08}}
@media(max-width:760px){{.workspace{{grid-template-columns:205px minmax(0,1fr)}}.explorer{{padding:8px 4px}}main{{padding:0 5px}}}}
</style></head><body>
<div class="workspace"><aside class="explorer"><h2>测试 Explorer</h2><input id="test-search" type="search" placeholder="搜索文件或测试函数" aria-label="搜索测试"><div class="tree" id="test-tree"></div><div class="tree" id="test-search-results" hidden></div></aside><main>
<div class="toolbar" id="trace-filters">{filter_buttons}</div>
<div class="selection-banner" id="selection-banner"></div><div class="aux-scopes" id="aux-scopes"><span>其他验证对象</span><div class="aux-scope" data-area="configuration">配置与部署</div><div class="aux-scope" data-area="security">认证与访问控制</div><div class="aux-scope" data-area="diagnostics">调试与测试基础设施</div><div class="aux-scope" data-area="runtime_support">通用运行组件</div><div class="aux-scope" data-area="data_structure">核心数据结构</div><div class="aux-scope" data-area="image_generation">图像生成系统</div><div class="aux-scope" data-area="training">训练与 RL</div></div><div class="layout"><div class="plot" id="graph-drop-zone">{svg}</div><section class="details match-result" id="match-result"></section><section class="details" id="details"><h2>选择一个验证单元</h2><p>查看它的性质、适用条件，以及两套框架中的源码与测试证据。</p></section></div>
</main></div><script type="application/json" id="space-data">{model}</script><script type="application/json" id="test-catalog">{catalog}</script><script type="application/json" id="runtime-index">{runtime}</script><script type="application/json" id="inherited-catalog">{inherited}</script><script type="application/json" id="helper-chain-catalog">{helper_catalog}</script><script type="application/json" id="call-oracle-catalog">{call_oracle_catalog}</script><script type="application/json" id="collectability-catalog">{collectability}</script><script type="application/json" id="dynamic-catalog">{dynamic_catalog}</script><script type="application/json" id="nonstandard-catalog">{nonstandard_catalog}</script><script type="application/json" id="source-review-catalog">{source_review}</script><script type="application/json" id="space-assessment-catalog">{space_assessment}</script><script type="application/json" id="batch-source-review-catalog">{batch_source_review}</script>
<script>
(() => {{
  const data=JSON.parse(document.getElementById('space-data').textContent);
  const units=new Map(data.units.map(u=>[u.id,u]));
  const details=document.getElementById('details');
  const roots=document.querySelectorAll('.unit');
  const edges=document.querySelectorAll('.edge');
  const plot=document.querySelector('.plot');
  const graph=plot.querySelector('svg');
  function fitGraph(){{
    const availableHeight=Math.max(300,window.innerHeight-plot.getBoundingClientRect().top-12);
    const box=graph.viewBox.baseVal;
    const width=Math.min(plot.clientWidth-2,availableHeight*box.width/box.height);
    graph.style.width=Math.max(0,width)+'px';
  }}
  window.addEventListener('resize',fitGraph);
  fitGraph();
  function escapeHtml(s){{return String(s).replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#39;'}}[c]));}}
  function clearExplorerSelection(){{
    document.getElementById('selection-banner').classList.remove('visible');
    document.getElementById('match-result').classList.remove('visible');
    document.getElementById('aux-scopes').classList.remove('visible');
    document.querySelectorAll('.aux-scope.active').forEach(n=>n.classList.remove('active'));
    document.querySelectorAll('.phase-heading.candidate').forEach(n=>n.classList.remove('candidate'));
    document.querySelectorAll('.unit.confirmed').forEach(n=>n.classList.remove('confirmed'));
    document.querySelectorAll('.unit.candidate').forEach(n=>n.classList.remove('candidate'));
    document.querySelectorAll('[aria-selected=true]').forEach(n=>n.setAttribute('aria-selected','false'));
    window.dispatchEvent(new Event('resize'));
  }}
  function directRefinements(id){{
    const active=new Set([id]);
    data.relations.forEach(r=>{{if(r.type==='refines'&&r.parent===id)active.add(r.child);}});
    return active;
  }}
  function paintFocus(active, selectedId=null){{
    roots.forEach(n=>{{
      n.classList.toggle('active',n.dataset.unit===selectedId);
      n.classList.toggle('dimmed',!active.has(n.dataset.unit));
    }});
    edges.forEach(n=>{{
      n.classList.remove('active');
      n.classList.toggle('dimmed',!active.has(n.dataset.parent)||!active.has(n.dataset.child));
    }});
  }}
  function show(id){{
    const u=units.get(id); if(!u)return;
    clearExplorerSelection();
    document.querySelectorAll('#trace-filters button').forEach(button=>
      button.setAttribute('aria-pressed','false'));
    paintFocus(directRefinements(id),id);
    const evidence=Object.entries(u.evidence).map(([framework,refs])=>'<h3>'+escapeHtml(framework)+'</h3><ul>'+refs.map(r=>{{
      const snap=data.snapshots[framework];
      const repo=framework==='vllm'?'vllm-project/vllm':'sgl-project/sglang';
      const url='https://github.com/'+repo+'/blob/'+snap.commit+'/'+r.path+'#L'+r.line;
      return '<li><a target="_blank" rel="noopener" href="'+url+'">'+escapeHtml(r.path+':'+r.line)+'</a> ('+escapeHtml(r.kind)+')</li>';
    }}).join('')+'</ul>').join('');
    details.innerHTML='<h2>'+escapeHtml(u.name)+'</h2><p>'+escapeHtml(u.property)+'</p><h3>条件</h3><p>'+escapeHtml(u.conditions.join('、'))+'</p><h3>观察与判定</h3><p>'+escapeHtml(u.observable.join('、'))+'。'+escapeHtml(u.oracle)+'</p>'+(evidence||'<p>组合性质；各组成义务见连线。</p>');
  }}
  roots.forEach(n=>{{n.addEventListener('click',()=>show(n.dataset.unit));n.addEventListener('keydown',e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();show(n.dataset.unit);}}}});}});
  document.querySelectorAll('#trace-filters button').forEach(btn=>btn.addEventListener('click',()=>{{
    clearExplorerSelection();
    document.querySelectorAll('#trace-filters button').forEach(b=>b.setAttribute('aria-pressed',String(b===btn)));
    const id=btn.dataset.trace;
    const trace=data.traces.find(t=>t.id===id);
    paintFocus(trace?new Set(trace.steps):new Set(data.units.map(u=>u.id)));
  }}));
}})();
</script><script>{explorer_script}</script></body></html>"""


def main() -> None:
    validate()
    data = json.loads((HERE / "space_seed.json").read_text(encoding="utf-8"))
    svg = render_svg(data)
    (HERE / "space_view.svg").write_text(svg, encoding="utf-8")
    (HERE / "space_view.html").write_text(render_html(data, svg), encoding="utf-8")
    if converter := shutil.which("rsvg-convert"):
        with (HERE / "space_view.png").open("wb") as output:
            subprocess.run(
                [converter, "-w", "2560", "-h", "1460"],
                input=svg.encode("utf-8"),
                stdout=output,
                check=True,
            )
    print("Rendered space_view.svg, space_view.html and optional space_view.png")


if __name__ == "__main__":
    main()
