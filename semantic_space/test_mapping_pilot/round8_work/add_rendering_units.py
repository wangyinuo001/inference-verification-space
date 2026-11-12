"""Insert two cross-framework, text-level obligations into the source model."""

import json
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = HERE.parents[1] / "space_seed.json"
ROOTS = {
    "vllm": HERE.parents[4] / "experiments/third_party/reference_repos/vllm-20261105",
    "sglang": HERE.parents[4] / "experiments/third_party/reference_repos/sglang-v0.5.14-complete",
}


def refs(*items):
    return [{"path": p, "line": n, "kind": k} for p, n, k in items]


def main():
    data = json.loads(SEED.read_text(encoding="utf-8"))
    unit_ids = {x["id"] for x in data["units"]}
    assert not ({"input.prompt_text_rendering", "input.media_marker_rendering"} & unit_ids)
    assert "prompt_render_profile" not in {x["id"] for x in data["condition_dimensions"]}
    dimension = {
        "id": "prompt_render_profile", "type": "structured",
        "fields": ["format_and_revision", "message_roles_content_and_order", "tools_and_tool_results",
                   "generation_prompt", "continue_final_message", "media_marker_policy"],
        "observed_values": [],
        "constraint": "Bind the same rendering format and message sequence. Fixed examples include Inkling and DeepSeek-V4 in vLLM, and ADD_COLON_SINGLE, CHATML, LLAMA3, QWEN2_AUDIO and PADDLE_OCR in SGLang. Rendered text, token IDs and media payload binding are separate observations.",
    }
    dpos = next(i for i, x in enumerate(data["condition_dimensions"])
                if x["id"] == "chat_template_profile") + 1
    data["condition_dimensions"].insert(dpos, dimension)
    prompt = {
        "id": "input.prompt_text_rendering", "name": "聊天消息生成的提示词文本符合模板",
        "kind": "obligation", "phase": "ingress", "state_scope": "request",
        "conditions": ["prompt_render_profile"],
        "property": "在指定模板、消息顺序、角色、工具内容和生成续写设置下，渲染的提示词文本保留相应内容，并按模板产生角色边界、分隔符、工具段及续写前缀。",
        "observable": ["messages", "tools", "render_controls", "rendered_prompt_text"],
        "oracle": "固定消息和模板，比较预期提示词文本与渲染结果；子串、前缀或缺席断言只证明其实际检查的局部性质。此处不比较后续编码的 token ID。",
        "execution_boundary": "消息与工具输入进入渲染函数，得到提示词文本；不包括 tokenizer.encode、模型计算或服务响应。",
        "evidence": {
            "vllm": refs(
                ("vllm/tokenizers/deepseek_v4_encoding.py", 548, "source"),
                ("vllm/renderers/inkling_encoding.py", 93, "source"),
                ("tests/tokenizers_/test_deepseek_v4.py", 118, "existing_test"),
                ("tests/tokenizers_/test_deepseek_v4.py", 132, "existing_oracle"),
                ("tests/renderers/test_inkling.py", 196, "existing_test")),
            "sglang": refs(
                ("python/sglang/srt/parser/conversation.py", 107, "source"),
                ("python/sglang/srt/entrypoints/openai/serving_chat.py", 723, "source"),
                ("test/registered/unit/parser/test_conversation.py", 39, "existing_test"),
                ("test/registered/unit/parser/test_conversation.py", 52, "existing_oracle")),
        },
    }
    media = {
        "id": "input.media_marker_rendering", "name": "媒体占位标记在提示词文本中与输入位置对应",
        "kind": "obligation", "phase": "ingress", "state_scope": "request",
        "conditions": ["prompt_render_profile", "modality"],
        "property": "给定图像或音频项目及其相邻文本，渲染文本在对应位置产生声明格式的媒体占位标记，并保留标记与文本片段的相对次序。",
        "observable": ["input_text_media_order", "rendered_prompt_text", "media_marker_text_and_positions"],
        "oracle": "构造交错的文本与媒体项目，检查渲染文本中的标记种类、数量和相对位置；仅检查子串的断言只证明该部分。媒体载荷、特征和最终 token 占位范围另行检查。",
        "execution_boundary": "多模态消息到提示词标记文本；不包括媒体载荷读取、媒体特征展开或输入张量绑定。",
        "evidence": {
            "vllm": refs(
                ("vllm/tokenizers/deepseek_v4_encoding.py", 220, "source"),
                ("vllm/renderers/inkling_encoding.py", 247, "source"),
                ("tests/renderers/test_inkling.py", 154, "existing_test"),
                ("tests/renderers/test_inkling.py", 165, "existing_oracle"),
                ("tests/tokenizers_/test_deepseek_v4.py", 451, "existing_test"),
                ("tests/tokenizers_/test_deepseek_v4.py", 469, "existing_oracle")),
            "sglang": refs(
                ("python/sglang/srt/parser/conversation.py", 574, "source"),
                ("python/sglang/srt/parser/conversation.py", 644, "source"),
                ("test/registered/unit/parser/test_conversation.py", 520, "existing_test"),
                ("test/registered/unit/parser/test_conversation.py", 536, "existing_oracle"),
                ("test/registered/unit/parser/test_conversation.py", 554, "existing_test"),
                ("test/registered/unit/parser/test_conversation.py", 570, "existing_oracle")),
        },
    }
    units = [prompt, media]
    upos = next(i for i, x in enumerate(data["units"])
                if x["id"] == "input.chat_template_tokenization")
    data["units"][upos:upos] = units
    for unit in units:
        for framework, evidence in unit["evidence"].items():
            for ref in evidence:
                lines = (ROOTS[framework] / ref["path"]).read_text(
                    encoding="utf-8", errors="replace").splitlines()
                assert 1 <= ref["line"] <= len(lines) and lines[ref["line"]-1].strip(), ref
    fd, temp = tempfile.mkstemp(prefix="space_seed.", suffix=".json", dir=SEED.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, SEED)
    print("added", [unit["id"] for unit in units], "units", len(data["units"]))


if __name__ == "__main__":
    main()
