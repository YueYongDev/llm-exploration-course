# search_app.py
import os
from pathlib import Path
import gradio as gr

from providers import (
    PHOTO_DIR,
    build_records,
    search_topk,
)

def on_build(files_list, use_vision_flag):
    paths = []
    for f in files_list or []:
        # gr.Files(type="filepath") 返回 str 路径
        paths.append(str(f))
    for line in build_records(paths, bool(use_vision_flag)):
        yield line

def on_search(q, topk):
    return search_topk(q or "", int(topk))

with gr.Blocks(title="本地相册 · 自然语言搜图", theme="soft") as demo:
    gr.Markdown("### 本地相册 · 自然语言搜图\n左侧上传并构建索引 → 右侧查看检索结果")

    with gr.Row():
        with gr.Column(scale=1, min_width=360):
            gr.Markdown("**① 添加图片 & 构建索引**")
            files = gr.Files(
                label="选择要加入索引的图片（可多选）",
                file_count="multiple",
                type="filepath",
            )
            use_vision = gr.Checkbox(value=False, label="用 Gemma3:4B 生成图片描述（较慢）")
            build_btn = gr.Button("🚀 构建索引", variant="primary")

            gr.Markdown("---\n**② 搜索**")
            q = gr.Textbox(value="我去年在土耳其拍的猫", label="查询")
            topk = gr.Slider(1, 100, value=24, step=1, label="返回数量")
            search_btn = gr.Button("🔎 搜索")

            gr.Markdown("---\n**构建索引状态**")
            log = gr.Textbox(lines=12, label="日志", value="", interactive=False)

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="结果", columns=6, height=720, value=[])

    build_btn.click(
        fn=on_build,
        inputs=[files, use_vision],
        outputs=[log],
        queue=True,
        concurrency_limit=1,
    )

    search_btn.click(
        fn=on_search,
        inputs=[q, topk],
        outputs=[gallery],
        queue=False,
    )

if __name__ == "__main__":
    photo_dir_abs = str(PHOTO_DIR)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[photo_dir_abs],  # 允许访问你的相册目录
    )