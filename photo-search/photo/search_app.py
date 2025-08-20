# search_app.py
import gradio as gr

from common.providers import build_records, search_topk, PHOTO_DIR, DATA_DIR


def on_build(files_list, vision_model):
    paths = []
    for f in files_list or []:
        # gr.Files(type="filepath") 返回 str 路径
        paths.append(str(f))

    # 判断是否使用视觉模型（只要不是"无"就使用）
    use_vision = vision_model != "无"

    log_text = ""
    for line in build_records(paths, use_vision, vision_model):
        log_text += line + "\n"
        yield log_text


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
            vision_model = gr.Radio(
                choices=["gemma3:4b", "gemma3:12b", "llama3.2-vision:11b", "minicpm-v:8b"],
                value="gemma3:4b",
                label="选择视觉模型生成图片描述"
            )
            build_btn = gr.Button("🚀 构建索引", variant="primary")

            gr.Markdown("---\n**构建索引状态**")
            log = gr.Textbox(lines=12, label="日志", value="", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("**② 搜索**")
            q = gr.Textbox(value="我去年在土耳其拍的猫", label="查询")
            topk = gr.Slider(1, 10, value=2, step=1, label="返回数量")
            search_btn = gr.Button("🔎 搜索")

            gallery = gr.Gallery(label="结果", columns=6, height=720, value=[])


    # 在构建索引时同时更新日志和结果区域
    def build_with_controls(files_list, vision_model):
        # 禁用搜索按钮并更新Gallery标签
        yield "", gr.update(interactive=False, variant="secondary"), gr.update(label="结果 (⏳ 构建索引中...)")

        # 执行构建过程
        paths = []
        for f in files_list or []:
            paths.append(str(f))

        # 判断是否使用视觉模型（只要不是"无"就使用）
        use_vision = vision_model != "无"

        log_text = ""
        for line in build_records(paths, use_vision, vision_model):
            log_text += line + "\n"
            yield log_text, gr.update(interactive=False, variant="secondary"), gr.update(label="结果 (⏳ 构建索引中...)")

        # 完成后恢复Gallery标签和按钮状态
        yield log_text, gr.update(interactive=True, variant="primary"), gr.update(label="结果")


    build_btn.click(
        fn=build_with_controls,
        inputs=[files, vision_model],
        outputs=[log, search_btn, gallery],
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
    temp_dir_abs = str(DATA_DIR / "temp")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[photo_dir_abs, temp_dir_abs],  # 允许访问你的相册目录和临时目录
    )