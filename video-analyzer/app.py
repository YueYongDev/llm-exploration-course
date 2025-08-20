import logging
import sys
from datetime import datetime
from pathlib import Path

import gradio as gr

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from frame import VideoProcessor
from prompt import PromptLoader
from analyzer import VideoAnalyzer
from audio_processor import AudioProcessor
from clients.ollama import OllamaClient

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogCollector:
    """用于收集和管理日志信息的类"""

    def __init__(self):
        self.logs = []

    def add_log(self, message):
        """添加日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        return "\n".join(self.logs)

    def clear_logs(self):
        """清空日志"""
        self.logs = []


log_collector = LogCollector()


def analyze_video(video_path):
    """
    实际分析视频文件
    """
    # 清空之前的日志
    log_collector.clear_logs()
    log_message = log_collector.add_log("开始分析视频文件")
    yield log_message, "开始分析...", "🔄 正在初始化分析组件...", {}

    if not video_path:
        log_message = log_collector.add_log("错误: 请上传视频文件")
        yield log_message, "请上传视频文件", "❌ 请上传视频文件", {}
        return

    try:
        # 创建输出目录
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化客户端
        log_message = log_collector.add_log("初始化Ollama客户端")
        yield log_message, "初始化分析组件", "🔄 正在初始化Ollama客户端...", {}

        client = OllamaClient("http://localhost:11434")
        model = "gemma3:4b"

        # 初始化其他组件
        log_message = log_collector.add_log("加载提示模板")
        yield log_message, "初始化分析组件", "🔄 正在加载提示模板...", {}

        prompt_loader = PromptLoader("prompts", [
            {
                "name": "Frame Analysis",
                "path": "frame_analysis/frame_analysis.txt"
            },
            {
                "name": "Video Reconstruction",
                "path": "frame_analysis/describe.txt"
            }
        ])

        # 更新进度
        log_message = log_collector.add_log("组件初始化完成")
        yield log_message, "组件初始化完成", "✅ 组件初始化完成", {}

        # 初始化音频处理器
        log_message = log_collector.add_log("初始化音频处理器")
        yield log_message, "初始化音频处理器", "🔄 正在初始化音频处理器...", {}

        audio_processor = AudioProcessor(
            language=None,
            model_size_or_path='medium',
            device='cpu'
        )

        # 提取音频
        log_message = log_collector.add_log("开始提取音频")
        yield log_message, "开始提取音频", "🎵 正在提取音频轨道...", {}

        try:
            audio_path = audio_processor.extract_audio(video_path, output_dir)
            log_message = log_collector.add_log("音频提取完成")
            yield log_message, "音频提取完成", "✅ 音频提取完成", {}
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            log_message = log_collector.add_log(f"音频提取失败: {str(e)}")
            yield log_message, "音频提取失败", "⚠️ 音频提取失败，将仅进行视频分析", {}
            audio_path = None

        # 转录音频
        transcript = None
        if audio_path is None:
            log_message = log_collector.add_log("视频中未找到音频")
            yield log_message, "视频中未找到音频", "ℹ️ 视频中未找到音频轨道", {}
        else:
            log_message = log_collector.add_log("开始转录音频")
            yield log_message, "开始转录音频", "🎤 正在转录音频内容...", {}

            transcript = audio_processor.transcribe(audio_path)
            if transcript is None:
                log_message = log_collector.add_log("无法生成可靠的转录，仅进行视频分析")
                yield log_message, "无法生成可靠的转录", "⚠️ 无法生成可靠的转录，仅进行视频分析", {}
            else:
                log_message = log_collector.add_log("音频转录完成")
                yield log_message, "音频转录完成", "✅ 音频转录完成", {}

        # 提取帧
        log_message = log_collector.add_log("开始提取视频帧")
        yield log_message, "开始提取视频帧", "🎬 正在提取关键视频帧...", {}

        processor = VideoProcessor(
            video_path,
            output_dir / "frames",
            ""
        )
        frames = processor.extract_keyframes(
            frames_per_minute=30,
            max_frames=50
        )

        log_message = log_collector.add_log(f"视频帧提取完成，共提取 {len(frames)} 帧")
        yield log_message, f"视频帧提取完成，共提取 {len(frames)} 帧", f"✅ 视频帧提取完成，共提取 {len(frames)} 帧", {}

        # 分析帧
        log_message = log_collector.add_log("开始分析视频帧")
        yield log_message, "开始分析视频帧", "🔍 正在分析视频帧内容...", {}

        analyzer = VideoAnalyzer(
            client,
            model,
            prompt_loader,
            0.2,
            ""
        )

        frame_analyses = []
        for i, frame in enumerate(frames):
            log_message = log_collector.add_log(f"正在分析第 {i + 1}/{len(frames)} 帧")
            progress_percent = int((i / len(frames)) * 100)
            yield log_message, f"正在分析第 {i + 1}/{len(frames)} 帧", f"🔍 正在分析视频帧 ({i + 1}/{len(frames)}) - {progress_percent}%", {}

            analysis = analyzer.analyze_frame(frame)
            frame_analyses.append(analysis)

        log_message = log_collector.add_log("视频帧分析完成")
        yield log_message, "视频帧分析完成", "✅ 视频帧分析完成", {}

        # 重建视频描述
        log_message = log_collector.add_log("开始生成视频描述")
        yield log_message, "开始生成视频描述", "📝 正在生成综合视频描述...", {}

        video_description = analyzer.reconstruct_video(frame_analyses, frames, transcript)

        log_message = log_collector.add_log("视频描述生成完成")
        yield log_message, "视频描述生成完成", "✅ 视频描述生成完成", {}

        # 准备结果
        result = {
            "video_path": str(video_path),
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "frame_analyses": frame_analyses,
            "video_description": video_description
        }

        # 格式化结果展示
        output_text = f"""
## 视频分析结果
**视频路径:** {result['video_path']}

"""

        if result['transcript'] and result['transcript']['text']:
            output_text += f"""
### 🎵 音频转录
{result['transcript']['text']}

"""

        if result['video_description'] and 'response' in result['video_description']:
            output_text += f"""
### 📝 视频内容描述
{result['video_description']['response']}

"""

        output_text += "### 🎬 帧分析详情\n\n"

        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            if 'response' in analysis:
                output_text += f"**{i + 1}.** 时间点 `{frame.timestamp:.2f}s`: {analysis['response']}\n\n"

        log_message = log_collector.add_log("分析完成！")
        yield log_message, "分析完成！", output_text, result

    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}", exc_info=True)
        log_message = log_collector.add_log(f"分析出错: {str(e)}")
        yield log_message, "分析出错", f"❌ 分析出错: {str(e)}", {}


def run_analysis(video_path):
    """
    运行分析并返回结果
    """
    for log_message, status, result_text, result_json in analyze_video(video_path):
        yield log_message, status, result_text, result_json


with gr.Blocks(title="Video Analyzer") as demo:
    gr.Markdown("# Video Analyzer")
    gr.Markdown("上传视频文件进行内容分析，包括音频转录和视觉内容理解。")

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="上传视频")
            analyze_button = gr.Button("分析视频", variant="primary")
            status_output = gr.Textbox(label="当前状态", interactive=False)
            log_output = gr.Textbox(label="分析日志", interactive=False, lines=10, max_lines=10)

        with gr.Column():
            text_output = gr.Markdown(label="分析结果")
            json_output = gr.JSON(label="详细结果")

    # 设置事件处理
    analyze_button.click(
        fn=run_analysis,
        inputs=[video_input],
        outputs=[log_output, status_output, text_output, json_output]
    )

if __name__ == "__main__":
    demo.launch()
