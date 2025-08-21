#!/usr/bin/env python3
"""
测试脚本，用于以编程方式运行视频分析器
"""

import json
import logging
import os
import sys
from pathlib import Path

import dotenv

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
dotenv.load_dotenv(project_root / '.env')

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


def get_env_config():
    """
    从环境变量获取配置
    """
    config = {
        # 客户端配置
        "client_type": os.getenv("CLIENT_TYPE", "ollama"),
        "ollama_url": os.getenv("OLLAMA_URL", "http://192.168.100.201:11434"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "minicpm-v:latest"),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "openai_api_url": os.getenv("OPENAI_API_URL", "https://openrouter.ai/api/v1"),
        "openai_model": os.getenv("OPENAI_MODEL", "meta-llama/llama-3.2-11b-vision-instruct"),

        # 通用配置
        "temperature": float(os.getenv("TEMPERATURE", "0.2")),
        "output_dir": os.getenv("OUTPUT_DIR", "output"),
        "keep_frames": os.getenv("KEEP_FRAMES", "false").lower() == "true",

        # 音频处理配置
        "whisper_model": os.getenv("WHISPER_MODEL", "medium"),
        "audio_device": os.getenv("AUDIO_DEVICE", "cpu"),
        "audio_language": os.getenv("AUDIO_LANGUAGE", None),

        # 帧处理配置
        "frames_per_minute": int(os.getenv("FRAMES_PER_MINUTE", "30")),
        "max_frames": int(os.getenv("MAX_FRAMES", "100")),

        # 自定义提示
        "custom_prompt": os.getenv("CUSTOM_PROMPT", ""),
    }

    return config


def run_video_analysis(video_path: str):
    """
    运行视频分析的主要函数
    
    Args:
        video_path: 视频文件路径
        config: 配置对象，如果为 None 则使用默认配置
    """
    # 从环境变量获取配置
    env_config = get_env_config()

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 创建输出目录
    output_dir = Path(env_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化客户端
    client_type = env_config["client_type"]
    if client_type == "ollama":
        client = OllamaClient(env_config["ollama_url"])
        model = env_config["ollama_model"]
    else:
        raise ValueError(f"Unknown client type: {client_type}")

    logger.info(f"Using client: {client_type}, model: {model}")

    # 初始化其他组件
    # 注意：这里仍然需要使用默认的提示文件
    prompt_loader = PromptLoader("video_analyzer/prompts", [
        {
            "name": "Frame Analysis",
            "path": "frame_analysis/frame_analysis.txt"
        },
        {
            "name": "Video Reconstruction",
            "path": "frame_analysis/describe.txt"
        }
    ])

    transcript = None
    frames = []
    frame_analyses = []
    video_description = None

    try:
        # 阶段1: 音频和帧处理
        logger.info("Stage 1: Processing audio and extracting frames...")

        # 初始化音频处理器
        audio_processor = AudioProcessor(
            language=None,
            model_size_or_path='medium',
            device='auto',
            num_workers=2,
            cpu_threads=8
        )

        # 提取音频
        logger.info("Extracting audio from video...")
        try:
            audio_path = audio_processor.extract_audio(video_path, output_dir)
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            audio_path = None

        # 转录音频
        if audio_path is None:
            logger.debug("No audio found in video - skipping transcription")
        else:
            logger.info("Transcribing audio...")
            transcript = audio_processor.transcribe(audio_path)
            if transcript is None:
                logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")

        # 提取帧
        logger.info(f"Extracting frames from video...")
        # 根据经验教训，这里不应该传递模型参数，因为帧提取是纯CV操作
        processor = VideoProcessor(
            video_path,
            output_dir / "frames",
            ""  # 不再传递模型参数
        )
        frames = processor.extract_keyframes(
            frames_per_minute=env_config["frames_per_minute"],
            max_frames=env_config["max_frames"]
        )

        # 阶段2: 帧分析
        logger.info("Stage 2: Analyzing frames...")
        analyzer = VideoAnalyzer(
            client,
            model,
            prompt_loader,
            0.2,
            ""
        )

        for frame in frames:
            analysis = analyzer.analyze_frame(frame)
            frame_analyses.append(analysis)
            logger.info(f"Analyzed frame {frame.number}")

        # 阶段3: 视频重建
        logger.info("Stage 3: Reconstructing video description...")
        video_description = analyzer.reconstruct_video(frame_analyses, frames, transcript)

        # 保存结果
        results = {
            "metadata": {
                "client": client_type,
                "model": model,
                "whisper_model": env_config["whisper_model"],
                "frames_per_minute": env_config["frames_per_minute"],
                "frames_extracted": len(frames),
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None
            } if transcript else None,
            "frame_analyses": frame_analyses,
            "video_description": video_description
        }

        results_file = output_dir / "analysis.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"Analysis complete. Results saved to {results_file}")

        # 输出结果摘要
        if transcript and transcript.text:
            logger.info("\nTranscript:")
            logger.info(transcript.text[:200] + "..." if len(transcript.text) > 200 else transcript.text)

        if video_description:
            logger.info("\nVideo Description:")
            description_text = video_description.get("response", "No description generated")
            logger.info(description_text[:500] + "..." if len(description_text) > 500 else description_text)

        return results

    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        raise
    finally:
        # 清理临时文件
        if not env_config["keep_frames"]:
            cleanup_files(output_dir)


def cleanup_files(output_dir: Path):
    """
    清理临时文件
    """
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            import shutil
            shutil.rmtree(frames_dir)
            logger.debug(f"Cleaned up frames directory: {frames_dir}")

        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def example_usage():
    """
    示例用法
    """
    # 运行分析
    video_file = "path/to/your/test_video.mp4"  # 替换为实际视频路径
    try:
        results = run_video_analysis(video_file)
        print("Analysis completed successfully!")
        return results
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


if __name__ == "__main__":
    # 如果直接运行此脚本，显示使用示例
    print("Video Analyzer Test Script")
    print("=" * 30)
    print("This script demonstrates how to use the video analyzer programmatically.")
    print("To use it in your project, make sure to set up the .env file and call run_video_analysis().")
    print("\nExample usage:")
    print(">>> results = run_video_analysis('path/to/video.mp4')")

    video_path = "/Users/yueyong/Downloads/video 2.mp4"
    print(f"\nAnalyzing video: {video_path}")
    results = run_video_analysis(video_path)