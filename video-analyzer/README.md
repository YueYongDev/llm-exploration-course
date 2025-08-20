# Video Analyzer

Video Analyzer 是一个用于视频内容分析的工具，能够提取视频帧、处理音频并利用 LLM（大语言模型）进行内容理解。

## 功能特性

- **视频帧提取**：从视频中提取关键帧图像
- **音频提取与转录**：将视频音频提取并转换为文本
- **内容理解**：通过 LLM 对帧和音频文本进行语义分析
- **可视化界面**：提供基于Gradio的展示界面

## 技术栈

- Python 3.10+
- OpenCV：用于视频帧提取
- Whisper/faster-whisper：用于音频转录
- Ollama：作为LLM后端
- Gradio：用于可视化界面

## 安装与使用

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 安装并启动 Ollama：
   访问 [Ollama官网](https://ollama.com/) 下载并安装，然后运行：
   ```bash
   ollama run gemma3:4b
   ```

3. 启动展示界面：
   ```bash
   python app.py
   ```

## 架构说明

项目采用模块化架构设计：
- `frame.py`：负责视频帧提取
- `audio_processor.py`：负责音频处理和转录
- `analyzer.py`：协调分析流程
- `clients/ollama.py`：Ollama客户端实现
- `app.py`：Gradio展示界面

## 注意事项

此项目当前仅提供展示界面，实际的视频分析功能需要通过源代码调用。