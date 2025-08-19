# LLM Exploration Course – 自然语言搜图 MVP

本仓库提供一个最小可运行的「自然语言搜图」演示：**Gradio 前端 + Ollama + 本地文本嵌入**。  
先用“弱 caption”（文件名/目录名）完成文本检索闭环，后续可替换为 CLIP 图像向量与自动 caption。

## 1) 环境
- Python 3.10+
- Ollama服务（用于图像描述和文本嵌入）
- 可选：GPU仅在使用大型视觉模型时收益明显

## 2) 功能特性

### 多模态图像描述生成
- 支持多种视觉模型生成图像描述：
  - `gemma3:4b` - Google Gemma 3 4B 参数模型
  - `gemma3:12b` - Google Gemma 3 12B 参数模型
  - `llama3.2-vision:11b` - Meta Llama 3.2 Vision 11B 参数模型
  - `minicpm-v:8b` - MiniCPM-V 8B 参数模型
- 可选择关闭图像描述功能，使用默认的文件名作为描述

### HEIC 图像格式支持
- 自动检测并转换 HEIC 格式图片为 JPEG 格式进行处理
- 保留原始 HEIC 文件，仅在处理时使用转换后的 JPEG 文件

### 智能索引构建
- 支持增量构建索引
- 实时显示构建进度日志
- 构建过程中禁用搜索功能，防止冲突操作

## 3) 使用方法

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件配置相关参数
   ```
   
   可配置参数：
   - `OLLAMA_BASE`: Ollama 服务地址 (默认: http://127.0.0.1:11434)
   - `EMBED_MODEL`: 文本嵌入模型 (默认: bge-m3)
   - `DATA_DIR`: 数据存储目录 (默认: ./data)
   - `PHOTO_DIR`: 图片存储目录 (默认: ./photos)

3. 启动 Ollama 服务并确保相关模型已下载：
   ```bash
   # 启动ollama服务
   ollama serve
   
   # 下载所需模型（根据需要选择）
   ollama pull gemma3:4b
   ollama pull gemma3:12b
   ollama pull llama3.2-vision:11b
   ollama pull minicpm-v:8b
   ollama pull bge-m3
   ```

4. 启动应用：
   ```bash
   python photo/search_app.py
   ```

5. 在 Web 界面中：
   - 上传图片文件构建索引
   - 选择视觉模型生成图像描述
   - 输入自然语言查询进行图片搜索

## 4) 技术架构

- 前端：Gradio Web界面
- 后端：Python脚本处理图像、文本向量化
- 向量存储：FAISS向量索引（本地文件存储）
- 模型服务：Ollama本地模型服务

## 5) 工作流程

1. 用户通过Gradio界面上传图片文件
2. 系统支持选择不同的视觉模型生成图像描述
3. 系统自动提取图像EXIF信息（如拍摄时间）
4. 使用嵌入模型将图像描述转换为向量表示
5. 构建FAISS向量索引用于高效检索
6. 用户输入自然语言查询进行图片搜索
7. 查询文本同样转换为向量并在索引中进行相似度搜索
8. 返回匹配的图像结果并展示

## 6) 目录结构

```
photo_search/
├── .env.example           # 环境变量示例配置
├── requirements.txt       # 项目依赖
├── common/                # 核心功能模块
│   └── providers.py       # 核心处理逻辑
├── photo/                 # 前端界面
│   └── search_app.py      # Gradio应用入口
├── data/                  # 数据存储目录
│   ├── temp/              # HEIC转换后的JPEG文件
│   ├── photos.jsonl       # 图片元数据
│   ├── id_map.json        # ID映射文件
│   ├── index.faiss        # FAISS向量索引
│   └── dim.txt            # 向量维度信息
└── photos/                # 默认图片存储目录
```