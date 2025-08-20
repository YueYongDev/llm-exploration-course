# LLM Exploration Course

这是一个探索大型语言模型（LLM）应用的课程项目集合，包含多个实验性模块，旨在展示如何结合LLM进行实际问题解决。项目通过自然语言处理技术实现图像搜索功能。

## 项目结构

```
llm-exploration-course/
├── photo_search/             # 自然语言搜图 MVP 实现
│   ├── common/               # 公共模块，包含核心功能实现
│   ├── photo/                # 图像处理相关功能
│   ├── data/                 # 数据存储目录
│   ├── photos/               # 默认图片存储目录
│   ├── requirements.txt      # 项目依赖
│   └── README.md             # 子项目说明
├── video-analyzer/           # 视频内容分析模块（开发中）
└── README.md                 # 项目根目录说明
```

## 功能特性

1. **自然语言搜图**：基于自然语言描述搜索相关图片
2. **图像向量化**：使用嵌入模型将图像和文本转换为向量表示
3. **EXIF信息提取**：自动提取图像元数据（如拍摄时间）
4. **向量相似度检索**：基于向量相似度进行高效图片检索
5. **多模型支持**：支持多种视觉模型生成图像描述
6. **HEIC格式支持**：自动检测并转换HEIC格式图片为JPEG格式进行处理

## 技术栈

- Python 3.10+
- Gradio（Web界面）
- Ollama（本地模型服务）
- Pillow（图像处理）
- FAISS（向量索引）
- pillow-heif（HEIC格式支持）

## 快速开始

进入 [photo_search](file:///Users/yueyong/Writing/llm-exploration-course/photo_search) 目录查看详细说明和使用方法：

```bash
cd photo-search
cat README.md
```

## 工作原理

1. **图像索引构建**：
   - 扫描指定目录中的图像文件
   - 提取图像EXIF信息（如拍摄时间）
   - 使用模型生成图像描述
   - 将描述文本转换为向量并存储到数据库

2. **图像搜索**：
   - 将查询语句转换为向量
   - 在向量数据库中查找最相似的图像向量
   - 返回匹配的图像结果

## 模型支持

项目默认使用以下模型：

- 文本嵌入：`bge-m3` 或 `mxbai-embed-large`
- 图像描述：`gemma3:4b`（可选）

可以通过修改 `.env` 文件中的 `EMBED_MODEL` 和 `CAPTION_MODEL` 变量来更换模型。

## 开发指南

### 目录说明

- `common/providers.py`：核心功能实现，包括向量计算、模型调用、数据库操作等
- `photo/ingest_exif.py`：图像信息提取和索引构建脚本
- `photo/search_app.py`：Gradio Web界面应用
- `sql/schema.sql`：数据库表结构定义

### 扩展功能

1. 替换嵌入模型：修改 `.env` 文件中的 `EMBED_MODEL` 参数
2. 更换图像描述模型：修改 `.env` 文件中的 `CAPTION_MODEL` 参数
3. 自定义搜索逻辑：修改 `common/providers.py` 中的 `search_topk` 函数