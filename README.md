# LLM Exploration Course

这是一个探索大型语言模型（LLM）应用的课程项目集合，包含多个实验性模块，旨在展示如何结合LLM进行实际问题解决。项目通过自然语言处理技术实现图像搜索功能。

## 项目结构

```
llm-exploration-course/
├── photo_search_1/           # 自然语言搜图 MVP 实现
│   ├── common/               # 公共模块，包含核心功能实现
│   ├── photo/                # 图像处理相关功能
│   ├── sql/                  # 数据库 schema 定义
│   ├── requirements.txt      # 项目依赖
│   └── README.md             # 子项目说明
└── README.md                 # 项目根目录说明
```

## 功能特性

1. **自然语言搜图**：基于自然语言描述搜索相关图片
2. **图像向量化**：使用嵌入模型将图像和文本转换为向量表示
3. **EXIF信息提取**：自动提取图像元数据（如拍摄时间）
4. **向量相似度检索**：基于向量相似度进行高效图片检索

## 技术栈

- Python 3.10+
- PostgreSQL + pgvector（向量数据库）
- Gradio（Web界面）
- Ollama（本地模型服务）
- Pillow（图像处理）
- exifread（EXIF信息读取）

## 快速开始

### 环境要求

1. Python 3.10+
2. PostgreSQL 15+ 并安装 `pgvector` 扩展
3. Ollama 服务（用于运行嵌入模型）

### 安装步骤

```bash
# 克隆项目
git clone <项目地址>
cd llm-exploration-course

# 进入项目目录
cd photo_search_1

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，配置数据库连接和照片目录
```

### 配置说明

在 `.env` 文件中配置以下参数：

```env
# 数据库连接字符串
PG_DSN=postgresql://username:password@host:port/database

# 嵌入模型名称
EMBED_MODEL=bge-m3:latest

# 照片目录路径
PHOTO_DIR=/path/to/your/photos
```

### 运行应用

```bash
cd photo_search_1
python photo/search_app.py
```

访问 `http://127.0.0.1:7860` 使用图形界面。

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