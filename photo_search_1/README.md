# LLM Exploration Course – 自然语言搜图 MVP

本仓库提供一个最小可运行的「自然语言搜图」演示：**Gradio 前端 + PostgreSQL/pgvector + 本地文本嵌入**。  
先用“弱 caption”（文件名/目录名）完成文本检索闭环，后续可替换为 CLIP 图像向量与自动 caption。

## 1) 环境
- Python 3.10+
- PostgreSQL 15+，安装 `pgvector` 扩展
- 可选：GPU 仅在更换为 CLIP 图像向量时收益明显

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，配置 PG_DSN 与 PHOTO_ROOT