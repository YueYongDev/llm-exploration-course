# providers.py
import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pillow_heif
import requests
from PIL import Image, ExifTags

# ------------- 环境配置 -------------
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

PHOTO_DIR = Path(os.getenv("PHOTO_DIR", "./photos")).resolve()
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")  # 也可 bge-m3 等

JSONL_PATH = DATA_DIR / "photos.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"
IDMAP_PATH = DATA_DIR / "id_map.json"     # id -> (path, caption, shot_time) 的映射
DIM_PATH = DATA_DIR / "dim.txt"           # 记录向量维度

pillow_heif.register_heif_opener()

# ------------- FAISS（可选）-------------
_FAISS_OK = True
try:
    import faiss  # type: ignore
except Exception:
    _FAISS_OK = False

# 全局状态：小 demo 直接用
_index = None                      # FAISS index 或 None
_id_list: List[str] = []           # 向量顺序对应的 id 列表
_meta: Dict[str, Dict] = {}        # id -> {path, caption, shot_time}

# ------------- 工具函数 -------------
def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _normalize(v: np.ndarray) -> np.ndarray:
    # 余弦检索：用内积 & 单位化
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / n

def clear_existing_data() -> None:
    """清除所有已存在的索引和数据文件"""
    # 清除JSONL文件
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()
    
    # 清除ID映射文件
    if IDMAP_PATH.exists():
        IDMAP_PATH.unlink()
    
    # 清除维度文件
    if DIM_PATH.exists():
        DIM_PATH.unlink()
    
    # 清除FAISS索引文件
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    
    # 清除向量文件（如果存在）
    V_PATH = DATA_DIR / "vectors.npy"
    if V_PATH.exists():
        V_PATH.unlink()
    
    # 清除临时目录中的文件
    temp_dir = DATA_DIR / "temp"
    if temp_dir.exists():
        import shutil
        shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 重置全局变量
    global _meta, _id_list, _index
    _meta = {}
    _id_list = []
    _index = None

def _load_meta() -> None:
    """从 JSONL 和 id_map.json 初始化内存映射（首次加载）"""
    global _meta, _id_list
    _meta = {}
    _id_list = []
    if IDMAP_PATH.exists():
        _meta = json.loads(IDMAP_PATH.read_text(encoding="utf-8"))
        _id_list = list(_meta.keys())
    elif JSONL_PATH.exists():
        # 首次构建 id_map
        with JSONL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rid = rec["id"]
                _meta[rid] = {
                    "path": rec["path"],
                    "caption": rec.get("caption", ""),
                    "shot_time": rec.get("shot_time"),
                    "tags": rec.get("tags", []),  # 加载标签
                }
                _id_list.append(rid)
        IDMAP_PATH.write_text(json.dumps(_meta, ensure_ascii=False, indent=2), encoding="utf-8")

def _save_meta_line(rec: Dict) -> None:
    """向 JSONL 追加一行，并同步 id_map.json"""
    with JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    _meta[rec["id"]] = {
        "path": rec["path"],
        "caption": rec.get("caption", ""),
        "shot_time": rec.get("shot_time"),
        "tags": rec.get("tags", []),  # 保存标签
    }
    _id_list.append(rec["id"])
    IDMAP_PATH.write_text(json.dumps(_meta, ensure_ascii=False, indent=2), encoding="utf-8")

def extract_exif_datetime(path: str) -> Optional[str]:
    """尽量从 EXIF 获取拍摄时间（ISO 字符串）"""
    try:
        im = Image.open(path)
        exif = im.getexif()
        if not exif:
            return None
        dt_str = None
        for tag_id, val in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag in ("DateTimeOriginal", "DateTime"):
                dt_str = str(val)
                break
        if dt_str:
            # 常见 EXIF 形如 2023:05:12 10:21:00
            dt_str = dt_str.replace("-", ":")
            y, m, d = dt_str.split(" ")[0].split(":")
            t = dt_str.split(" ")[1]
            return f"{y}-{m}-{d}T{t}"
    except Exception:
        return None
    return None

def convert_heic_to_jpg(heic_path: str) -> str:
    """
    将HEIC格式图片转换为JPEG格式，并保存在临时目录中
    返回转换后JPEG图片的路径
    """
    # 检查是否为HEIC格式（不区分大小写）
    if not heic_path.lower().endswith('.heic'):
        return heic_path
    
    # 创建临时目录用于存放转换后的JPEG文件
    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成JPEG文件路径
    heic_filename = Path(heic_path).stem
    jpg_path = str(temp_dir / f"{heic_filename}.jpg")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(heic_path):
            print(f"HEIC文件不存在: {heic_path}")
            return heic_path
            
        # 尝试打开HEIC图片并转换为JPEG
        with Image.open(heic_path) as image:
            # 转换为RGB模式（去除alpha通道）
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            # 保存为JPEG格式
            image.save(jpg_path, 'JPEG', quality=95)
        
        return jpg_path
    except Exception as e:
        # 如果转换失败，记录详细错误并返回原始路径
        print(f"转换HEIC到JPEG失败: {e}")
        print(f"文件路径: {heic_path}")
        try:
            # 尝试获取文件信息
            stat = os.stat(heic_path)
            print(f"文件大小: {stat.st_size} 字节")
        except Exception as stat_err:
            print(f"获取文件信息失败: {stat_err}")
            
        # 返回原始路径，让后续处理决定如何处理
        return heic_path

# ------------- Ollama -------------
def embed_text(text: str) -> np.ndarray:
    """调用 Ollama embeddings 接口，把文本转向量"""
    payload = {"model": EMBED_MODEL, "prompt": text}
    r = requests.post(f"{OLLAMA_BASE}/api/embeddings", json=payload, timeout=120)
    r.raise_for_status()
    vec = r.json().get("embedding") or r.json().get("embeddings")
    if isinstance(vec, list) and isinstance(vec[0], (float, int)):
        arr = np.array([vec], dtype="float32")
    elif isinstance(vec, list) and isinstance(vec[0], list):
        arr = np.array(vec, dtype="float32")
    else:
        raise RuntimeError("Invalid embeddings response from Ollama")
    return arr  # shape (1, d)
# 参考：Ollama embeddings 接口与示例。 [oai_citation:8‡ollama.apidog.io](https://ollama.apidog.io/examples-14809545e0?utm_source=chatgpt.com) [oai_citation:9‡Ollama](https://ollama.com/blog/embedding-models?utm_source=chatgpt.com) [oai_citation:10‡Stack Overflow](https://stackoverflow.com/questions/79364221/what-is-the-right-way-to-generate-ollama-embeddings?utm_source=chatgpt.com)

def caption_image(image_path: str, model_name: str = "gemma3:4b") -> dict:
    """用指定的多模态模型生成一句话图像描述"""
    # 如果是HEIC格式，先转换为JPEG
    processed_image_path = convert_heic_to_jpg(image_path)
    
    try:
        with open(processed_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"读取图片文件失败: {e}")
        return {
            "description": f"无法读取图片文件: {Path(image_path).name}",
            "tags": []
        }
    
    payload = {
        "model": model_name,
        "prompt": (
            "请仔细分析这张图片，详细描述图片的主要内容，包括但不限于："
            "图片的分类、主体物体、颜色、环境背景、光线状况、可能的拍摄设备、物体之间的关系、材质等。"
            "请避免模糊描述，尽量具体。然后提取2-5个与图片内容高度相关的标签。同时也要标记出图片的类型，例如：风景、人物、动物、建筑、截图等。"
            "请严格以如下JSON格式返回内容，不要输出任何其它文本：\n"
            '{"description":"详细描述内容...","tags":["标签1","标签2"]}'
        ),
        "images": [b64],
        "stream": False,
        "format": "json"  # 启用JSON模式
    }
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
    except Exception as e:
        print(f"请求Ollama服务失败: {e}")
        return {
            "description": f"请求Ollama服务失败: {str(e)}",
            "tags": []
        }
    
    # 解析返回的JSON
    try:
        response_json = r.json()
        response_text = (response_json.get("response") or "").strip()
        
        # 尝试直接解析JSON
        result = json.loads(response_text)
        # 确保返回的是字典格式并包含必要的字段
        if isinstance(result, dict):
            if "description" not in result:
                result["description"] = response_text
            if "tags" not in result:
                result["tags"] = []
            return result
        else:
            # 如果不是字典格式，返回默认格式
            return {
                "description": response_text,
                "tags": []
            }
    except json.JSONDecodeError:
        # 如果直接解析失败，尝试修复常见的JSON格式问题
        try:
            fixed_text = response_text
            
            # 查找JSON对象的开始和结束位置
            start = fixed_text.find('{')
            end = fixed_text.rfind('}')
            
            if start != -1 and end != -1 and end > start:
                fixed_text = fixed_text[start:end+1]
                
                # 尝试解析截取后的文本
                result = json.loads(fixed_text)
                # 确保返回的是字典格式并包含必要的字段
                if isinstance(result, dict):
                    if "description" not in result:
                        result["description"] = response_text
                    if "tags" not in result:
                        result["tags"] = []
                    return result
            
            # 如果上述方法都不行，尝试手动提取内容
            # 这是一种简单但有效的处理方式
            description_match = None
            import re
            
            # 尝试从文本中提取描述信息
            desc_pattern = r'"description"\s*:\s*"([^"]*)"'
            desc_match = re.search(desc_pattern, fixed_text) if start != -1 else None
            
            if desc_match:
                description = desc_match.group(1)
            else:
                # 如果找不到明确的描述字段，使用整个响应文本
                description = response_text
                
            # 尝试提取标签
            tags = []
            tags_pattern = r'"tags"\s*:\s*$(.*?)$'
            tags_match = re.search(tags_pattern, fixed_text)
            if tags_match:
                try:
                    tags_str = tags_match.group(1)
                    # 简单解析标签数组
                    tag_matches = re.findall(r'"([^"]*)"', tags_str)
                    tags = list(tag_matches)
                except:
                    tags = []
            
            return {
                "description": description,
                "tags": tags
            }
        except Exception:
            # 如果所有修复方法都失败，返回默认格式
            return {
                "description": response_text,
                "tags": []
            }
    except Exception as e:
        print(f"解析响应失败: {e}")
        return {
            "description": f"解析响应失败: {str(e)}",
            "tags": []
        }
# 官方提示：REST API 可在 images 参数中传 base64 图像。 [oai_citation:11‡Ollama](https://ollama.com/blog/vision-models?utm_source=chatgpt.com)

# ------------- 索引 -------------
def _create_index(dim: int):
    """创建或加载 FAISS 索引（内积检索），并记录维度"""
    global _index
    if _FAISS_OK:
        if INDEX_PATH.exists():
            _index = faiss.read_index(str(INDEX_PATH))
        else:
            _index = faiss.IndexFlatIP(dim)  # 内积
            faiss.write_index(_index, str(INDEX_PATH))
        DIM_PATH.write_text(str(dim), encoding="utf-8")
    else:
        _index = None  # 将走 numpy 退化检索

def _load_index_if_needed(dim: int):
    global _index
    if _index is None:
        _create_index(dim)

def _add_to_index(vecs: np.ndarray):
    """把向量增量加入索引（自动单位化）"""
    if vecs.ndim == 1:
        vecs = vecs[None, :]
    vecs = vecs.astype("float32")
    vecs = _normalize(vecs)

    if _FAISS_OK:
        faiss.normalize_L2(vecs)  # 双保险
        _index.add(vecs)
        faiss.write_index(_index, str(INDEX_PATH))
    else:
        # 无 faiss 时，仅把向量存到 .npy 做退化检索
        V_PATH = DATA_DIR / "vectors.npy"
        if V_PATH.exists():
            old = np.load(V_PATH)
            new = np.vstack([old, vecs])
            np.save(V_PATH, new)
        else:
            np.save(V_PATH, vecs)

def _vectors_all() -> np.ndarray:
    """退化检索：加载所有已存向量"""
    V_PATH = DATA_DIR / "vectors.npy"
    if V_PATH.exists():
        return np.load(V_PATH)
    return np.zeros((0, int(DIM_PATH.read_text() or "0")), dtype="float32")

def ensure_loaded():
    """首次调用时加载元数据与索引"""
    _load_meta()
    # 尝试从 dim.txt 获取维度；没有就等第一次 embed 时写入
    if DIM_PATH.exists():
        dim = int(DIM_PATH.read_text().strip())
        if dim > 0:
            _load_index_if_needed(dim)

# ------------- 对外：构建 & 查询 -------------
def build_records(paths: List[str], use_vision: bool = False, vision_model: str = "gemma3:4b") -> Iterable[str]:
    """
    构建/追加索引：针对给定图片绝对路径列表
    逐条 yield 日志（供 Gradio 文本框流式显示）

    参数:
        paths: 图片的绝对路径列表
        use_vision: 是否使用视觉模型生成描述
        vision_model: 用于生成图片描述的多模态模型名称
    """
    if not paths:
        yield "⚠️ 没有选择任何图片。"
        return

    # 清除之前的数据
    clear_existing_data()
    yield "🔄 已清除之前的索引数据"

    ensure_loaded()
    yield f"📁 本次共 {len(paths)} 张，开始构建索引…"

    ok = 0
    dim_written = DIM_PATH.exists()
    for p in paths:
        try:
            p = str(p)
            if not os.path.exists(p):
                yield f"❌ 跳过：路径不存在 {p}"
                continue

            rid = Path(p).stem + "_" + str(int(time.time() * 1000))
            shot_time = extract_exif_datetime(p)
            
            # 处理新的caption格式
            if use_vision:
                caption_data = caption_image(p, vision_model)
                if isinstance(caption_data, dict):
                    caption = caption_data.get("description", Path(p).stem)
                    tags = caption_data.get("tags", [])
            else:
                caption = Path(p).stem
                tags = []
                
            # 如果是HEIC格式图片，使用转换后的JPEG进行向量化
            processed_image_path = convert_heic_to_jpg(p)
            
            try:
                vec = embed_text(caption)  # (1, d)
                d = vec.shape[1]
            except Exception as e:
                yield f"❌ 特征提取失败：{Path(p).name} | {e}"
                continue

            # 首次写入记录维度，随后加载/创建索引
            if not dim_written:
                _create_index(d)
                dim_written = True
            else:
                _load_index_if_needed(d)

            _add_to_index(vec)
            rec = {
                "id": rid,
                "path": p,  # 保存原始路径
                "shot_time": shot_time,
                "caption": caption,
                "tags": tags,  # 保存标签
                "vec": None,  # 向量不放 JSONL，节省空间；如需可改为 list(vec[0].tolist())
                "created_at": _now_iso(),
            }
            _save_meta_line(rec)
            ok += 1
            
            # 在日志中显示使用的实际路径
            if p != processed_image_path:
                yield f"✅ 已入库：{Path(p).name} (已转换为JPEG) | {caption[:40]}{'...' if len(caption) > 40 else ''}"
            else:
                yield f"✅ 已入库：{Path(p).name} | {caption[:40]}{'...' if len(caption) > 40 else ''}"
        except Exception as e:
            yield f"❌ 失败：{Path(p).name} | {e}"

    yield f"🎉 完成：成功 {ok}/{len(paths)}"

def search_topk(query: str, k: int = 10) -> List[Tuple[str, str]]:
    """查询：返回 (image_path, caption) 列表供 Gallery"""
    if not query.strip():
        return []
    q = embed_text(query)  # (1, d)
    d = q.shape[1]
    _load_index_if_needed(d)

    if _FAISS_OK and _index is not None and _index.ntotal > 0:
        q = _normalize(q.astype("float32"))
        # faiss 内部也会 normalize_L2，这里统一
        faiss.normalize_L2(q)
        scores, idxs = _index.search(q, min(k, len(_id_list)))
        order = idxs[0]
    else:
        # 退化：numpy 计算所有内积
        V = _vectors_all()
        if V.shape[0] == 0:
            return []
        qn = _normalize(q.astype("float32"))
        Vn = _normalize(V)
        sims = (Vn @ qn[0].T).reshape(-1)
        order = np.argsort(-sims)[: min(k, len(_id_list))]

    results: List[Tuple[str, str]] = []
    for i in order:
        if i < 0 or i >= len(_id_list):
            continue
        rid = _id_list[i]
        meta = _meta.get(rid)
        if not meta:
            continue
        p = meta["path"]
        
        # 如果是HEIC格式图片，确保返回可以显示的JPEG版本
        if p.lower().endswith('.heic'):
            # 检查转换后的JPEG文件是否存在
            heic_filename = Path(p).stem
            jpg_path = str(DATA_DIR / "temp" / f"{heic_filename}.jpg")
            if os.path.exists(jpg_path):
                p = jpg_path  # 使用转换后的JPEG文件
            # 如果JPEG文件不存在，仍然使用原始HEIC路径（依赖系统支持）
        
        if os.path.exists(p):
            # 构造包含标签的显示文本
            tags = meta.get("tags", [])
            description = meta.get("caption") or ""
            display_text = ""
            if tags:
                display_text = "标签: " + ", ".join(tags)
                display_text += "\n\n"
            display_text += "描述:" + description
            results.append((p, display_text))
    return results
