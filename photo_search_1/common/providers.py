# providers.py
import os
import json
import base64
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import requests
from PIL import Image, ExifTags

# ------------- 环境配置 -------------
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

PHOTO_DIR = Path(os.getenv("PHOTO_DIR", "./photos")).resolve()
OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "gemma3:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")  # 也可 bge-m3 等

JSONL_PATH = DATA_DIR / "photos.jsonl"
INDEX_PATH = DATA_DIR / "index.faiss"
IDMAP_PATH = DATA_DIR / "id_map.json"     # id -> (path, caption, shot_time) 的映射
DIM_PATH = DATA_DIR / "dim.txt"           # 记录向量维度

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

def caption_image(image_path: str) -> str:
    """用 Gemma3:4B（或其他多模态）生成一句话图像描述"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": CAPTION_MODEL,
        "prompt": "请用一句话描述这张图片的内容。",
        "images": [b64],
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()
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
def build_records(paths: List[str], use_vision: bool = False) -> Iterable[str]:
    """
    构建/追加索引：针对给定图片绝对路径列表
    逐条 yield 日志（供 Gradio 文本框流式显示）
    """
    if not paths:
        yield "⚠️ 没有选择任何图片。"
        return

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
            caption = caption_image(p) if use_vision else Path(p).stem

            vec = embed_text(caption)  # (1, d)
            d = vec.shape[1]

            # 首次写入记录维度，随后加载/创建索引
            if not dim_written:
                _create_index(d)
                dim_written = True
            else:
                _load_index_if_needed(d)

            _add_to_index(vec)
            rec = {
                "id": rid,
                "path": p,
                "shot_time": shot_time,
                "caption": caption,
                "vec": None,  # 向量不放 JSONL，节省空间；如需可改为 list(vec[0].tolist())
                "created_at": _now_iso(),
            }
            _save_meta_line(rec)
            ok += 1
            yield f"✅ 已入库：{Path(p).name} | {caption[:40]}"
        except Exception as e:
            yield f"❌ 失败：{Path(p).name} | {e}"

    yield f"🎉 完成：成功 {ok}/{len(paths)}"

def search_topk(query: str, k: int = 24) -> List[Tuple[str, str]]:
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
        if os.path.exists(p):
            results.append((p, meta.get("caption") or ""))
    return results