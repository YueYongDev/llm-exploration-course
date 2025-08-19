import os
import base64
import requests
from pathlib import Path
from datetime import datetime
import exifread

from common.providers import get_db, encode_text  # 假设你已封装好 DB 与 embedding
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "gemma3:4b")
PHOTO_DIR = os.getenv("PHOTO_DIR", "./photos")

def generate_caption(image_path: str) -> str:
    """调用 Ollama 模型生成图像一句话描述，避免 bytes 序列化问题"""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": CAPTION_MODEL,
        "prompt": "请用一句话描述这张图片的内容。",
        "images": [b64],
        "stream": False
    }
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def extract_exif_datetime(path: str) -> datetime | None:
    with open(path, "rb") as f:
        tags = exifread.process_file(f, stop_tag="EXIF DateTimeOriginal")
        dt = tags.get("EXIF DateTimeOriginal")
        if dt:
            return datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S")
    return None

def upsert_photo(conn, row: dict):
    sql = """
    INSERT INTO photos (path, shot_time, caption, img_vec)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (path) DO UPDATE SET
      shot_time = EXCLUDED.shot_time,
      caption   = EXCLUDED.caption,
      img_vec   = EXCLUDED.img_vec;
    """
    conn.execute(sql, (row["path"], row["shot_time"], row["caption"], row["img_vec"]))

def main():
    photo_dir = PHOTO_DIR
    files = [p for p in Path(photo_dir).iterdir()
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"[ingest] scanning {photo_dir} → found {len(files)} files")

    with get_db() as conn:
        for p in files:
            path = str(p)
            print(f"[ingest] processing {path}")
            shot_time = extract_exif_datetime(path)
            caption = generate_caption(path)
            vec = encode_text(caption)
            row = {"path": path, "shot_time": shot_time, "caption": caption, "img_vec": vec}
            upsert_photo(conn, row)
        conn.commit()
    print("[ingest] done, total", len(files))

if __name__ == "__main__":
    main()