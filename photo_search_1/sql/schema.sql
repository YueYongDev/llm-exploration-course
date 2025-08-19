CREATE TABLE IF NOT EXISTS photos (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE,
    shot_time TIMESTAMP,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    camera_model TEXT,
    lens_model TEXT,
    img_vec VECTOR(1024)  -- bge-m3 返回 1024 维
);

-- 推荐直接建好索引
CREATE INDEX IF NOT EXISTS photos_img_vec_hnsw
    ON photos
    USING hnsw (img_vec vector_cosine_ops);