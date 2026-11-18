-- run as admin_owner only after loading data. This will create the HNSW index on the vector_embeddings table. 
-- not fully needed for normal operation, but will speed up searches a lot
CREATE INDEX IF NOT EXISTS vector_embeddings_hnsw_idx
  ON vector_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);