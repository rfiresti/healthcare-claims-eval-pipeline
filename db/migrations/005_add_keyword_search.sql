-- Week 4: keyword search over policy clause chunks, for the hybrid retrieve() function.
--
-- Additive only: one generated column and one index on vector_embeddings. No grants or
-- policies change. vector_embeddings.content holds the clause-level chunk text and
-- clause_id, so keyword search goes here, not on documents.content.
--
-- Apply as admin_owner (the table owner):
--   docker exec -i pgvector-db psql -U admin_owner -d claims_db < db/migrations/005_add_keyword_search.sql

ALTER TABLE vector_embeddings
  ADD COLUMN IF NOT EXISTS content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS vector_embeddings_content_tsv_idx
  ON vector_embeddings USING gin (content_tsv);

