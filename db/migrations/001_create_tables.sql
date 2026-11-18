-- create document table: each row corresponds to a single source document (policy doc, EOB, or denial letter)
CREATE TABLE documents (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   uuid NOT NULL,
  doc_type    text NOT NULL CHECK (doc_type IN ('policy', 'eob', 'denial')),
  source_name text NOT NULL,
  content     text NOT NULL,
  metadata    jsonb NOT NULL DEFAULT '{}',
  created_at  timestamptz NOT NULL DEFAULT now(),
  UNIQUE (id, tenant_id)
);

-- create vector embeddings table: each document can have multiple embeddings, one per clause or chunk of text
CREATE TABLE vector_embeddings (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id   uuid NOT NULL,
  document_id uuid NOT NULL,
  clause_id   text,
  chunk_index int NOT NULL DEFAULT 0,
  content     text NOT NULL,
  embedding   vector(384) NOT NULL,
  model_name  text NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  FOREIGN KEY (document_id, tenant_id) REFERENCES documents (id, tenant_id) ON DELETE CASCADE,
  UNIQUE (tenant_id, document_id, clause_id, chunk_index)
);

-- langraph will handle the checkpoint table(s) automatically, but we need to create an intermediary conversation table so we don't have to query checkpoint automatically
-- indirect interaction with checkpoint table allows for more secure RLS
CREATE TABLE conversations (
  id         text PRIMARY KEY DEFAULT gen_random_uuid()::text,
  tenant_id  uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ON documents (tenant_id);
CREATE INDEX ON vector_embeddings (tenant_id);
CREATE INDEX ON conversations (tenant_id);

-- grant the app_user access to use the tables. We will use RLS to restrict access to only the rows that belong to the tenant_id of the user making the request
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;