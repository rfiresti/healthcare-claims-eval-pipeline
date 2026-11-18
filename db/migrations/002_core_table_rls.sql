-- Enable Row Level Security (RLS) for the documents table to ensure that users can only access documents belonging to their tenant.
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

CREATE POLICY documents_tenant_isolation ON documents
  USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid);

-- create RLS policies for the vector_embeddings table to ensure that users can only access embeddings belonging to their tenant.
ALTER TABLE vector_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE vector_embeddings FORCE ROW LEVEL SECURITY;

CREATE POLICY vector_embeddings_tenant_isolation ON vector_embeddings
  USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid);

-- create RLS on conversations table (ownership table that the checkpoint policies rely on)
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversations FORCE ROW LEVEL SECURITY;

CREATE POLICY conversations_tenant_isolation ON conversations
  USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid);