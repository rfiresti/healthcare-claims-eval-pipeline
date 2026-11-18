-- run from infra/postgres/ 
-- make sure to run as app_user using: 
--   docker compose exec -T postgres psql -U app_user -d claims_db < ../../tests/test_rls_isolation.sql
-- use ROLLBACK at the end to avoid leaving test data in the database

\set ON_ERROR_STOP on
BEGIN;

CREATE TEMP TABLE results (name text, ok boolean);

-- returns true if the statement is rejected by RLS
CREATE FUNCTION pg_temp.blocked(stmt text) RETURNS boolean AS $$
BEGIN
  EXECUTE stmt;
  RETURN false;
EXCEPTION WHEN insufficient_privilege THEN
  RETURN true;
END $$ LANGUAGE plpgsql;

INSERT INTO results
SELECT 'running as a role that cannot bypass RLS', NOT (rolsuper OR rolbypassrls)
FROM pg_roles WHERE rolname = current_user;

INSERT INTO results
SELECT 'RLS enabled and forced on 6 tables', count(*) = 6
FROM pg_class
WHERE relname IN ('documents', 'vector_embeddings', 'conversations',
                  'checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
  AND relrowsecurity AND relforcerowsecurity;

-- creating tenant A data
SET LOCAL app.current_tenant_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';
INSERT INTO conversations (id, tenant_id)
  VALUES ('test-conv-a', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa');
INSERT INTO documents (id, tenant_id, doc_type, source_name, content)
  VALUES ('11111111-1111-1111-1111-111111111111', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'policy', 'doc-a', 'text a');
INSERT INTO vector_embeddings (tenant_id, document_id, clause_id, content, embedding, model_name)
  VALUES ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '11111111-1111-1111-1111-111111111111', 'C-1', 'clause a',
          array_fill(0.1::real, ARRAY[384])::vector(384), 'test');
INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint)
  VALUES ('test-conv-a', '', 'cp-1', '{}');

-- creating tenant B data
SET LOCAL app.current_tenant_id = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb';
INSERT INTO conversations (id, tenant_id)
  VALUES ('test-conv-b', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb');
INSERT INTO documents (id, tenant_id, doc_type, source_name, content)
  VALUES ('22222222-2222-2222-2222-222222222222', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'policy', 'doc-b', 'text b');
INSERT INTO vector_embeddings (tenant_id, document_id, clause_id, content, embedding, model_name)
  VALUES ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', '22222222-2222-2222-2222-222222222222', 'C-1', 'clause b',
          array_fill(0.2::real, ARRAY[384])::vector(384), 'test');

-- testing if tenant B can only see its own rows. If RLS is working, count(*) should be 1 for each table, and 0 for checkpoints since B has no checkpoint data.
INSERT INTO results SELECT 'B sees only its documents',     count(*) = 1 FROM documents;
INSERT INTO results SELECT 'B sees only its embeddings',    count(*) = 1 FROM vector_embeddings;
INSERT INTO results SELECT 'B sees only its conversations', count(*) = 1 FROM conversations;
INSERT INTO results SELECT 'B sees no checkpoints',         count(*) = 0 FROM checkpoints;
INSERT INTO results SELECT 'B cannot write to A''s thread',
  pg_temp.blocked($$INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint)
                    VALUES ('test-conv-a', '', 'cp-bad', '{}')$$);

-- testing if tenant A can only see its own rows. If RLS is working, count(*) should be 1 for each table, and 0 for checkpoints since A has no checkpoint data.
SET LOCAL app.current_tenant_id = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa';
INSERT INTO results SELECT 'A sees only its documents',     count(*) = 1 FROM documents;
INSERT INTO results SELECT 'A sees only its embeddings',    count(*) = 1 FROM vector_embeddings;
INSERT INTO results SELECT 'A sees only its conversations', count(*) = 1 FROM conversations;
INSERT INTO results SELECT 'A sees its checkpoint',         count(*) = 1 FROM checkpoints;


INSERT INTO results SELECT 'A cannot insert a document for B',
  pg_temp.blocked($$INSERT INTO documents (tenant_id, doc_type, source_name, content)
                    VALUES ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'policy', 'x', 'x')$$);
INSERT INTO results SELECT 'A cannot insert a conversation for B',
  pg_temp.blocked($$INSERT INTO conversations (id, tenant_id)
                    VALUES ('x', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb')$$);
INSERT INTO results SELECT 'A cannot insert an embedding for B',
  pg_temp.blocked($$INSERT INTO vector_embeddings (tenant_id, document_id, clause_id, content, embedding, model_name)
                    VALUES ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', '22222222-2222-2222-2222-222222222222', 'C-9', 'x',
                            array_fill(0.3::real, ARRAY[384])::vector(384), 'test')$$);
INSERT INTO results SELECT 'A cannot move its document to B',
  pg_temp.blocked($$UPDATE documents SET tenant_id = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
                    WHERE id = '11111111-1111-1111-1111-111111111111'$$);
INSERT INTO results SELECT 'A cannot write to a thread with no conversation',
  pg_temp.blocked($$INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint)
                    VALUES ('no-such-thread', '', 'cp-x', '{}')$$);

WITH u AS (UPDATE documents SET content = 'changed' WHERE id = '22222222-2222-2222-2222-222222222222' RETURNING 1)
INSERT INTO results SELECT 'A cannot update B''s document', count(*) = 0 FROM u;

WITH d AS (DELETE FROM conversations WHERE id = 'test-conv-b' RETURNING 1)
INSERT INTO results SELECT 'A cannot delete B''s conversation', count(*) = 0 FROM d;

-- testing what happens if the current_tenant_id is not set. If RLS is working, count(*) should be 0 for all tables.
SET LOCAL app.current_tenant_id = '';
INSERT INTO results SELECT 'no tenant: documents empty',         count(*) = 0 FROM documents;
INSERT INTO results SELECT 'no tenant: embeddings empty',        count(*) = 0 FROM vector_embeddings;
INSERT INTO results SELECT 'no tenant: conversations empty',     count(*) = 0 FROM conversations;
INSERT INTO results SELECT 'no tenant: checkpoints empty',       count(*) = 0 FROM checkpoints;
INSERT INTO results SELECT 'no tenant: checkpoint_blobs empty',  count(*) = 0 FROM checkpoint_blobs;
INSERT INTO results SELECT 'no tenant: checkpoint_writes empty', count(*) = 0 FROM checkpoint_writes;

SELECT name, CASE WHEN ok THEN 'pass' ELSE 'FAIL' END AS result FROM results;
SELECT count(*) FILTER (WHERE NOT ok) AS failures, count(*) AS total FROM results;

ROLLBACK;