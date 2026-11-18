-- create RLS policies for checkpoints, checkpoint_blobs, and checkpoint_writes tables

-- checkpoint table RLS
ALTER TABLE checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoints FORCE ROW LEVEL SECURITY;

CREATE POLICY checkpoints_tenant_isolation ON checkpoints
  USING (EXISTS (
    SELECT 1 FROM conversations c
    WHERE c.id = checkpoints.thread_id
      AND c.tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid
  ))
  WITH CHECK (EXISTS (
    SELECT 1 FROM conversations c
    WHERE c.id = checkpoints.thread_id
      AND c.tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid
  ));

-- create RLS policy for checkpoint_blobs

ALTER TABLE checkpoint_blobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoint_blobs FORCE ROW LEVEL SECURITY;

CREATE POLICY checkpoint_blobs_tenant_isolation ON checkpoint_blobs
  USING (EXISTS (
    SELECT 1 FROM conversations c
    WHERE c.id = checkpoint_blobs.thread_id
      AND c.tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid
  ))
  WITH CHECK (EXISTS (
    SELECT 1 FROM conversations c
    WHERE c.id = checkpoint_blobs.thread_id
      AND c.tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid
  ));

-- create RLS policy for checkpoint_writes
ALTER TABLE checkpoint_writes ENABLE ROW LEVEL SECURITY;
ALTER TABLE checkpoint_writes FORCE ROW LEVEL SECURITY;

CREATE POLICY checkpoint_writes_tenant_isolation ON checkpoint_writes
  USING (EXISTS (
    SELECT 1 FROM conversations c
    WHERE c.id = checkpoint_writes.thread_id
      AND c.tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid
  ))
  WITH CHECK (EXISTS (
    SELECT 1 FROM conversations c
    WHERE c.id = checkpoint_writes.thread_id
      AND c.tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid
  ));

-- checkpoint_migrations does not contain tenant data so no RLS policy is needed
