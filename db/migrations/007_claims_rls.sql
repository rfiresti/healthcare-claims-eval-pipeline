--claims RLS
ALTER TABLE claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE claims FORCE ROW LEVEL SECURITY;

CREATE POLICY claims_tenant_isolation ON claims
  USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid);

-- claims labels RLS
ALTER TABLE claim_labels ENABLE ROW LEVEL SECURITY;
ALTER TABLE claim_labels FORCE ROW LEVEL SECURITY;

CREATE POLICY claim_labels_tenant_isolation ON claim_labels
  USING (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid)
  WITH CHECK (tenant_id = NULLIF(current_setting('app.current_tenant_id', true), '')::uuid);