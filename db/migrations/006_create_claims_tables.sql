-- claims: the input side of each synthetic claim. Everything the pipeline is allowed to see.
CREATE TABLE claims (
  tenant_id       uuid NOT NULL,
  claim_id        text NOT NULL,
  dataset_version text NOT NULL,
  patient_id      text NOT NULL,
  billed_amount   numeric(12,2) NOT NULL CHECK (billed_amount >= 0),
  claim_status    text NOT NULL,                  -- from the file's "status" (e.g. DENIED)
  details         jsonb NOT NULL DEFAULT '{}',    -- all other input fields (denial_details, later: code, provider, dates)
  created_at      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, claim_id)
);

-- claim_labels: the answer key. Kept in its own table so the pipeline can never read it.
CREATE TABLE claim_labels (
  tenant_id              uuid NOT NULL,
  claim_id               text NOT NULL,
  ground_truth_clause_id text NOT NULL,
  ground_truth_category  text NOT NULL
    CHECK (ground_truth_category IN ('Exclusion', 'PreAuth', 'Limitation', 'MedicalNecessity')),
  ground_truth_policy_id text,                    -- not in the file yet, will need this once the policies are generated. 
  extra_labels           jsonb NOT NULL DEFAULT '{}',  -- expected extraction values, expected outcome, etc.
  created_at             timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, claim_id),
  FOREIGN KEY (tenant_id, claim_id) REFERENCES claims (tenant_id, claim_id) ON DELETE CASCADE
);

COMMENT ON TABLE claims IS 'Synthetic claim inputs. Safe for the pipeline to read.';
COMMENT ON TABLE claim_labels IS 'Ground-truth answer key for scoring only. app_user has no access.';

-- The app role works with claims (RLS limits which rows) but has no access to the labels.
GRANT SELECT, INSERT, UPDATE, DELETE ON claims TO app_user;
REVOKE ALL ON claim_labels FROM app_user;