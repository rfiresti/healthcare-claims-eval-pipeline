"""Load the synthetic claims into Postgres, split into two tables:
claims (the inputs) and claim_labels (the answer key).

Run as admin_owner. 
"""
import argparse
import json
import os
import uuid
import zlib
from collections import Counter
from decimal import Decimal

import psycopg

DEFAULT_FILE = "data/synthetic_output/labeled_claims.json"
DEFAULT_TENANTS = "00000000-0000-0000-0000-000000000001,00000000-0000-0000-0000-000000000002"
CATEGORIES = {"Exclusion", "PreAuth", "Limitation", "MedicalNecessity"}

# These input fields get their own column. Any other input field goes into claims.details.
OWN_COLUMNS = {"claim_id", "patient_id", "billed_amount", "status", "tenant_id"}
# These label fields get their own column. Any other ground_truth_* field goes into extra_labels.
LABEL_COLUMNS = {"ground_truth_clause_id", "ground_truth_category", "ground_truth_policy_id"}
REQUIRED = ["claim_id", "patient_id", "billed_amount", "status",
            "ground_truth_clause_id", "ground_truth_category"]

UPSERT_CLAIM = """
INSERT INTO claims (tenant_id, claim_id, dataset_version, patient_id, billed_amount, claim_status, details)
VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT (tenant_id, claim_id) DO UPDATE SET
  dataset_version = EXCLUDED.dataset_version,
  patient_id      = EXCLUDED.patient_id,
  billed_amount   = EXCLUDED.billed_amount,
  claim_status    = EXCLUDED.claim_status,
  details         = EXCLUDED.details
"""

UPSERT_LABEL = """
INSERT INTO claim_labels (tenant_id, claim_id, ground_truth_clause_id, ground_truth_category,
                          ground_truth_policy_id, extra_labels)
VALUES (%s, %s, %s, %s, %s, %s::jsonb)
ON CONFLICT (tenant_id, claim_id) DO UPDATE SET
  ground_truth_clause_id = EXCLUDED.ground_truth_clause_id,
  ground_truth_category  = EXCLUDED.ground_truth_category,
  ground_truth_policy_id = EXCLUDED.ground_truth_policy_id,
  extra_labels           = EXCLUDED.extra_labels
"""


def check_records(records):
    """Return a list of problems. An empty list means the file is good to load."""
    problems = []
    seen = set()
    for number, record in enumerate(records, start=1):
        name = record.get("claim_id", f"record #{number}")

        for key in REQUIRED:
            if key not in record:
                problems.append(f"{name}: missing {key}")

        if name in seen:
            problems.append(f"{name}: duplicate claim_id")
        seen.add(name)

        if "ground_truth_category" in record and record["ground_truth_category"] not in CATEGORIES:
            problems.append(f"{name}: unknown category {record['ground_truth_category']!r}")

        if "billed_amount" in record:
            amount = record["billed_amount"]
            if not isinstance(amount, (int, float)) or amount < 0:
                problems.append(f"{name}: billed_amount must be a number >= 0")

        if record.get("tenant_id"):
            try:
                uuid.UUID(str(record["tenant_id"]))
            except ValueError:
                problems.append(f"{name}: tenant_id is not a valid UUID")
    return problems


def pick_tenant(claim_id, tenant_ids):
    """The same claim_id always gets the same tenant, no matter the order of the file."""
    return tenant_ids[zlib.crc32(claim_id.encode()) % len(tenant_ids)]


def split_record(record, dataset_version, tenant_ids):
    """Turn one record into a row for claims and a row for claim_labels."""
    claim_id = record["claim_id"]
    tenant_id = record.get("tenant_id") or pick_tenant(claim_id, tenant_ids)

    details = {key: value for key, value in record.items()
               if key not in OWN_COLUMNS and not key.startswith("ground_truth_")}
    extra_labels = {key[len("ground_truth_"):]: value for key, value in record.items()
                    if key.startswith("ground_truth_") and key not in LABEL_COLUMNS}

    claim_row = (
        tenant_id, claim_id, dataset_version, record["patient_id"],
        Decimal(str(record["billed_amount"])),   # exact money value, not a float
        record["status"], json.dumps(details),
    )
    label_row = (
        tenant_id, claim_id, record["ground_truth_clause_id"], record["ground_truth_category"],
        record.get("ground_truth_policy_id"), json.dumps(extra_labels),
    )
    return claim_row, label_row


parser = argparse.ArgumentParser(description="Load synthetic claims into the claims and claim_labels tables.")
parser.add_argument("file", nargs="?", default=DEFAULT_FILE)
parser.add_argument("--tenant-ids", default=DEFAULT_TENANTS,
                    help="comma-separated tenant UUIDs to spread the claims across")
parser.add_argument("--dataset-version", default="v1")
parser.add_argument("--replace", action="store_true",
                    help="first delete every existing claim (and its label) with this dataset version")
parser.add_argument("--dry-run", action="store_true", help="check the file and print a summary, write nothing")
args = parser.parse_args()

tenant_ids = [str(uuid.UUID(t.strip())) for t in args.tenant_ids.split(",")]

with open(args.file) as f:
    records = json.load(f)
if not isinstance(records, list):
    raise SystemExit("Expected the file to hold a list of claim records.")

problems = check_records(records)
if problems:
    print(f"Found {len(problems)} problem(s), nothing was loaded. First few:")
    for problem in problems[:10]:
        print("  -", problem)
    raise SystemExit(1)

rows = [split_record(record, args.dataset_version, tenant_ids) for record in records]
claims = [claim for claim, _ in rows]
labels = [label for _, label in rows]

# ---- Summary ----
print(f"{len(claims)} claims, dataset version {args.dataset_version!r}")
for tenant_id, count in sorted(Counter(claim[0] for claim in claims).items()):
    print(f"  tenant {tenant_id}: {count} claims")
clause_counts = Counter(label[2] for label in labels)
print(f"{len(clause_counts)} distinct ground-truth clauses. Most common: {clause_counts.most_common(5)}")

# Does the claim's own input text give away its answer? (claim[6] is the details JSON text)
leaks = sum(1 for claim, label in rows if label[2] in claim[6])
if leaks:
    print(f"WARNING: {leaks} claims mention their own answer clause ID in the input details. "
          "Don't use details as a retrieval query.")

if args.dry_run:
    print("\nSample claim row:", claims[0])
    print("Sample label row:", labels[0])
    print("\nDry run, nothing written.")
    raise SystemExit(0)

# ---- Write to the database ----
uri = os.environ.get("ADMIN_DB_URI")
if not uri:
    raise SystemExit("Set ADMIN_DB_URI first (the admin_owner connection string).")

conn = psycopg.connect(uri)
with conn.cursor() as cur:
    if args.replace:
        cur.execute("DELETE FROM claims WHERE dataset_version = %s", (args.dataset_version,))
        print(f"Removed {cur.rowcount} existing claims for this version")
    cur.executemany(UPSERT_CLAIM, claims)    # claims first, labels point at them
    cur.executemany(UPSERT_LABEL, labels)
conn.commit()
print("Done")