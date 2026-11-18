# Embed policy clauses and load them into Postgres. Always connect as app_user, so RLS applies
import argparse
import json
import os
import uuid

import psycopg
from sentence_transformers import SentenceTransformer

from chunk_policies import load_policies, chunk_policy
from embedding_settings import MODEL_NAME, EXPECTED_DIMENSION

parser = argparse.ArgumentParser()
parser.add_argument("policy_file", help="JSON file with one policy or a list of policies")
parser.add_argument("--tenant-id", required=True, help="UUID of the tenant that owns these policies")
args = parser.parse_args()

policies = load_policies(args.policy_file)

model = SentenceTransformer(MODEL_NAME)
if model.get_embedding_dimension() != EXPECTED_DIMENSION:
    raise SystemExit(
        f"Model makes {model.get_embedding_dimension()}-number vectors but the "
        f"database column holds {EXPECTED_DIMENSION}."
    )

conn = psycopg.connect(os.environ["APP_DB_URI"])
# tell the database which tenant this is.
conn.execute("SELECT set_config('app.current_tenant_id', %s, true)", (args.tenant_id,))

for policy in policies:
    chunks = chunk_policy(policy)

    # Same tenant + same policy always gives the same id, so reruns update instead of duplicating.
    document_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{args.tenant_id}:{policy['policy_id']}"))

    full_text = "\n\n".join(f"{c['clause_id']}: {c['clause_text']}" for c in policy["clauses"])
    info = {
        "policy_id": policy["policy_id"],
        "payer_name": policy["payer_name"],
        "effective_date": policy["effective_date"],
    }

    conn.execute(
        """
        INSERT INTO documents (id, tenant_id, doc_type, source_name, content, metadata)
        VALUES (%s, %s, 'policy', %s, %s, %s::jsonb)
        ON CONFLICT (id) DO UPDATE
        SET content = EXCLUDED.content, metadata = EXCLUDED.metadata
        """,
        (document_id, args.tenant_id, policy["policy_id"], full_text, json.dumps(info)),
    )

    # delete any old chunks for this policy, then insert fresh ones. Don't want to duplicate here so need to delete first.
    conn.execute("DELETE FROM vector_embeddings WHERE document_id = %s", (document_id,))

    # turn every chunk's text into a list of 384 numbers
    vectors = model.encode([c["text"] for c in chunks], batch_size=32, normalize_embeddings=True)

    for chunk, vector in zip(chunks, vectors):
        conn.execute(
            """
            INSERT INTO vector_embeddings
              (tenant_id, document_id, clause_id, chunk_index, content, embedding, model_name)
            VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
            """,
            (args.tenant_id, document_id, chunk["clause_id"], chunk["chunk_index"],
             chunk["text"], str(vector.tolist()), MODEL_NAME),
        )

    print(f"{policy['policy_id']}: loaded {len(chunks)} chunks")

conn.commit()
print("Done")