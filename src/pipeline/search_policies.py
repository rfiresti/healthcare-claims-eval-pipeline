# This file will find the policy clauses closest to a question. Connects as app_user as always to enforce RLS
import os
import sys

import psycopg
from sentence_transformers import SentenceTransformer

from embedding_settings import MODEL_NAME

tenant_id = sys.argv[1]
question = sys.argv[2]

model = SentenceTransformer(MODEL_NAME)
vector = str(model.encode(question, normalize_embeddings=True).tolist())

conn = psycopg.connect(os.environ["APP_DB_URI"])
conn.execute("SELECT set_config('app.current_tenant_id', %s, true)", (tenant_id,))

rows = conn.execute(
    """
    SELECT clause_id, content, 1 - (embedding <=> %s::vector) AS similarity
    FROM vector_embeddings
    ORDER BY embedding <=> %s::vector
    LIMIT 3
    """,
    (vector, vector),
).fetchall()

if not rows:
    print("No results (wrong tenant, or nothing loaded)")
for clause_id, content, similarity in rows:
    print(f"{similarity:.3f}  {clause_id}  {content[:80]}")