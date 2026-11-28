"""Hybrid retrieval over policy clause chunks, isolated by tenant through Postgres row-level security.

    from retrieve import retrieve
    hits = retrieve(conn, tenant_id, query, k=5)
    # -> [{"clause_id": ..., "text": ..., "score": ...}, ...]

Vector search (pgvector, cosine) and keyword search (full text, on content_tsv from migration 005)
each return their top candidates; the two ranked lists are merged with Reciprocal Rank Fusion (RRF).

Isolation comes only from RLS: there is no WHERE tenant_id in either query, on purpose, so a broken
policy shows up in the isolation tests instead of being hidden by application code. Connect as
app_user (never a superuser or the table owner, who bypass RLS).
"""
import functools
import re
import uuid

import psycopg

from embedding_settings import MODEL_NAME

RRF_K = 60

VECTOR_SQL = """
    SELECT id, clause_id, content
    FROM vector_embeddings
    WHERE model_name = %s
    ORDER BY embedding <=> %s::vector
    LIMIT %s
"""

KEYWORD_SQL = """
    SELECT id, clause_id, content
    FROM vector_embeddings
    WHERE model_name = %s AND content_tsv @@ to_tsquery('english', %s)
    ORDER BY ts_rank_cd(content_tsv, to_tsquery('english', %s)) DESC, id
    LIMIT %s
"""


@functools.lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer   # imported here so this module loads without it
    return SentenceTransformer(MODEL_NAME)


def embed_query(text):
    """Same model and normalization as embed_and_load.py and search_policies.py. Loaded once per process."""
    return _model().encode(text, normalize_embeddings=True).tolist()


def check_tenant(tenant_id):
    """Return tenant_id as a UUID string, or raise ValueError (None, empty and malformed values all raise)."""
    return str(uuid.UUID(str(tenant_id)))


def fuse(vec_rows, kw_rows, k, w_vec=1.0, w_kw=1.0):
    """Reciprocal Rank Fusion. Each row is (id, clause_id, content), best first. Returns the top k."""
    scores, rows = {}, {}
    for weight, lst in ((w_vec, vec_rows), (w_kw, kw_rows)):
        for rank, (id_, clause_id, content) in enumerate(lst, start=1):
            scores[id_] = scores.get(id_, 0.0) + weight / (RRF_K + rank)
            rows[id_] = (clause_id, content)
    top = sorted(scores, key=lambda i: (-scores[i], str(i)))[:k]      # ties broken by id, so results are repeatable
    return [{"clause_id": rows[i][0], "text": rows[i][1], "score": scores[i]} for i in top]


def retrieve(conn, tenant_id, query, k=5, n_candidates=20, w_vec=1.0, w_kw=1.0, ef_search=100):
    """Top-k clause chunks for `query`, as the given tenant.

    `conn` must be a psycopg 3 connection with no transaction open: the tenant setting is local to this
    function's own transaction, so it cannot outlive the call (the pooled-connection leak the isolation
    tests check for). Raises ValueError for a missing or malformed tenant_id before touching the database.
    """
    tenant = check_tenant(tenant_id)
    if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
        raise RuntimeError("retrieve() needs a connection with no open transaction")

    q_emb = str(embed_query(query))
    terms = " | ".join(re.findall(r"[^\W_]+", query.lower()))        # letters and digits only, ORed together

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SELECT set_config('app.current_tenant_id', %s, true)", (tenant,))
        # HNSW fetches nearest candidates first and RLS filters afterwards, which can leave fewer than
        # n_candidates rows. A larger candidate pool and iterative scans (pgvector 0.8+) fix that.
        # strict_order, not relaxed_order: RRF uses ranks, so the vector list must be in exact order.
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
        cur.execute("SELECT set_config('hnsw.iterative_scan', 'strict_order', true)")
        cur.execute(VECTOR_SQL, (MODEL_NAME, q_emb, n_candidates))
        vec_rows = cur.fetchall()
        kw_rows = []
        if terms:
            cur.execute(KEYWORD_SQL, (MODEL_NAME, terms, terms, n_candidates))
            kw_rows = cur.fetchall()
    return fuse(vec_rows, kw_rows, k, w_vec, w_kw)

