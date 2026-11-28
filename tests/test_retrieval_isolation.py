"""Tenant-isolation tests for retrieval and row-level security (Week 4). Run with:

    ADMIN_DB_URI=... APP_DB_URI=... pytest -v -m isolation --junitxml=eval/isolation_report.xml

ADMIN_DB_URI (the table owner, admin_owner) is used only to seed and clean up test data and to read ground
truth. APP_DB_URI (app_user, the role the application uses) is used for everything under test.

Two dedicated test tenants, A and B, each get 5 policy chunks carrying a canary marker, plus a conversation
with one row in each checkpoint table. They are separate from the evaluation data. Tenant IDs and markers can
be fixed with TEST_TENANT_A / TEST_TENANT_B / TEST_MARKER_A / TEST_MARKER_B (Week 7 stress tenants).

Section 1 goes through retrieve(); section 2 is raw SQL (the part that may belong to whoever owns the policies).
The checkpoint tables have no tenant_id column: they are tested through thread_id, which the policies tie to
conversations.tenant_id.

Register the marker in pytest.ini to silence the warning:  markers = isolation: tenant isolation tests
"""
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pipeline"))
import retrieve                                   # noqa: E402
from embedding_settings import MODEL_NAME         # noqa: E402

pytestmark = pytest.mark.isolation

ADMIN_DSN = os.environ["ADMIN_DB_URI"]
APP_DSN = os.environ["APP_DB_URI"]

A = os.environ.get("TEST_TENANT_A") or str(uuid.uuid4())
B = os.environ.get("TEST_TENANT_B") or str(uuid.uuid4())
OTHER = {A: B, B: A}
MARK = {A: os.environ.get("TEST_MARKER_A", "TENANT-A-CANARY-9f3k"),
        B: os.environ.get("TEST_MARKER_B", "TENANT-B-CANARY-2m8x")}
DOC = {A: str(uuid.uuid4()), B: str(uuid.uuid4())}
THREAD = {A: "isolation-thread-" + A, B: "isolation-thread-" + B}
N_CHUNKS = 5

TENANT_TABLES = ["documents", "vector_embeddings", "conversations"]
THREAD_TABLES = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]
ALL_TABLES = TENANT_TABLES + THREAD_TABLES


def admin():
    return psycopg.connect(ADMIN_DSN)


def app():
    return psycopg.connect(APP_DSN)


def as_tenant(conn, tenant):
    """Open cursor with the tenant set for the current transaction (call inside conn.transaction())."""
    cur = conn.cursor()
    cur.execute("SELECT set_config('app.current_tenant_id', %s, true)", (tenant,))
    return cur


def _cleanup(adm):
    for t in (A, B):
        for table in THREAD_TABLES:
            adm.execute(f"DELETE FROM {table} WHERE thread_id = %s", (THREAD[t],))
        adm.execute("DELETE FROM conversations WHERE tenant_id = %s", (t,))
        adm.execute("DELETE FROM documents WHERE tenant_id = %s", (t,))     # embeddings go with them (cascade)


@pytest.fixture(scope="session", autouse=True)
def seed():
    with admin() as adm:
        _cleanup(adm)
        for t in (A, B):
            adm.execute(
                "INSERT INTO documents (id, tenant_id, doc_type, source_name, content) VALUES (%s, %s, 'policy', %s, %s)",
                (DOC[t], t, "isolation-test-" + t, "isolation test policy " + MARK[t]))
            for i in range(N_CHUNKS):
                text = f"Clause {i}: coverage exclusion {MARK[t]}"
                adm.execute(
                    "INSERT INTO vector_embeddings (tenant_id, document_id, clause_id, chunk_index, content, embedding, model_name)"
                    " VALUES (%s, %s, %s, %s, %s, %s::vector, %s)",
                    (t, DOC[t], f"ISO-{t[:8]}-{i}", 0, text, str(retrieve.embed_query(text)), MODEL_NAME))
            adm.execute("INSERT INTO conversations (id, tenant_id) VALUES (%s, %s)", (THREAD[t], t))
            adm.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)"
                " VALUES (%s, '', 'c1', '{}'::jsonb, %s::jsonb)", (THREAD[t], '{"marker": "%s"}' % MARK[t]))
            adm.execute(
                "INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)"
                " VALUES (%s, '', 'ch', '1', 'json', %s)", (THREAD[t], MARK[t].encode()))
            adm.execute(
                "INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob)"
                " VALUES (%s, '', 'c1', 't1', 0, 'ch', 'json', %s)", (THREAD[t], MARK[t].encode()))
    yield
    with admin() as adm:
        _cleanup(adm)


def admin_count(sql, params=()):
    with admin() as adm:
        return adm.execute(sql, params).fetchone()[0]


# ---------------------------------------------------------------- 1. through retrieve()

@pytest.mark.parametrize("me", [A, B])
@pytest.mark.parametrize("k", [5, 50])
def test_retrieve_never_returns_other_tenant(me, k):
    with app() as conn:
        res = retrieve.retrieve(conn, me, MARK[OTHER[me]] + " coverage exclusion", k=k)
    assert len(res) == N_CHUNKS, "expected exactly the caller's own %d chunks, got %d" % (N_CHUNKS, len(res))
    assert all(MARK[OTHER[me]] not in r["text"] for r in res)
    assert all(MARK[me] in r["text"] for r in res)


@pytest.mark.parametrize("bad", [None, "", "not-a-uuid"])
def test_retrieve_rejects_missing_or_malformed_tenant(bad):
    with app() as conn, pytest.raises(ValueError):
        retrieve.retrieve(conn, bad, "coverage", k=5)


def test_tenant_does_not_leak_across_transactions():
    with app() as conn:
        retrieve.retrieve(conn, A, "coverage exclusion", k=5)          # its transaction has ended
        with conn.transaction(), conn.cursor() as cur:                  # no tenant set now
            cur.execute("SELECT count(*) FROM vector_embeddings")
            assert cur.fetchone()[0] == 0


def _hammer(me, n=100):
    with app() as conn:
        for _ in range(n):
            res = retrieve.retrieve(conn, me, "coverage exclusion", k=5)
            assert res and all(MARK[OTHER[me]] not in r["text"] for r in res)
    return True


def test_concurrent_tenants():
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(_hammer, A) for _ in range(4)] + [ex.submit(_hammer, B) for _ in range(4)]
        assert all(f.result() for f in futs)


# ---------------------------------------------------------------- 2. raw SQL

@pytest.mark.parametrize("table", ALL_TABLES)
def test_no_tenant_means_no_rows(table):
    with app() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        assert cur.fetchone()[0] == 0


@pytest.mark.parametrize("table", TENANT_TABLES)
def test_raw_select_sees_only_own_rows(table):
    with app() as conn, conn.transaction():
        cur = as_tenant(conn, A)
        cur.execute(f"SELECT count(*) FROM {table} WHERE tenant_id <> %s", (A,))
        assert cur.fetchone()[0] == 0
        cur.execute(f"SELECT count(*) FROM {table}")
        mine = cur.fetchone()[0]
    assert mine > 0 and mine == admin_count(f"SELECT count(*) FROM {table} WHERE tenant_id = %s", (A,))


@pytest.mark.parametrize("table", THREAD_TABLES)
def test_checkpoint_tables_see_only_own_threads(table):
    with app() as conn, conn.transaction():
        cur = as_tenant(conn, A)
        cur.execute(f"SELECT count(*) FROM {table} WHERE thread_id = %s", (THREAD[B],))
        assert cur.fetchone()[0] == 0
        cur.execute(f"SELECT count(*) FROM {table}")
        mine = cur.fetchone()[0]
    assert mine > 0 and mine == admin_count(f"SELECT count(*) FROM {table} WHERE thread_id = %s", (THREAD[A],))


WRITES_AS_A = [
    ("documents", "INSERT INTO documents (tenant_id, doc_type, source_name, content) VALUES (%(b)s, 'policy', 'x', 'x')"),
    ("vector_embeddings", "INSERT INTO vector_embeddings (tenant_id, document_id, clause_id, chunk_index, content, embedding, model_name)"
                          " VALUES (%(b)s, %(bdoc)s, 'x', 98, 'x', %(vec)s::vector, 'x')"),
    ("conversations", "INSERT INTO conversations (id, tenant_id) VALUES ('isolation-x', %(b)s)"),
    ("checkpoints on B's thread", "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint) VALUES (%(bthread)s, '', 'x', '{}'::jsonb)"),
    ("checkpoints on no thread", "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint) VALUES ('isolation-no-such-thread', '', 'x', '{}'::jsonb)"),
    ("checkpoint_blobs on B's thread", "INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type) VALUES (%(bthread)s, '', 'x', 'x', 'x')"),
    ("checkpoint_writes on B's thread", "INSERT INTO checkpoint_writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, blob) VALUES (%(bthread)s, '', 'x', 'x', 0, 'x', 'x')"),
]


@pytest.mark.parametrize("name,sql", WRITES_AS_A, ids=[w[0] for w in WRITES_AS_A])
def test_cannot_insert_for_other_tenant(name, sql):
    params = {"b": B, "bdoc": DOC[B], "bthread": THREAD[B], "vec": "[" + ",".join(["1"] + ["0"] * 383) + "]"}
    with app() as conn, pytest.raises(psycopg.errors.InsufficientPrivilege) as e:
        with conn.transaction():
            cur = as_tenant(conn, A)
            cur.execute(sql, params)
    assert "row-level security" in str(e.value)                  # fails for the right reason, not a missing grant


def test_embedding_cannot_point_at_other_tenants_document():
    vec = "[" + ",".join(["1"] + ["0"] * 383) + "]"
    with app() as conn, pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction():
            cur = as_tenant(conn, A)
            cur.execute(
                "INSERT INTO vector_embeddings (tenant_id, document_id, clause_id, chunk_index, content, embedding, model_name)"
                " VALUES (%s, %s, 'x', 97, 'x', %s::vector, 'x')", (A, DOC[B], vec))


MODIFY = [
    ("documents", "tenant_id", lambda: B, "content = 'x'"),
    ("vector_embeddings", "tenant_id", lambda: B, "content = 'x'"),
    ("conversations", "tenant_id", lambda: B, "created_at = now()"),
    ("checkpoints", "thread_id", lambda: THREAD[B], "type = 'x'"),
    ("checkpoint_blobs", "thread_id", lambda: THREAD[B], "type = 'x'"),
    ("checkpoint_writes", "thread_id", lambda: THREAD[B], "type = 'x'"),
]


@pytest.mark.parametrize("table,col,val,setc", MODIFY, ids=[m[0] for m in MODIFY])
def test_cannot_update_or_delete_other_tenants_rows(table, col, val, setc):
    before = admin_count(f"SELECT count(*) FROM {table} WHERE {col} = %s", (val(),))
    with app() as conn, conn.transaction():
        cur = as_tenant(conn, A)
        cur.execute(f"UPDATE {table} SET {setc} WHERE {col} = %s", (val(),))
        assert cur.rowcount == 0
        cur.execute(f"DELETE FROM {table} WHERE {col} = %s", (val(),))
        assert cur.rowcount == 0
    assert before > 0 and admin_count(f"SELECT count(*) FROM {table} WHERE {col} = %s", (val(),)) == before


@pytest.mark.parametrize("table", ALL_TABLES)
def test_app_role_cannot_truncate(table):
    # TRUNCATE ignores RLS, so only the missing privilege stops it. Checked in the catalog, never executed.
    with app() as conn:
        ok = conn.execute("SELECT has_table_privilege(current_user, %s, 'TRUNCATE')", (table,)).fetchone()[0]
    assert ok is False

