"""Checkpointer isolation through LangGraph's PostgresSaver, run as app_user under RLS.

The checkpoint tables are protected by migration 003: a checkpoint row is visible only when its
thread_id matches a conversations row owned by the tenant in app.current_tenant_id. These tests
drive the real library (put, put_writes, get_tuple, list) as app_user, so they check that the
saver works with those policies, not only that raw SQL is filtered. Run:

    ADMIN_DB_URI=... APP_DB_URI=... pytest -v -m isolation tests/test_checkpointer_postgres.py

ADMIN_DB_URI is used only to check the schema version, count rows and clean up.
"""
import os
import uuid

import psycopg
import pytest
from psycopg.rows import dict_row
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres import PostgresSaver

pytestmark = pytest.mark.isolation

ADMIN_DSN = os.environ["ADMIN_DB_URI"]
APP_DSN = os.environ["APP_DB_URI"]
A = str(uuid.uuid4())
B = str(uuid.uuid4())
TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


def app_conn(tenant):
    """A dedicated app_user connection with the tenant set for the whole session.

    Same settings PostgresSaver.from_conn_string uses. One connection per graph run: a pooled
    connection could carry the previous tenant's setting.
    """
    conn = psycopg.connect(APP_DSN, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    if tenant is not None:
        conn.execute("SELECT set_config('app.current_tenant_id', %s, false)", (tenant,))
    return conn


def cfg(thread_id):
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def write_checkpoint(saver, thread_id, value):
    """One checkpoint with a structured channel value and a pending write.

    The library keeps primitive values inline in checkpoints and stores others in checkpoint_blobs,
    so the draft is a dict to make sure all three tables are written.
    """
    cp = empty_checkpoint()
    version = saver.get_next_version(None, None)
    cp["channel_values"] = {"draft": {"text": value}}
    cp["channel_versions"] = {"draft": version}
    saved = saver.put(cfg(thread_id), cp, {"source": "input", "step": 0}, {"draft": version})
    saver.put_writes(saved, [("draft", value + " (pending)")], task_id="task-1")
    return saved


def admin_count(table, thread_id):
    with psycopg.connect(ADMIN_DSN) as c:
        return c.execute(f"SELECT count(*) FROM {table} WHERE thread_id = %s", (thread_id,)).fetchone()[0]


@pytest.fixture(scope="module")
def threads():
    """Each tenant opens its own conversation as app_user; its id is the LangGraph thread_id."""
    ids = {}
    for tenant in (A, B):
        with app_conn(tenant) as c:
            ids[tenant] = c.execute(
                "INSERT INTO conversations (tenant_id) VALUES (%s) RETURNING id", (tenant,)
            ).fetchone()["id"]
    yield ids
    # Clean up by tenant, not only by these thread ids: if isolation is broken, a test may have
    # created rows it should not have been able to create.
    with psycopg.connect(ADMIN_DSN, autocommit=True) as c:
        mine = "SELECT id FROM conversations WHERE tenant_id = ANY(%s::uuid[])"
        for table in TABLES:
            c.execute(f"DELETE FROM {table} WHERE thread_id IN ({mine})", ([A, B],))
        c.execute("DELETE FROM conversations WHERE tenant_id = ANY(%s::uuid[])", ([A, B],))


@pytest.fixture(scope="module")
def a_saved(threads):
    with app_conn(A) as c:
        return write_checkpoint(PostgresSaver(c), threads[A], "tenant A draft")


def test_schema_matches_installed_library():
    """If the library is newer than the tables, setup() would try to migrate them at startup."""
    with psycopg.connect(ADMIN_DSN) as c:
        applied = c.execute("SELECT max(v) FROM checkpoint_migrations").fetchone()[0]
    assert applied == len(PostgresSaver.MIGRATIONS) - 1


def test_owner_round_trip(threads, a_saved):
    with app_conn(A) as c:
        tup = PostgresSaver(c).get_tuple(cfg(threads[A]))
    assert tup is not None
    assert tup.config["configurable"]["checkpoint_id"] == a_saved["configurable"]["checkpoint_id"]
    assert tup.checkpoint["channel_values"]["draft"] == {"text": "tenant A draft"}
    assert [w[2] for w in tup.pending_writes] == ["tenant A draft (pending)"]
    for table in TABLES:
        assert admin_count(table, threads[A]) >= 1, f"nothing written to {table}"


def test_resume_on_new_connection(threads, a_saved):
    """A fresh connection (as after a process or container restart) finds the same checkpoint."""
    with app_conn(A) as c:
        tup = PostgresSaver(c).get_tuple(cfg(threads[A]))
    assert tup.config["configurable"]["checkpoint_id"] == a_saved["configurable"]["checkpoint_id"]


@pytest.mark.parametrize("tenant", [B, None], ids=["other tenant", "no tenant"])
def test_cannot_read_other_tenants_thread(threads, a_saved, tenant):
    with app_conn(tenant) as c:
        saver = PostgresSaver(c)
        assert saver.get_tuple(cfg(threads[A])) is None
        assert list(saver.list(cfg(threads[A]))) == []
        assert all(t.config["configurable"]["thread_id"] != threads[A] for t in saver.list(None))


@pytest.mark.parametrize("tenant", [B, None], ids=["other tenant", "no tenant"])
def test_cannot_write_to_other_tenants_thread(threads, a_saved, tenant):
    before = {t: admin_count(t, threads[A]) for t in TABLES}
    with app_conn(tenant) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            write_checkpoint(PostgresSaver(c), threads[A], "injected")
    assert {t: admin_count(t, threads[A]) for t in TABLES} == before


def test_cannot_open_conversation_for_other_tenant(threads):
    with app_conn(B) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("INSERT INTO conversations (tenant_id) VALUES (%s)", (A,))


def test_own_thread_still_writable_by_owner_only(threads):
    """Tenant B can checkpoint its own thread; the same thread stays invisible to A."""
    with app_conn(B) as c:
        write_checkpoint(PostgresSaver(c), threads[B], "tenant B draft")
    with app_conn(A) as c:
        assert PostgresSaver(c).get_tuple(cfg(threads[B])) is None













