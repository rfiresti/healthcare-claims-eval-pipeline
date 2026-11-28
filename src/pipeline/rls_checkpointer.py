import os
import uuid

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row

from appeal_graph import build_appeal_graph
from retrieve import check_tenant


def get_app_db_uri() -> str:
    uri = os.environ.get("APP_DB_URI")
    if not uri:
        raise RuntimeError("APP_DB_URI is not set in environment.")
    return uri


def set_checkpoint_tenant(checkpoint_conn, tenant_id: str) -> str:
    tenant = check_tenant(tenant_id)
    checkpoint_conn.execute(
        "SELECT set_config('app.current_tenant_id', %s, false)",
        (tenant,),
    )
    return tenant


def ensure_conversation_thread(
    checkpoint_conn,
    tenant_id: str,
    thread_id: str | None = None,
) -> str:
    tenant = set_checkpoint_tenant(checkpoint_conn, tenant_id)
    if thread_id:
        tid = str(uuid.UUID(str(thread_id)))
        checkpoint_conn.execute(
            "INSERT INTO conversations (id, tenant_id) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
            (tid, tenant),
        )
        return tid
    row = checkpoint_conn.execute(
        "INSERT INTO conversations (tenant_id) VALUES (%s) RETURNING id",
        (tenant,),
    ).fetchone()
    return str(row["id"])


def create_rls_graph(
    tenant_id: str,
    app_db_uri: str | None = None,
    run_setup: bool = False,
):
    uri = app_db_uri or get_app_db_uri()

    # Dedicated autocommit connection with prepare_threshold=0 and dict_row for PostgresSaver (3.2.1)
    checkpoint_conn = psycopg.connect(
        uri,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    )
    set_checkpoint_tenant(checkpoint_conn, tenant_id)
    saver = PostgresSaver(checkpoint_conn)
    if run_setup:
        saver.setup()

    # Separate idle connection for transaction-scoped RLS tenant context in retrieve()
    retrieve_conn = psycopg.connect(uri, prepare_threshold=0)
    graph = build_appeal_graph(db_conn=retrieve_conn, checkpointer=saver)
    return graph, checkpoint_conn, retrieve_conn
