# to run this test: 
#export APP_DB_URI='postgresql://app_user:<app_user_password>@10.4.36.107:5432/claims_db'
#python tests/test_checkpoint_tenancy.py

import os
import uuid
import psycopg
from psycopg.rows import dict_row
from typing import TypedDict
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import StateGraph, START, END

# Connection string for app_user (not admin_owner)
DB_URI = os.environ["APP_DB_URI"]

TENANT_A = str(uuid.uuid4())
TENANT_B = str(uuid.uuid4())


# A tiny LangGraph workflow: it just adds 1 to a number
class State(TypedDict):
    count: int


def add_one(state):
    return {"count": state["count"] + 1}


builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_edge(START, "add_one")
builder.add_edge("add_one", END)


def connect(tenant_id=None):
    """Open a connection and tell the database which tenant we are."""
    conn = psycopg.connect(DB_URI, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    if tenant_id:
        conn.execute("SELECT set_config('app.current_tenant_id', %s, false)", (tenant_id,))
    return conn


def count_rows(conn, table):
    return conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]


def run_graph(conn, thread_id):
    """Run the workflow on a thread. Returns 'allowed' or 'blocked'."""
    graph = builder.compile(checkpointer=PostgresSaver(conn))
    try:
        graph.invoke({"count": 0}, {"configurable": {"thread_id": thread_id}})
        return "allowed"
    except Exception:
        return "blocked"


# Tenant A creates a conversation and runs the workflow on it
conn_a = connect(TENANT_A)
thread_a = conn_a.execute(
    "INSERT INTO conversations (tenant_id) VALUES (%s) RETURNING id", (TENANT_A,)
).fetchone()["id"]

print("Test 1: A runs the workflow on its own conversation")
print("  expected: allowed, got:", run_graph(conn_a, thread_a))

print("Test 2: A can see its own checkpoints")
print("  expected: more than 0, got:", count_rows(conn_a, "checkpoints"))

print("Test 3: A tries a thread that has no conversation")
print("  expected: blocked, got:", run_graph(conn_a, str(uuid.uuid4())))

# Tenant B should see nothing of A's
conn_b = connect(TENANT_B)

print("Test 4: B can see A's data")
print("  expected: 0 checkpoints, got:", count_rows(conn_b, "checkpoints"))
print("  expected: 0 conversations, got:", count_rows(conn_b, "conversations"))

print("Test 5: B tries to run the workflow on A's thread")
print("  expected: blocked, got:", run_graph(conn_b, thread_a))

# No tenant set: nothing should be visible
conn_none = connect()
print("Test 6: no tenant set")
print("  expected: 0 checkpoints, got:", count_rows(conn_none, "checkpoints"))
print("  expected: 0 conversations, got:", count_rows(conn_none, "conversations"))

# Clean up (checkpoints first, then the conversation)
PostgresSaver(conn_a).delete_thread(thread_a)
conn_a.execute("DELETE FROM conversations WHERE id = %s", (thread_a,))
print("Cleanup done")

conn_a.close()
conn_b.close()
conn_none.close()