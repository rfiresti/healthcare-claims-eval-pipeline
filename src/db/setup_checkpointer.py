import os
from langgraph.checkpoint.postgres import PostgresSaver

# run as the admin_owner role with permission to create tables
with PostgresSaver.from_conn_string(os.environ["ADMIN_DB_URI"]) as checkpointer:
    checkpointer.setup()
    print("checkpointer setup complete")