import json
import os
from pathlib import Path
from urllib.parse import quote_plus
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

def build_database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    host = os.environ["DB_HOST"]
    port = os.getenv("DB_PORT", "5432")
    name = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = quote_plus(os.environ["DB_PASSWORD"])
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


DATABASE_URL = build_database_url()

app = FastAPI(title="ICAE-Cloud VM1 Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory state cache for status polling
# Postgres remains source of truth for RLS
APPEALS = {}


class DocInput(BaseModel):
    doc_type: str  # "denial", "eob", or "policy"
    source_name: str = "session_input.txt"
    content: str


class AppealRequest(BaseModel):
    denial_id: UUID
    eob_id: UUID
    policy_id: UUID


class ApprovalInput(BaseModel):
    approved: bool
    letter: str


def get_tenant(x_tenant_id: str = Header(...)) -> UUID:
    try:
        return UUID(x_tenant_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="X-Tenant-ID must be a valid UUID")


def run_as_tenant(
    tenant_id: UUID,
    query: str,
    params: tuple = (),
    fetch_one: bool = False,
    fetch_all: bool = False,
):
    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('app.current_tenant_id', %s, true);",
                (str(tenant_id),),
            )
            cur.execute(query, params)
            result = None
            if fetch_one:
                result = cur.fetchone()
            elif fetch_all:
                result = cur.fetchall()
        conn.commit()
        return result


@app.get("/health")
def health():
    row = run_as_tenant(
        UUID("00000000-0000-0000-0000-000000000000"),
        "SELECT 1 AS ok;",
        fetch_one=True,
    )
    return {"status": "ok", "db_connected": row["ok"] == 1}


@app.post("/docs")
def create_doc(req: DocInput, x_tenant_id: str = Header(...)):
    tenant_id = get_tenant(x_tenant_id)
    doc_id = uuid4()

    metadata = {"input_mode": "plain_text"}
    try:
        parsed = json.loads(req.content)
        if isinstance(parsed, dict):
            metadata = parsed
    except json.JSONDecodeError:
        pass

    return run_as_tenant(
        tenant_id,
        """
        INSERT INTO documents (id, tenant_id, doc_type, source_name, content, metadata)
        VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        RETURNING id, tenant_id, doc_type, source_name, created_at;
        """,
        (
            str(doc_id),
            str(tenant_id),
            req.doc_type,
            req.source_name,
            req.content,
            json.dumps(metadata),
        ),
        fetch_one=True,
    )


@app.post("/docs/pdf")
async def upload_pdf(
    doc_type: str = Form(...),
    file: UploadFile = File(...),
    x_tenant_id: str = Header(...),
):
    tenant_id = get_tenant(x_tenant_id)
    placeholder_text = f"PDF uploaded (parsing pending): {file.filename}"
    return create_doc(
        DocInput(doc_type=doc_type, source_name=file.filename, content=placeholder_text),
        str(tenant_id),
    )


@app.post("/appeal/start")
def start_appeal(req: AppealRequest, x_tenant_id: str = Header(...)):
    tenant_id = get_tenant(x_tenant_id)
    conversation_id = str(uuid4())

    run_as_tenant(
        tenant_id,
        "INSERT INTO conversations (id, tenant_id) VALUES (%s, %s);",
        (conversation_id, str(tenant_id)),
    )

    docs = run_as_tenant(
        tenant_id,
        "SELECT id, doc_type, content FROM documents WHERE id IN (%s, %s, %s);",
        (str(req.denial_id), str(req.eob_id), str(req.policy_id)),
        fetch_all=True,
    )

    draft = (
        f"Appeal / Conversation ID: {conversation_id}\n"
        f"Session Tenant ID: {tenant_id}\n"
        f"Documents found under this tenant: {len(docs)}\n\n"
        f"Draft Appeal: Please review this claim denial against the policy terms."
    )

    record = {
        "appeal_id": conversation_id,
        "tenant_id": str(tenant_id),
        "status": "awaiting_approval",
        "docs_found": len(docs),
        "draft": draft,
        "cited_clauses": [],       # Teammates can populate ["CLS-1042", ...] later
        "critique_passed": True,   # Teammates can set False if grounding check fails
        "critique_notes": "Placeholder critique check (LangGraph connection pending)",
    }
    APPEALS[conversation_id] = record
    return record


@app.get("/appeal/{appeal_id}")
def get_appeal(appeal_id: str, x_tenant_id: str = Header(...)):
    tenant_id = get_tenant(x_tenant_id)

    conv = run_as_tenant(
        tenant_id,
        "SELECT id FROM conversations WHERE id = %s;",
        (appeal_id,),
        fetch_one=True,
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Appeal not found for this session")

    record = APPEALS.get(appeal_id)
    if not record or record["tenant_id"] != str(tenant_id):
        raise HTTPException(status_code=404, detail="Appeal state not found for this session")

    return record


@app.post("/appeal/{appeal_id}/approve")
def approve_appeal(appeal_id: str, req: ApprovalInput, x_tenant_id: str = Header(...)):
    tenant_id = get_tenant(x_tenant_id)

    conv = run_as_tenant(
        tenant_id,
        "SELECT id FROM conversations WHERE id = %s;",
        (appeal_id,),
        fetch_one=True,
    )
    if not conv:
        raise HTTPException(status_code=404, detail="Appeal not found for this session")

    existing = APPEALS.get(appeal_id, {})
    updated = {
        "appeal_id": appeal_id,
        "tenant_id": str(tenant_id),
        "status": "approved" if req.approved else "rejected",
        "draft": req.letter,
        "cited_clauses": existing.get("cited_clauses", []),
        "critique_passed": existing.get("critique_passed", True),
        "critique_notes": existing.get("critique_notes", ""),
    }
    APPEALS[appeal_id] = updated
    return updated