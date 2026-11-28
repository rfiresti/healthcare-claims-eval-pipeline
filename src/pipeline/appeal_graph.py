import threading
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from draft_critique import MAX_REVISIONS, critique_draft, draft_appeal
from extract import extract
from retrieve import retrieve

LLM_SEMAPHORE = threading.BoundedSemaphore(4)


class AppealState(TypedDict, total=False):
    tenant_id: str
    claim: dict
    extracted: dict
    denial_category: str
    retrieved_clauses: list[dict]
    draft_text: str
    revision_count: int
    critique_feedback: str
    grounding_status: str
    human_decision: dict
    submission_status: str


def _call_llm_bounded(fn, *args, **kwargs):
    with LLM_SEMAPHORE:
        return fn(*args, **kwargs)


def ingest_node(state: AppealState) -> AppealState:
    claim = state["claim"]
    return {
        "claim": claim,
        "revision_count": state.get("revision_count", 0),
        "grounding_status": "pending",
    }


def classify_node(state: AppealState) -> AppealState:
    extracted = _call_llm_bounded(extract, state["claim"])
    return {
        "extracted": extracted,
        "denial_category": extracted["denial_category"],
    }


def make_retrieve_node(db_conn, k: int = 5):
    def retrieve_node(state: AppealState) -> AppealState:
        claim = state["claim"]
        extracted = state.get("extracted", {})
        query = (
            f"{state.get('denial_category', '')} "
            f"{extracted.get('denial_code') or ''} "
            f"{claim.get('denial_details') or ''}"
        ).strip()
        hits = retrieve(db_conn, state["tenant_id"], query, k=k)
        return {"retrieved_clauses": hits}

    return retrieve_node


def draft_node(state: AppealState) -> AppealState:
    rev = state.get("revision_count", 0)
    text = _call_llm_bounded(
        draft_appeal,
        state["claim"],
        state.get("extracted", {}),
        state.get("retrieved_clauses", []),
        state.get("critique_feedback") if rev > 0 else None,
    )
    next_rev = rev + 1 if state.get("critique_feedback") else rev
    return {
        "draft_text": text,
        "revision_count": next_rev,
    }


def critique_node(state: AppealState) -> AppealState:
    result = critique_draft(
        state.get("draft_text", ""),
        state.get("retrieved_clauses", []),
        state.get("revision_count", 0),
    )
    return {
        "grounding_status": result["grounding_status"],
        "critique_feedback": result["critique_feedback"],
    }


def route_after_critique(state: AppealState) -> str:
    status = state.get("grounding_status")
    if status == "needs_revision" and state.get("revision_count", 0) <= MAX_REVISIONS:
        return "draft"
    return "interrupt_review"


def interrupt_review_node(state: AppealState) -> AppealState:
    decision = interrupt(
        {
            "claim_id": state["claim"].get("claim_id"),
            "denial_category": state.get("denial_category"),
            "grounding_status": state.get("grounding_status"),
            "critique_feedback": state.get("critique_feedback"),
            "draft_text": state.get("draft_text"),
        }
    )
    updated_draft = (
        decision.get("draft_text", state.get("draft_text"))
        if isinstance(decision, dict)
        else state.get("draft_text")
    )
    return {
        "human_decision": decision if isinstance(decision, dict) else {"approved": bool(decision)},
        "draft_text": updated_draft,
    }


def submit_node(state: AppealState) -> AppealState:
    decision = state.get("human_decision", {})
    if decision.get("approved") is False:
        return {"submission_status": "rejected_by_reviewer"}
    return {"submission_status": "submitted"}


def build_appeal_graph(db_conn=None, checkpointer=None):
    graph = StateGraph(AppealState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", make_retrieve_node(db_conn))
    graph.add_node("draft", draft_node)
    graph.add_node("critique", critique_node)
    graph.add_node("interrupt_review", interrupt_review_node)
    graph.add_node("submit", submit_node)

    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "draft": "draft",
            "interrupt_review": "interrupt_review",
        },
    )
    graph.add_edge("interrupt_review", "submit")
    graph.add_edge("submit", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
