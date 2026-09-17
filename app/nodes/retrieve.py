"""Node 1: retrieve top-k chunks from the thread's FAISS index (if any).

Purely deterministic -- no LLM call. Also resets all per-turn transient
state fields so nothing leaks over from a previous turn via the
checkpointer (scalar state fields are otherwise carried forward unless a
node explicitly overwrites them).
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig

from app.state import GraphState
from app.utils.heuristics import is_simple_chitchat
from app.utils.ingestion import get_retriever

RESET_FIELDS = {
    "documents": [],
    "doc_relevant": False,
    "rewritten_query": None,
    "used_web": False,
    "web_results": [],
    "answer": "",
    "sources": [],
    "supported": True,
    "regenerated": False,
    "skip_retrieval": False,
}


def _thread_id(config: Optional[RunnableConfig]) -> Optional[str]:
    if not config:
        return None
    return config.get("configurable", {}).get("thread_id")


def retrieve_node(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    update: Dict[str, Any] = dict(RESET_FIELDS)

    thread_id = _thread_id(config)
    question = state.get("question", "")
    retriever = get_retriever(thread_id)

    if is_simple_chitchat(question):
        # Deterministic (regex) short-circuit: no retrieval, no grading, no
        # web search, no verification needed for small talk.
        update["skip_retrieval"] = True
        update["documents"] = []
        return update

    if retriever is None or not question:
        update["documents"] = []
        return update

    try:
        docs = retriever.invoke(question)
    except Exception:
        # FAISS/embedding failure: fail gracefully, fall back to web search.
        update["documents"] = []
        return update

    update["documents"] = [
        {"content": doc.page_content[:1000], "source": doc.metadata.get("source", "PDF")}
        for doc in docs
    ]
    return update
