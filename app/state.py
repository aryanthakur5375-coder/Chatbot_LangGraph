"""Shared state definition for the CRAG / Self-RAG LangGraph workflow.

Kept intentionally minimal: only the current turn's question, retrieved
context and answer are tracked as scalar fields (overwritten each turn).
Only ``messages`` accumulates across turns (via the ``add_messages``
reducer) so the Streamlit sidebar can show conversation history and
LangGraph's SQLite checkpointer can persist multi-turn threads.

Full retrieved document text is never appended to ``messages``, so history
storage stays small.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RelevanceGrade(BaseModel):
    """Structured output for the document-grading node."""

    relevant: bool = Field(
        description="True if the documents help answer the question."
    )


class SupportGrade(BaseModel):
    """Structured output for the Self-RAG verification node."""

    supported: bool = Field(
        description="True if the answer is grounded in the given context."
    )


class GraphState(TypedDict, total=False):
    # ----------------------------------------------------------------
    # Conversation memory
    # ----------------------------------------------------------------

    # Persisted by the SQLite LangGraph checkpointer.
    messages: Annotated[
        List[BaseMessage],
        add_messages,
    ]

    # ----------------------------------------------------------------
    # Current turn input
    # ----------------------------------------------------------------

    question: str

    # True when a PDF has been successfully indexed for this thread.
    #
    # This is used by app.graph to decide:
    #
    #     PDF available    -> CRAG/RAG workflow
    #     No PDF           -> direct LLM answer
    #
    pdf_available: bool

    # ----------------------------------------------------------------
    # Retrieval / CRAG
    # ----------------------------------------------------------------

    documents: List[Dict[str, str]]

    doc_relevant: bool

    # Used when retrieval should be skipped for simple conversational
    # queries.
    skip_retrieval: bool

    # ----------------------------------------------------------------
    # Query rewriting + web fallback
    # ----------------------------------------------------------------

    rewritten_query: Optional[str]

    used_web: bool

    web_results: List[Dict[str, str]]

    # ----------------------------------------------------------------
    # Generation + Self-RAG verification
    # ----------------------------------------------------------------

    answer: str

    sources: List[Dict[str, Any]]

    supported: bool

    # Prevents the verification/regeneration process from looping.
    regenerated: bool