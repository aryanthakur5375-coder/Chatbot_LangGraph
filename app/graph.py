"""Corrective RAG (CRAG) + Self-RAG LangGraph workflow.

Workflow:

                    START
                      │
              PDF available?
                 /          \
               YES           NO
                │             │
             retrieve      generate
                │             │
          grade_documents    END
            /        \
      relevant       not relevant
         │                │
      generate       rewrite_query
                         │
                     web_search
                         │
                      generate
                         │
                       verify
                      /      \
              supported      unsupported
                  │              │
               finalize      regenerate
                  │              │
                  └──────┬───────┘
                         │
                        END

When no PDF is available:
START -> generate -> END

This avoids unnecessary FAISS retrieval, document grading,
query rewriting, web search, and verification for normal
questions when there is no document.

With a PDF:
START -> retrieve -> grade -> CRAG workflow.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes.finalize import finalize_node
from app.nodes.generate import generate_node, regenerate_node
from app.nodes.grade import grade_documents_node
from app.nodes.retrieve import retrieve_node
from app.nodes.rewrite import rewrite_query_node
from app.nodes.search import web_search_node
from app.nodes.verify import verify_node
from app.state import GraphState
from app.utils.db import checkpointer


# --------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------

def _route_after_start(state: GraphState) -> str:
    """
    Decide whether to use the RAG/CRAG workflow.

    If a PDF has been successfully indexed for the current
    conversation, start with FAISS retrieval.

    If no PDF is available, skip RAG completely and answer
    directly with the LLM.
    """

    if state.get("pdf_available", False):
        return "retrieve"

    return "generate"


def _route_after_retrieve(state: GraphState) -> str:
    """
    Decide whether document grading is required.

    Simple conversational messages can skip document grading.
    """

    return (
        "generate"
        if state.get("skip_retrieval")
        else "grade_documents"
    )


def _route_after_grade(state: GraphState) -> str:
    """
    Route based on document relevance.

    Relevant documents:
        -> generate

    Irrelevant documents:
        -> rewrite query
        -> web search
        -> generate
    """

    return (
        "generate"
        if state.get("doc_relevant")
        else "rewrite_query"
    )


def _route_after_generate(state: GraphState) -> str:
    """
    Verify answers only when supporting context exists.

    If there is no document/web context, verification is skipped
    to avoid unnecessary LLM calls.
    """

    has_context = (
        bool(state.get("documents"))
        or bool(state.get("web_results"))
    )

    return (
        "verify"
        if has_context
        else "finalize"
    )


def _route_after_verify(state: GraphState) -> str:
    """
    Allow at most one regeneration.

    Supported:
        -> finalize

    Unsupported:
        -> regenerate

    If regeneration already happened:
        -> finalize
    """

    if (
        state.get("supported", True)
        or state.get("regenerated")
    ):
        return "finalize"

    return "regenerate"


# --------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------

def build_graph():

    graph = StateGraph(GraphState)

    # ---------------------------------------------------------------
    # Nodes
    # ---------------------------------------------------------------

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "grade_documents",
        grade_documents_node,
    )

    graph.add_node(
        "rewrite_query",
        rewrite_query_node,
    )

    graph.add_node(
        "web_search",
        web_search_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "verify",
        verify_node,
    )

    graph.add_node(
        "regenerate",
        regenerate_node,
    )

    graph.add_node(
        "finalize",
        finalize_node,
    )

    # ---------------------------------------------------------------
    # START routing
    #
    # THIS IS THE IMPORTANT CHANGE.
    #
    # Previously:
    #
    # START -> retrieve
    #
    # Now:
    #
    # PDF available
    #       YES -> retrieve
    #       NO  -> generate
    # ---------------------------------------------------------------

    graph.add_conditional_edges(
        START,
        _route_after_start,
        {
            "retrieve": "retrieve",
            "generate": "generate",
        },
    )

    # ---------------------------------------------------------------
    # Retrieve -> Grade / Generate
    # ---------------------------------------------------------------

    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {
            "generate": "generate",
            "grade_documents": "grade_documents",
        },
    )

    # ---------------------------------------------------------------
    # Grade -> Generate / Rewrite
    # ---------------------------------------------------------------

    graph.add_conditional_edges(
        "grade_documents",
        _route_after_grade,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
        },
    )

    # ---------------------------------------------------------------
    # Rewrite -> Web Search -> Generate
    # ---------------------------------------------------------------

    graph.add_edge(
        "rewrite_query",
        "web_search",
    )

    graph.add_edge(
        "web_search",
        "generate",
    )

    # ---------------------------------------------------------------
    # Generate -> Verify / Finalize
    # ---------------------------------------------------------------

    graph.add_conditional_edges(
        "generate",
        _route_after_generate,
        {
            "verify": "verify",
            "finalize": "finalize",
        },
    )

    # ---------------------------------------------------------------
    # Verify -> Finalize / Regenerate
    # ---------------------------------------------------------------

    graph.add_conditional_edges(
        "verify",
        _route_after_verify,
        {
            "finalize": "finalize",
            "regenerate": "regenerate",
        },
    )

    # ---------------------------------------------------------------
    # Regenerate -> Finalize
    #
    # There is NO edge back to verify.
    # Therefore regeneration can happen only once.
    # ---------------------------------------------------------------

    graph.add_edge(
        "regenerate",
        "finalize",
    )

    # ---------------------------------------------------------------
    # Finalize -> END
    # ---------------------------------------------------------------

    graph.add_edge(
        "finalize",
        END,
    )

    # ---------------------------------------------------------------
    # Compile with SQLite checkpointing
    # ---------------------------------------------------------------

    return graph.compile(
        checkpointer=checkpointer
    )


# --------------------------------------------------------------------
# Compiled chatbot
# --------------------------------------------------------------------

chatbot = build_graph()