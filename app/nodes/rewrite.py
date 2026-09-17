"""Node 3: rewrite the user query for web search.

Only reached when FAISS retrieval was graded irrelevant (see graph.py
routing), so this never runs on the "good retrieval" happy path.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig

from app.config import MissingAPIKeyError, get_llm

_PROMPT = (
    "Rewrite the question below into a short, precise web search query "
    "(max 12 words, no punctuation-heavy phrasing).\n"
    "Question: {question}\n"
    "Reply with ONLY the rewritten query, nothing else."
)


def rewrite_query_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    question = state.get("question", "")

    try:
        llm = get_llm(max_tokens=40, temperature=0.2)
        response = llm.invoke(_PROMPT.format(question=question))
        rewritten = (response.content or "").strip().strip('"')
    except MissingAPIKeyError:
        rewritten = ""
    except Exception:
        rewritten = ""

    return {"rewritten_query": rewritten or question}
