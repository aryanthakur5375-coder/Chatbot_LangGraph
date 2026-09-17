"""Node 2: grade retrieved documents for relevance (CRAG).

Uses at most ONE small LLM call, and only when documents were actually
retrieved -- an empty retrieval is deterministically graded as irrelevant
without spending a token.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig

from app.llm_utils import call_structured
from app.state import GraphState, RelevanceGrade

_PROMPT = """You grade whether documents help answer a question.
Question: {question}
Documents:
{documents}

Reply with ONLY compact JSON, nothing else: {{"relevant": true}} or {{"relevant": false}}"""


def grade_documents_node(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    documents = state.get("documents") or []

    if not documents:
        # Deterministic: nothing retrieved means nothing relevant. No LLM call.
        return {"doc_relevant": False}

    snippet = "\n---\n".join(doc["content"][:300] for doc in documents[:4])
    prompt = _PROMPT.format(question=state.get("question", ""), documents=snippet)

    grade = call_structured(
        prompt,
        schema=RelevanceGrade,
        default=RelevanceGrade(relevant=True),
        max_tokens=20,
    )
    return {"doc_relevant": grade.relevant}
