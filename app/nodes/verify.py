"""Node 6: Self-RAG verification (grounding check).

Skipped entirely (no LLM call) when there is no retrieved/web context to
check against -- i.e. plain conversational messages -- since verification
provides little value there.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig

from app.llm_utils import call_structured
from app.state import GraphState, SupportGrade

_PROMPT = """Context:
{context}

Answer:
{answer}

Is the answer fully supported by the context above? Reply with ONLY compact JSON: {{"supported": true}} or {{"supported": false}}"""


def verify_node(state: GraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    has_context = bool(state.get("documents")) or bool(state.get("web_results"))
    if not has_context:
        return {"supported": True}

    context_parts = [d["content"] for d in (state.get("documents") or [])]
    context_parts += [r.get("snippet", "") for r in (state.get("web_results") or [])]
    context = "\n---\n".join(context_parts)[:1500]

    prompt = _PROMPT.format(context=context, answer=(state.get("answer") or "")[:800])

    grade = call_structured(
        prompt,
        schema=SupportGrade,
        default=SupportGrade(supported=True),
        max_tokens=20,
    )
    return {"supported": grade.supported}
