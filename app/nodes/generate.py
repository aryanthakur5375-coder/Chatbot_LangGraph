"""Node 5: generate the answer, and the (max-one-time) regeneration node.

Only the current question + a compact context window are sent to the
model -- prior conversation turns are intentionally NOT replayed into the
prompt, to keep token usage minimal (see README "Token Optimization").
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig

from app.config import MissingAPIKeyError, get_llm

MAX_CONTEXT_CHARS = 3000

_GROUNDED_SYSTEM = (
    "Answer the question using ONLY the provided context. Be concise. "
    "If the context does not contain the answer, say you don't have enough information."
)
_STRICT_SYSTEM = (
    "Answer using ONLY the provided context. Do not add any claim that is not "
    "directly supported by the context. If unsure, say you don't have enough information."
)
_PLAIN_SYSTEM = "Answer the question briefly and helpfully."


def _build_context(state: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    parts: List[str] = []
    sources: List[Dict[str, Any]] = []

    if state.get("doc_relevant") and state.get("documents"):
        for doc in state["documents"]:
            parts.append(doc["content"])
            label = doc.get("source", "PDF")
            if not any(s.get("type") == "pdf" and s.get("label") == label for s in sources):
                sources.append({"type": "pdf", "label": label})

    if state.get("used_web") and state.get("web_results"):
        for result in state["web_results"]:
            if result.get("snippet"):
                parts.append(result["snippet"])
            sources.append(
                {"type": "web", "label": result.get("title", "Web"), "url": result.get("url", "")}
            )

    context = "\n---\n".join(parts)[:MAX_CONTEXT_CHARS]
    return context, sources


def _generate(state: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    question = state.get("question", "")
    context, sources = _build_context(state)

    if context:
        system = _STRICT_SYSTEM if strict else _GROUNDED_SYSTEM
        prompt = f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"{_PLAIN_SYSTEM}\nQuestion: {question}\nAnswer:"

    try:
        llm = get_llm(max_tokens=450, temperature=0.1 if strict else 0.3)
        response = llm.invoke(prompt)
        answer = (response.content or "").strip() or "I don't have a response for that."
    except MissingAPIKeyError as exc:
        answer = f"⚠️ {exc}"
    except Exception as exc:
        answer = f"⚠️ Sorry, the model call failed: {exc}"

    return {"answer": answer, "sources": sources}


def generate_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    return _generate(state, strict=False)


def regenerate_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Self-RAG correction: regenerate once with stricter grounding."""
    result = _generate(state, strict=True)
    result["regenerated"] = True
    return result
