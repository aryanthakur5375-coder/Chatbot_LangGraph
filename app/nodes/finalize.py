"""Final node: record the turn as chat history for the sidebar/checkpointer.

Only the question and final answer (plus a compact source footer) are
appended to ``messages`` -- never the intermediate retrieved documents or
web snippets -- keeping the persisted conversation state small while still
letting citations survive a thread reload.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig

from langchain_core.messages import AIMessage, HumanMessage


def _format_footer(sources: List[Dict[str, Any]], used_web: bool) -> str:
    if not sources:
        return ""
    lines = ["", "Sources:"]
    for source in sources:
        if source.get("type") == "pdf":
            lines.append(f"- 📄 {source.get('label', 'PDF')}")
        else:
            label = source.get("label", "Web")
            url = source.get("url", "")
            lines.append(f"- 🔗 {label} ({url})" if url else f"- 🔗 {label}")
    return "\n".join(lines)


def finalize_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    answer = state.get("answer", "")
    footer = _format_footer(state.get("sources") or [], state.get("used_web", False))
    full_answer = f"{answer}\n{footer}" if footer else answer

    return {
        "messages": [
            HumanMessage(content=state.get("question", "")),
            AIMessage(content=full_answer),
        ]
    }
