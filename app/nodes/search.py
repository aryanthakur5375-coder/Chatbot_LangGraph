"""Node 4: web search fallback (deterministic, no LLM call).

Only reached when FAISS retrieval was irrelevant/insufficient. Returns
titles + URLs so the UI can show real source citations.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig

from app.config import WEB_SEARCH_RESULTS


def web_search_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    query = state.get("rewritten_query") or state.get("question", "")

    try:
        from langchain_community.tools import DuckDuckGoSearchResults

        tool = DuckDuckGoSearchResults(
            output_format="list",
            num_results=WEB_SEARCH_RESULTS,
        )
        raw_results = tool.invoke(query)
    except Exception:
        return {"used_web": True, "web_results": []}

    results = []
    if isinstance(raw_results, list):
        for item in raw_results[:WEB_SEARCH_RESULTS]:
            if not isinstance(item, dict):
                continue
            results.append(
                {
                    "title": item.get("title") or "Web result",
                    "url": item.get("link") or item.get("url") or "",
                    "snippet": (item.get("snippet") or "")[:400],
                }
            )

    return {"used_web": True, "web_results": results}
