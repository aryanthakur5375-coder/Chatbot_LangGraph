"""Small helpers shared by grading/verification nodes.

Rather than relying on OpenRouter's (inconsistent, model-dependent) native
function-calling support, we ask the model for a single compact JSON object
and validate it with Pydantic. This works with virtually any chat model
available on OpenRouter -- including free-tier models that don't reliably
support tool calling -- while still giving us strict, typed output.
"""
from __future__ import annotations

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel

from app.config import get_llm

T = TypeVar("T", bound=BaseModel)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_structured(
    prompt: str,
    schema: Type[T],
    default: T,
    max_tokens: int = 20,
) -> T:
    """Call the LLM once, expecting a tiny JSON object matching ``schema``.

    On any failure (missing API key, network/rate-limit error, invalid or
    unparsable JSON) this returns ``default`` instead of raising, so a single
    flaky model response never crashes the graph.
    """
    try:
        llm = get_llm(max_tokens=max_tokens, temperature=0.0)
        response = llm.invoke(prompt)
        text = (response.content or "").strip()
        match = _JSON_RE.search(text)
        raw = match.group(0) if match else text
        data = json.loads(raw)
        return schema.model_validate(data)
    except Exception:
        return default
