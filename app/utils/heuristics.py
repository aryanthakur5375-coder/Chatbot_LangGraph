"""Deterministic (non-LLM) heuristics shared by graph nodes."""
from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^a-z0-9'\s]")
_SPACE_RE = re.compile(r"\s+")

_CHITCHAT_PHRASES = {
    "hi", "hello", "hey", "yo", "sup", "hiya",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how are you doing", "how's it going", "hows it going",
    "what's up", "whats up",
    "hi how are you", "hello how are you", "hey how are you",
    "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty", "cheers",
    "bye", "goodbye", "see ya", "see you", "see you later", "later",
    "ok", "okay", "cool", "nice", "great", "awesome", "got it", "sounds good",
    "who are you", "what can you do", "what do you do",
}


def _normalize(text: str) -> str:
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return _SPACE_RE.sub(" ", cleaned).strip()


def is_simple_chitchat(text: str) -> bool:
    """Return True for short greetings/small talk that need no retrieval.

    Purely rule-based (normalize + set lookup) -- no LLM call -- per the
    project's rule to never spend a model call classifying obvious cases.
    """
    if not text or len(text) > 40:
        return False
    return _normalize(text) in _CHITCHAT_PHRASES
