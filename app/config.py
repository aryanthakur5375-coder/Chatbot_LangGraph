"""Central configuration: environment variables and lazy model factories.

All tunables that affect token usage (top_k, chunk size, max_tokens) live
here so they can be adjusted from the environment without touching code.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# OpenRouter / LLM
# -------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free").strip()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# -------------------------------------------------------------------
# RAG / retrieval tuning (kept small to minimize tokens)
# -------------------------------------------------------------------
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))

# -------------------------------------------------------------------
# Storage
# -------------------------------------------------------------------
SQLITE_DB_PATH = os.getenv("CHATBOT_DB_PATH", "chatbot.db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENROUTER_API_KEY is not configured."""


def has_api_key() -> bool:
    return bool(OPENROUTER_API_KEY)


def get_llm(max_tokens: int = 400, temperature: float = 0.3):
    """Return a fresh ChatOpenAI client pointed at OpenRouter.

    A small ``max_tokens`` is set per call-site so every node only pays for
    the output it actually needs (grading/verification need ~10 tokens,
    generation needs more).
    """
    if not OPENROUTER_API_KEY:
        raise MissingAPIKeyError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file "
            "or environment before starting the app."
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Load the embedding model once per process (expensive to init)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
