"""PDF ingestion and per-thread FAISS vector stores.

Each chat thread gets its own in-memory FAISS index so documents from one
conversation never leak into another. Multiple PDFs uploaded to the same
thread are merged into the same index (rather than replacing it), so
earlier uploads stay searchable.

Note: indexes live in-process only (not persisted to disk), matching the
original project's behavior -- restarting the app requires re-uploading
PDFs. This keeps the app lightweight and dependency-free (no extra DB for
vectors); it's documented as a known limitation in the README.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, RAG_TOP_K, get_embeddings

_THREAD_STORES: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, Dict[str, Any]] = {}


class PDFIngestionError(ValueError):
    """Raised for empty/corrupt/unreadable PDFs."""


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """Parse, chunk, embed and index a PDF for the given thread.

    Returns metadata about the ingested file. Raises ``PDFIngestionError``
    for empty files or parsing failures so the caller can show a clean
    error message instead of crashing.
    """
    if not file_bytes:
        raise PDFIngestionError("The uploaded PDF is empty.")

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        try:
            docs = PyPDFLoader(temp_path).load()
        except Exception as exc:  # corrupt / unreadable PDF
            raise PDFIngestionError(f"Could not parse PDF: {exc}") from exc

        if not docs:
            raise PDFIngestionError("No extractable text found in this PDF.")

        for doc in docs:
            doc.metadata["source"] = filename or "uploaded.pdf"

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise PDFIngestionError("Document produced no usable text chunks.")

        key = str(thread_id)
        embeddings = get_embeddings()

        try:
            existing_store = _THREAD_STORES.get(key)
            if existing_store is None:
                store = FAISS.from_documents(chunks, embeddings)
            else:
                existing_store.add_documents(chunks)
                store = existing_store
        except Exception as exc:
            raise PDFIngestionError(f"Failed to build vector index: {exc}") from exc

        _THREAD_STORES[key] = store

        meta = _THREAD_METADATA.setdefault(
            key, {"files": [], "documents": 0, "chunks": 0}
        )
        meta["files"] = meta["files"] + [filename] if filename not in meta["files"] else meta["files"]
        meta["documents"] += len(docs)
        meta["chunks"] += len(chunks)
        meta["filename"] = filename
        meta["last_documents"] = len(docs)
        meta["last_chunks"] = len(chunks)

        return dict(meta)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def get_retriever(thread_id: Optional[str]):
    """Return a top-k retriever for the thread, or ``None`` if no PDF yet."""
    if thread_id is None:
        return None
    store = _THREAD_STORES.get(str(thread_id))
    if store is None:
        return None
    return store.as_retriever(search_kwargs={"k": RAG_TOP_K})


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_STORES


def thread_document_metadata(thread_id: str) -> Dict[str, Any]:
    return _THREAD_METADATA.get(str(thread_id), {})
