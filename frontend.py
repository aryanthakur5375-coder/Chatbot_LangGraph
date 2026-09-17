"""
Streamlit UI for the CRAG + Self-RAG AI Research Assistant.

Features:
- PDF upload and ingestion status
- CRAG workflow status
- Direct LLM answering when no PDF is available
- Web-search indicator
- Source citations
- Persistent conversation threads
- New/clear conversation controls
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from app.config import OPENROUTER_MODEL, has_api_key
from app.graph import chatbot
from app.utils.db import retrieve_all_threads
from app.utils.ingestion import (
    PDFIngestionError,
    ingest_pdf,
    thread_document_metadata,
)


# --------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------------------------
# Node status messages
# --------------------------------------------------------------------

NODE_STATUS = {
    "retrieve": ("📄", "Searching your documents..."),
    "grade_documents": ("🧠", "Checking relevance..."),
    "rewrite_query": ("✏️", "Rewriting your question..."),
    "web_search": ("🔍", "Searching the web..."),
    "generate": ("✍️", "Generating answer..."),
    "verify": ("✅", "Checking the answer is grounded..."),
    "regenerate": ("🔁", "Regenerating a more grounded answer..."),
    "finalize": ("💾", "Wrapping up..."),
    "direct_answer": ("✍️", "Generating answer..."),
}


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def generate_thread_id() -> str:
    """Generate a unique conversation/thread ID."""
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    """Add a thread to session state if it doesn't already exist."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    """Create a new conversation."""
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []

    add_thread(thread_id)


def load_conversation(thread_id: str) -> list[dict]:
    """Load saved conversation messages from LangGraph/SQLite."""

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    messages = state.values.get("messages", [])

    history = []

    for msg in messages:

        if isinstance(msg, HumanMessage):
            role = "user"

        elif isinstance(msg, AIMessage):

            if not msg.content:
                continue

            role = "assistant"

        else:
            continue

        history.append(
            {
                "role": role,
                "content": msg.content,
            }
        )

    return history


# --------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()


add_thread(
    st.session_state["thread_id"]
)


thread_key = str(
    st.session_state["thread_id"]
)


threads = st.session_state["chat_threads"][::-1]

selected_thread = None


# --------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------

st.sidebar.title(
    "🔎 AI Research Assistant"
)


st.sidebar.caption(
    f"Model: `{OPENROUTER_MODEL}`"
)


st.sidebar.markdown(
    f"**Thread ID:** `{thread_key}`"
)


# --------------------------------------------------------------------
# API key status
# --------------------------------------------------------------------

if not has_api_key():

    st.sidebar.error(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file."
    )


# --------------------------------------------------------------------
# New chat
# --------------------------------------------------------------------

if st.sidebar.button(
    "➕ New chat",
    use_container_width=True,
):

    reset_chat()

    st.rerun()


# --------------------------------------------------------------------
# Clear conversation
# --------------------------------------------------------------------

if st.sidebar.button(
    "🗑️ Clear this conversation",
    use_container_width=True,
):

    st.session_state["message_history"] = []

    st.rerun()


st.sidebar.divider()


# --------------------------------------------------------------------
# PDF status
# --------------------------------------------------------------------

doc_meta = thread_document_metadata(
    thread_key
)


if doc_meta:

    files = ", ".join(
        file_name
        for file_name in doc_meta.get("files", [])
        if file_name
    ) or "document"

    st.sidebar.success(
        f"Indexed: {files}\n\n"
        f"{doc_meta.get('chunks', 0)} chunks "
        f"from {doc_meta.get('documents', 0)} pages"
    )

else:

    st.sidebar.info(
        "No PDF indexed yet for this chat."
    )


# --------------------------------------------------------------------
# PDF upload
# --------------------------------------------------------------------

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
)


if uploaded_pdf is not None:

    already_done = uploaded_pdf.name in (
        doc_meta.get("files") or []
    )

    if already_done:

        st.sidebar.info(
            f"`{uploaded_pdf.name}` already indexed."
        )

    else:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True,
        ) as status_box:

            try:

                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )

                status_box.update(
                    label=(
                        f"✅ Indexed "
                        f"{summary.get('last_chunks', 0)} chunks"
                    ),
                    state="complete",
                    expanded=False,
                )

            except PDFIngestionError as exc:

                status_box.update(
                    label=f"❌ {exc}",
                    state="error",
                    expanded=True,
                )

            except Exception as exc:

                status_box.update(
                    label=f"❌ Unexpected error: {exc}",
                    state="error",
                    expanded=True,
                )


st.sidebar.divider()


# --------------------------------------------------------------------
# Past conversations
# --------------------------------------------------------------------

st.sidebar.subheader(
    "💬 Past conversations"
)


if not threads:

    st.sidebar.write(
        "No conversations yet."
    )

else:

    for tid in threads:

        if st.sidebar.button(
            str(tid),
            key=f"thread-{tid}",
            use_container_width=True,
        ):

            selected_thread = tid


# --------------------------------------------------------------------
# Main UI
# --------------------------------------------------------------------

st.title(
    "AI Research Assistant"
)


st.caption(
    "Corrective RAG (CRAG) + Self-RAG · "
    "PDF grounding with web-search fallback"
)


# --------------------------------------------------------------------
# Display conversation history
# --------------------------------------------------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# --------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------

user_input = st.chat_input(
    "Ask anything..."
)


if user_input:

    # ---------------------------------------------------------------
    # API key check
    # ---------------------------------------------------------------

    if not has_api_key():

        st.error(
            "Please set OPENROUTER_API_KEY "
            "before chatting."
        )

    else:

        # -----------------------------------------------------------
        # Refresh document metadata
        # -----------------------------------------------------------

        doc_meta = thread_document_metadata(
            thread_key
        )

        # True only when a PDF has been successfully indexed
        pdf_available = bool(doc_meta)


        # -----------------------------------------------------------
        # Save/display user message
        # -----------------------------------------------------------

        st.session_state["message_history"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                user_input
            )


        # -----------------------------------------------------------
        # LangGraph configuration
        # -----------------------------------------------------------

        config = {
            "configurable": {
                "thread_id": thread_key
            }
        }


        # -----------------------------------------------------------
        # Graph input
        #
        # pdf_available tells the graph whether it should use
        # the CRAG/RAG path or direct LLM answering.
        # -----------------------------------------------------------

        graph_input = {
            "question": user_input,
            "pdf_available": pdf_available,
        }


        # -----------------------------------------------------------
        # Assistant response
        # -----------------------------------------------------------

        with st.chat_message("assistant"):

            status_box = st.status(
                "🧠 Thinking...",
                expanded=True,
            )

            final_state: dict = {}


            try:

                for update in chatbot.stream(
                    graph_input,
                    config=config,
                    stream_mode="updates",
                ):

                    for node_name, node_update in update.items():

                        # ------------------------------------------------
                        # IMPORTANT:
                        #
                        # If there is no indexed PDF, don't display
                        # the document retrieval message.
                        #
                        # The graph should route to direct_answer.
                        # ------------------------------------------------

                        if (
                            node_name == "retrieve"
                            and not pdf_available
                        ):
                            continue


                        icon, label = NODE_STATUS.get(
                            node_name,
                            ("⚙️", node_name),
                        )


                        status_box.update(
                            label=f"{icon} {label}",
                            state="running",
                            expanded=True,
                        )


                        if isinstance(
                            node_update,
                            dict,
                        ):

                            final_state.update(
                                node_update
                            )


                # --------------------------------------------------------
                # Successful completion
                # --------------------------------------------------------

                status_box.update(
                    label="✅ Done",
                    state="complete",
                    expanded=False,
                )


            except Exception as exc:

                # --------------------------------------------------------
                # Graceful error handling
                # --------------------------------------------------------

                status_box.update(
                    label="⚠️ Something went wrong",
                    state="error",
                    expanded=True,
                )


                final_state["answer"] = (
                    f"Sorry, an error occurred: {exc}"
                )


                final_state.setdefault(
                    "sources",
                    [],
                )


                final_state.setdefault(
                    "used_web",
                    False,
                )


            # -----------------------------------------------------------
            # Final answer
            # -----------------------------------------------------------

            answer = (
                final_state.get("answer")
                or "Sorry, I couldn't produce an answer."
            )


            st.markdown(
                answer
            )


            # -----------------------------------------------------------
            # Web search indicator
            # -----------------------------------------------------------

            if final_state.get("used_web"):

                st.caption(
                    "🔍 Answer includes live web search results"
                )


            # -----------------------------------------------------------
            # Sources
            # -----------------------------------------------------------

            sources = (
                final_state.get("sources")
                or []
            )


            if sources:

                with st.expander(
                    "Sources"
                ):

                    for source in sources:

                        if source.get("type") == "pdf":

                            st.write(
                                f"📄 "
                                f"{source.get('label', 'PDF')}"
                            )

                        else:

                            label = source.get(
                                "label",
                                "Web source",
                            )

                            url = source.get(
                                "url",
                                "",
                            )

                            if url:

                                st.markdown(
                                    f"🔗 [{label}]({url})"
                                )

                            else:

                                st.write(
                                    f"🔗 {label}"
                                )


        # -----------------------------------------------------------
        # Save assistant response
        # -----------------------------------------------------------

        st.session_state["message_history"].append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


# --------------------------------------------------------------------
# Thread switching
# --------------------------------------------------------------------

st.divider()


if selected_thread:

    st.session_state["thread_id"] = str(
        selected_thread
    )


    st.session_state["message_history"] = (
        load_conversation(
            selected_thread
        )
    )


    st.rerun()