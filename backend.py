from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------------------------------------
# LLM
# -------------------------------------------------

llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
)

# -------------------------------------------------
# Embeddings
# -------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# Thread Stores
# -------------------------------------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

# This variable stores the current thread while
# tool execution is happening.
CURRENT_THREAD_ID: Optional[str] = None


def _get_retriever(thread_id: Optional[str]):
    if thread_id is None:
        return None

    return _THREAD_RETRIEVERS.get(str(thread_id))


# -------------------------------------------------
# PDF Ingestion
# -------------------------------------------------

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
):
    if not file_bytes:
        raise ValueError("Empty PDF.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks),
        }

        print("Indexed:", thread_id)

        return _THREAD_METADATA[str(thread_id)]

    finally:

        try:
            os.remove(temp_path)
        except Exception:
            pass


# -------------------------------------------------
# Search Tool
# -------------------------------------------------

search_tool = DuckDuckGoSearchRun(region="us-en")


# -------------------------------------------------
# Calculator
# -------------------------------------------------

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
):
    """Basic Calculator"""

    if operation == "add":
        result = first_num + second_num

    elif operation == "sub":
        result = first_num - second_num

    elif operation == "mul":
        result = first_num * second_num

    elif operation == "div":

        if second_num == 0:
            return {"error": "Division by zero"}

        result = first_num / second_num

    else:

        return {
            "error": "Invalid Operation"
        }

    return {
        "result": result
    }


# -------------------------------------------------
# Stock Tool
# -------------------------------------------------

@tool
def get_stock_price(symbol: str):

    """Get current stock price"""

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey=C9PE94QUEW9VWGFM"
    )

    data = requests.get(
        url,
        timeout=10,
    ).json()

    quote = data.get(
        "Global Quote",
        {},
    )

    if not quote:
        return {
            "error": "Stock not found"
        }

    return {
        "symbol": quote["01. symbol"],
        "price": quote["05. price"],
    }


# -------------------------------------------------
# RAG Tool
# -------------------------------------------------

@tool
def rag_tool(query: str):
    """
    Search uploaded PDF.
    """

    global CURRENT_THREAD_ID

    print("Current Thread:", CURRENT_THREAD_ID)

    retriever = _get_retriever(
        CURRENT_THREAD_ID
    )

    if retriever is None:

        return {
            "error": "Please upload a PDF first."
        }

    docs = retriever.invoke(query)

    return {
        "context": [
            doc.page_content
            for doc in docs
        ]
    }


# -------------------------------------------------
# Register Tools
# -------------------------------------------------

tools = [
    rag_tool,
    calculator,
    search_tool,
    get_stock_price,
]

llm_with_tools = llm.bind_tools(
    tools
)

# -------------------------------------------------
# State
# -------------------------------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------------------------------------
# Chat Node
# -------------------------------------------------

def chat_node(state: ChatState, config=None):

    global CURRENT_THREAD_ID

    CURRENT_THREAD_ID = None

    if config:
        CURRENT_THREAD_ID = (
            config.get("configurable", {})
            .get("thread_id")
        )

    print("Current Thread:", CURRENT_THREAD_ID)

    system_message = SystemMessage(
        content=f"""
You are a helpful AI assistant.

Current Thread ID:
{CURRENT_THREAD_ID}

Rules:

1. If the user asks anything about the uploaded PDF,
always call rag_tool.

2. Never answer PDF questions from your own knowledge.

3. If rag_tool returns context,
use ONLY that context.

4. If rag_tool returns an error,
tell the user to upload a PDF.

5. Always answer in plain text.

6. Do not use markdown.
"""
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {
        "messages": [response]
    }


# -------------------------------------------------
# Tool Node
# -------------------------------------------------

tool_node = ToolNode(
    tools
)

# -------------------------------------------------
# Checkpointer
# -------------------------------------------------

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn)


# -------------------------------------------------
# Graph
# -------------------------------------------------

graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_node(
    "tools",
    tool_node,
)

graph.add_edge(
    START,
    "chat_node",
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chat_node",
)

chatbot = graph.compile(
    checkpointer=checkpointer,
)


# -------------------------------------------------
# Helper Functions
# -------------------------------------------------

def retrieve_all_threads():

    thread_ids = set()

    for checkpoint in checkpointer.list(None):

        thread_ids.add(
            checkpoint.config["configurable"]["thread_id"]
        )

    return list(thread_ids)


def thread_has_document(
    thread_id: str,
):

    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


def thread_document_metadata(
    thread_id: str,
):

    return _THREAD_METADATA.get(
        str(thread_id),
        {},
    )