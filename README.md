# AI Research Assistant (RAG Chatbot)

An intelligent AI-powered Research Assistant built using LangGraph, LangChain, Streamlit, and FAISS. The chatbot combines Retrieval-Augmented Generation (RAG) with web search to answer questions accurately using both uploaded documents and live internet information.

---

## Features

- Upload PDF documents for question answering
- Retrieval-Augmented Generation (RAG) using FAISS
- Automatic web search for recent or unavailable information
- LangGraph-based agent workflow
- Conversational chat interface built with Streamlit
- Chat history support
- Semantic document retrieval using vector embeddings
- Environment variable support with `.env`

---

## Tech Stack

### Programming Language
- Python

### AI Frameworks
- LangGraph
- LangChain

### Large Language Model
- Configurable LLM (currently Google Gemini)

### Retrieval-Augmented Generation
- FAISS
- PyPDFLoader
- Recursive Character Text Splitter
- HuggingFace Embeddings

### Frontend
- Streamlit

### Utilities
- SQLite
- DuckDuckGo Search
- Python Dotenv

---

## Architecture

```text
                User Query
                     │
                     ▼
              LangGraph Agent
                     │
      ┌──────────────┴──────────────┐
      │                             │
      ▼                             ▼
Retrieve from PDFs           Search the Web
      │                             │
      └──────────────┬──────────────┘
                     ▼
             Large Language Model
                     ▼
              Final Response
```

---

## Installation

...

---

## Environment Variables

```env
GOOGLE_API_KEY=your_api_key
```

---

## Running the Application

```bash
streamlit run frontend.py
```

---

## Future Improvements

- Support multiple document formats
- Hybrid search (BM25 + Vector Search)
- Persistent conversation memory
- Source citations
- User authentication
- Multi-agent workflows
- Streaming responses
- Cloud deployment

---

## Learning Outcomes

- Retrieval-Augmented Generation (RAG)
- LangGraph workflows
- LangChain tool integration
- Vector databases with FAISS
- Semantic search
- Prompt engineering
- LLM orchestration
- Streamlit application development
- Agent-based AI systems

---

## Author

Aryan Thakur

GitHub: https://github.com/aryanthakur5375-coder
