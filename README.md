# 🔎 AI Research Assistant

An AI-powered research assistant built with **Python, LangChain, LangGraph, FAISS, SQLite, Streamlit, and OpenRouter**.

The application combines **Corrective Retrieval-Augmented Generation (CRAG)** with **Self-RAG style answer verification** to improve the reliability of answers generated from user-provided PDF documents.

It can also fall back to web search when the retrieved document content is not relevant enough to answer the user's question.

---

## 🚀 Features

- 📄 PDF document upload and indexing
- 🔎 Semantic document retrieval using FAISS
- 🧠 Corrective RAG (CRAG) workflow
- ✏️ Automatic query rewriting when retrieved documents are insufficient
- 🌐 Web-search fallback for missing or irrelevant information
- ✅ Self-RAG style answer verification
- 🔁 Controlled answer regeneration when verification fails
- 💬 Multi-turn conversations
- 🧵 Persistent conversation threads
- 💾 SQLite-based LangGraph checkpointing
- 🤖 OpenRouter LLM integration
- 🖥️ Streamlit user interface
- 🐳 Docker support
- ⚡ CPU-compatible local embeddings
- 🔐 Environment-variable based API configuration

---

# 🏗️ Architecture

The application is implemented as a state-based workflow using **LangGraph**.

### Without a PDF

When no PDF is uploaded, the application does not perform document retrieval.

```text
User Question
     ↓
Direct Answer
     ↓
   Finalize
     ↓
    END
```

This avoids unnecessary retrieval and web-search operations.

### With a PDF

When a PDF is available, the question enters the CRAG workflow.

```text
                    ┌───────────────┐
                    │ User Question │
                    └───────┬───────┘
                            │
                     PDF available?
                      /           \
                    No             Yes
                    │               │
                    ↓               ↓
              Direct Answer      Retrieve
                    │               │
                    │          Grade Documents
                    │            /       \
                    │       Relevant     Not Relevant
                    │          │             │
                    │          ↓             ↓
                    │       Generate    Rewrite Query
                    │                        │
                    │                        ↓
                    │                   Web Search
                    │                        │
                    │                        ↓
                    │                    Generate
                    │                        │
                    │                   Verify Answer
                    │                    /         \
                    │              Supported    Unsupported
                    │                  │             │
                    │                  │             ↓
                    │                  │        Regenerate
                    │                  │             │
                    └──────────────────┴─────────────┘
                                       ↓
                                    Finalize
                                       ↓
                                      END
```

---

# 🧠 CRAG Workflow

**Corrective Retrieval-Augmented Generation (CRAG)** improves a normal RAG pipeline by evaluating whether the retrieved documents are actually useful.

The workflow is:

```text
Question
   ↓
FAISS Retrieval
   ↓
Document Relevance Grading
   ↓
 ┌───────────────┐
 │               │
Relevant      Not Relevant
 │               │
 ↓               ↓
Generate      Rewrite Query
                 ↓
             Web Search
                 ↓
              Generate
```

If the retrieved PDF content is relevant, the answer is generated using the document context.

If the retrieved content is not relevant, the question is rewritten and the system can use web search as a fallback.

---

# ✅ Self-RAG Verification

After generating an answer from retrieved context, the system performs an additional verification step.

```text
Generated Answer
       ↓
Context Available?
       ↓
   Verification
       ↓
 ┌───────────────┐
 │               │
Supported    Unsupported
 │               │
 ↓               ↓
Finalize      Regenerate
                 ↓
              Finalize
```

The verification step checks whether the generated answer is supported by the available context.

This helps reduce unsupported statements and hallucination when answering questions based on retrieved documents.

---

# 📄 PDF RAG Pipeline

Uploaded PDFs are processed using the following pipeline:

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Extraction
    ↓
Recursive Character Splitting
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Semantic Search
    ↓
Relevant Context
```

The system stores document metadata per conversation thread so that different conversations can maintain their own indexed documents.

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| Streamlit | User interface |
| LangChain | LLM and retrieval components |
| LangGraph | Agent/workflow orchestration |
| FAISS | Vector similarity search |
| Sentence Transformers | Local text embeddings |
| SQLite | Conversation checkpointing |
| OpenRouter | LLM API |
| PyPDF | PDF processing |
| DuckDuckGo Search | Web-search fallback |
| Pydantic | Structured data validation |
| Docker | Containerization |
| Git/GitHub | Version control |

---

# 📁 Project Structure

```text
Chatbot_LangGraph/
│
├── app/
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── retrieve.py
│   │   ├── grade.py
│   │   ├── rewrite.py
│   │   ├── search.py
│   │   ├── generate.py
│   │   ├── verify.py
│   │   └── finalize.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── db.py
│   │   ├── heuristics.py
│   │   └── ingestion.py
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   └── state.py
│
├── frontend.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── .env.example
├── requirements.txt
└── README.md
```

---

# 🔄 LangGraph Nodes

The workflow is divided into independent nodes.

### `retrieve`

Retrieves relevant document chunks from the FAISS vector store.

### `grade_documents`

Evaluates whether the retrieved documents are relevant to the user's question.

### `rewrite_query`

Reformulates the original question when the retrieved documents are insufficient.

### `web_search`

Performs a web search when the PDF context cannot adequately answer the question.

### `generate`

Generates the final answer using the available context.

### `verify`

Checks whether the generated answer is supported by the retrieved context.

### `regenerate`

Generates a corrected answer when verification indicates insufficient grounding.

### `finalize`

Prepares the final response and source information.

---

# 💾 Conversation Persistence

LangGraph's SQLite checkpointer is used to persist conversation state.

Each conversation receives a unique:

```text
thread_id
```

This allows the application to:

- Maintain separate conversations
- Restore previous conversations
- Preserve message history
- Store workflow checkpoints

SQLite is intentionally used instead of a larger external database to keep the project lightweight.

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

TOP_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
MAX_OUTPUT_TOKENS=500
WEB_RESULT_LIMIT=4
```

> Never commit your `.env` file or API keys to GitHub.

A `.env.example` file can be used to document the required configuration without exposing secrets.

---

# ⚙️ Local Installation

## 1. Clone the repository

```bash
git clone https://github.com/aryanthakur5375-coder/Chatbot_LangGraph.git
cd Chatbot_LangGraph
```

## 2. Create a virtual environment

Python **3.11** is recommended for compatibility with the ML and LangChain ecosystem.

### Windows

```powershell
py -3.11 -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
python3.11 -m venv venv
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Configure Environment

Create:

```text
.env
```

and add:

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Add the remaining configuration variables if required by your local setup.

---

# ▶️ Run the Application

Start Streamlit:

```bash
python -m streamlit run frontend.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🐳 Docker

The project also includes a Docker configuration for running the application in a container.

## Build the image

```bash
docker build -t ai-research-assistant .
```

## Run the container

```bash
docker run --rm -p 8501:8501 --env-file .env ai-research-assistant
```

Then open:

```text
http://localhost:8501
```

---

# 🧪 Example Usage

### Without PDF

Ask:

```text
What is the difference between TCP and UDP?
```

The system directly generates an answer without running PDF retrieval.

### With PDF

Upload a research paper or technical document and ask:

```text
What methodology does the paper use?
```

The system:

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retrieval
 ↓
Relevance Grading
 ↓
Answer Generation
 ↓
Verification
 ↓
Final Answer
```

### When the PDF is insufficient

```text
Question
   ↓
PDF Retrieval
   ↓
Low Relevance
   ↓
Query Rewriting
   ↓
Web Search
   ↓
Answer Generation
```

---

# ⚡ Design Goals

The project focuses on keeping the architecture **simple, modular, and efficient**.

### Minimal LLM usage

LLM calls are only performed when required by the workflow.

For example:

```text
No PDF
→ Direct generation

Relevant PDF
→ Document grading
→ Generation
→ Verification

Irrelevant PDF
→ Document grading
→ Query rewriting
→ Web search
→ Generation
→ Verification
```

Deterministic operations such as routing and basic checks are handled without additional LLM calls wherever possible.

---

# 🧩 Why LangGraph?

LangGraph is used to represent the application as a controllable state machine.

Instead of a simple linear pipeline:

```text
Input → Retrieval → LLM → Output
```

the application can make decisions:

```text
Input
 ↓
Is PDF available?
 ↓
Retrieve
 ↓
Are documents relevant?
 ├── Yes → Generate
 └── No  → Rewrite → Web Search → Generate
                         ↓
                     Verify
                         ↓
                  Regenerate if needed
```

This makes the workflow easier to extend and debug.

---

# 🔎 Why FAISS?

FAISS is used for efficient vector similarity search.

It allows the application to find document chunks that are semantically related to a user's question rather than relying only on exact keyword matching.

---

# 🗄️ Why SQLite?

SQLite is used for lightweight persistent checkpoint storage.

It provides conversation persistence without requiring an external database server.

This keeps the application easy to run locally and inside Docker.

---

# 🔒 Security

The project follows basic secret-management practices:

- API keys are stored in `.env`
- `.env` is excluded through `.gitignore`
- `.env` is excluded from Docker builds
- No API keys are hard-coded into the source code

---

# 📌 Future Improvements

Possible future improvements include:

- Streaming token-by-token responses
- Better source citation formatting
- Multiple PDF support
- Persistent vector-store management
- Authentication
- PostgreSQL-backed persistence
- More advanced answer evaluation
- Automated testing
- GitHub Actions CI/CD
- Docker Hub deployment
- Observability and tracing
- Additional LLM providers

---

# 👨‍💻 Author

**Aryan Thakur**

Computer Science & Engineering  
JECRC Foundation

### Technologies

```text
Python • LangChain • LangGraph • FAISS • Streamlit
OpenRouter • SQLite • Docker • Git
```

---

# ⭐ Project Highlights

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- Corrective RAG (CRAG)
- Self-RAG concepts
- Agent/workflow orchestration
- Vector databases
- Semantic search
- LLM integration
- Structured state management
- Persistent conversations
- Web-search fallback
- Docker containerization
- Modular Python application architecture

If you find the project useful, consider giving it a ⭐ on GitHub.