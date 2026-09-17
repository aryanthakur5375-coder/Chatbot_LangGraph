# AI Research Assistant — CRAG + Self-RAG

A lightweight, local-first PDF research assistant built on **Streamlit + LangGraph + FAISS + SQLite + OpenRouter**. It upgrades a simple tool-calling PDF chatbot into an explicit **Corrective RAG (CRAG)** workflow with a **Self-RAG** grounding check, while keeping LLM/token usage to a minimum.

## Overview

Ask questions about an uploaded PDF. The assistant retrieves relevant chunks from a FAISS index, grades whether they actually answer the question, and — only if they don't — rewrites the query and falls back to a live web search. The final answer is checked for groundedness and regenerated (at most once) if it isn't well supported.

## Features

- 📄 PDF upload, chunking, embedding, and FAISS indexing (per conversation thread, multiple PDFs merge into one index)
- 🧠 **Corrective RAG**: a lightweight LLM call grades retrieved chunks; irrelevant retrieval triggers query rewriting + web search instead of guessing
- 🔍 Web search fallback (DuckDuckGo) with real source titles/URLs
- ✅ **Self-RAG** grounding check with a single automatic regeneration if the first answer isn't supported by the context
- 💬 Multi-turn conversation threads persisted via LangGraph's SQLite checkpointer
- 🔗 Source citations shown in the UI and preserved across thread reloads
- 🪪 Clear indication whenever web search was used
- 🧵 New chat / clear conversation / switch between past threads
- 🧯 Graceful error handling (missing API key, OpenRouter errors, empty/corrupt PDFs, FAISS failures, web search failures, malformed model output)
- 🐳 Docker-ready, no secrets baked into the image

## Architecture — CRAG + Self-RAG workflow

```
START
  |
  v
retrieve --(simple greeting? regex, no LLM)-----------------> generate
  | (FAISS top-k, deterministic)
  v
grade_documents --(no docs retrieved? skip LLM, irrelevant)
  | (1 compact LLM call: {"relevant": true/false})
  |
  |-- relevant -------------------------------------> generate
  |
  '-- not relevant
        v
      rewrite_query (1 short LLM call -> search-optimized query)
        v
      web_search (DuckDuckGo, no LLM call, returns titles+URLs)
        v
      generate
            | (1 LLM call, answers from PDF context or web context)
            |
            |-- no context at all (plain chit-chat) --------> finalize
            |
            '-- has context
                  v
                verify -- (1 compact LLM call: {"supported": true/false})
                  |
                  |-- supported OR already regenerated ----> finalize
                  |
                  '-- not supported
                        v
                      regenerate (1 LLM call, stricter grounding prompt)
                        v
                      finalize
                        v
                       END
```

**LLM call budget per turn** (this is the core token-optimization strategy):

| Scenario | LLM calls |
|---|---|
| Simple greeting / small talk | 1 (generate only — retrieval, grading, verification all skipped deterministically) |
| Good PDF match | 2 (grade, generate) |
| Good PDF match, answer grounded | 3 (grade, generate, verify) |
| Poor PDF match → web fallback | 4 (grade, rewrite, generate, verify) |
| Any of the above + one correction | +1 (regenerate), capped — never loops |

Retrieval, web search, and the chit-chat check never call the LLM — they're deterministic Python/regex logic, per the project's token-minimization goal.

### CRAG (Corrective RAG)
- `grade_documents` asks the model for nothing more than `{"relevant": true|false}`, built from up to 4 short chunk snippets. If no documents were retrieved at all, this is decided in code with zero LLM calls.
- Irrelevant retrieval → `rewrite_query` (short, search-optimized rewrite) → `web_search`. Web search is *never* triggered when PDF retrieval already looks good.

### Self-RAG / self-correction
- `verify` asks for `{"supported": true|false}` comparing the answer against the exact context used to generate it.
- Skipped entirely (no LLM call) when there's no context to check against, i.e. plain conversational turns.
- On `false`, `regenerate` re-runs generation once with a stricter grounding prompt. There is **no second verification pass** — this hard caps correction at one cycle and makes an infinite loop structurally impossible.

### Structured output without relying on tool-calling
Grading and verification ask the model for a single compact JSON object and validate it with a Pydantic model (`RelevanceGrade`, `SupportGrade`) rather than using OpenRouter's function-calling, since tool-calling support is inconsistent across free/open models. Any parsing failure falls back to a safe default instead of crashing the graph.

## Tech stack

Python · Streamlit · LangChain · LangGraph · FAISS · SQLite (LangGraph checkpointer) · sentence-transformers (`all-MiniLM-L6-v2`) embeddings · OpenRouter (any OpenAI-compatible chat model) · DuckDuckGo Search · Docker

## Project structure

```
app/
  config.py            # env vars, OpenRouter LLM factory, embeddings
  state.py             # GraphState TypedDict + Pydantic grading schemas
  llm_utils.py         # compact structured-JSON LLM call helper (graceful fallback)
  graph.py             # LangGraph StateGraph wiring + routing functions
  nodes/
    retrieve.py         # FAISS top-k retrieval + chit-chat short-circuit (no LLM)
    grade.py            # CRAG relevance grading (1 LLM call, conditional)
    rewrite.py          # query rewriting for web search (1 LLM call, conditional)
    search.py           # DuckDuckGo web search fallback (no LLM)
    generate.py         # answer generation + the capped regeneration node
    verify.py           # Self-RAG groundedness check (1 LLM call, conditional)
    finalize.py         # appends Human/AI turn (+ source footer) to history
  utils/
    ingestion.py         # PDF parsing, chunking, per-thread FAISS store
    db.py                # SQLite checkpointer + thread listing
    heuristics.py         # regex-based chit-chat detector (no LLM)
frontend.py             # Streamlit UI
requirements.txt
Dockerfile
.dockerignore
.env.example
```

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Your OpenRouter API key |
| `OPENROUTER_MODEL` | No | `openai/gpt-oss-20b:free` | Any OpenRouter-compatible chat model |
| `RAG_TOP_K` | No | `4` | FAISS retrieval count |
| `CHUNK_SIZE` | No | `800` | PDF chunk size (characters) |
| `CHUNK_OVERLAP` | No | `120` | PDF chunk overlap |
| `WEB_SEARCH_RESULTS` | No | `3` | Max web results per fallback search |
| `CHATBOT_DB_PATH` | No | `chatbot.db` | SQLite checkpoint DB path |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |

Copy `.env.example` to `.env` and fill in your key.

## Local execution

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then edit .env and add OPENROUTER_API_KEY
streamlit run frontend.py
```

Open the printed local URL, upload a PDF in the sidebar, and start chatting.

## Docker execution

```bash
docker build -t ai-research-assistant .
docker run -p 8501:8501 \
  -e OPENROUTER_API_KEY=your_key_here \
  -e OPENROUTER_MODEL=openai/gpt-oss-20b:free \
  -v $(pwd)/data:/app/data \
  ai-research-assistant
```

No secrets are baked into the image — the API key is supplied at `docker run` time. The `-v` volume mount persists the SQLite conversation history across container restarts.

## Known limitations

- **FAISS indexes are in-memory only** (per running process), matching the original project's behavior. Restarting the app requires re-uploading PDFs; conversation history and citations embedded in past messages still persist via SQLite.
- To minimize tokens, `generate` does **not** replay the full conversation history into the prompt — each turn is answered from the current question plus freshly retrieved/searched context only. Long multi-turn follow-ups that depend on earlier turns' phrasing may need to be asked more explicitly (e.g. "What about France's population?" instead of "What about its population?").
- Web search failures (network errors, rate limits) degrade gracefully to an empty result set — the model then answers from general knowledge and says so if it can't ground the answer.

## What changed from the original repo

- Replaced the single `backend.py` ReAct/tool-calling agent with an explicit CRAG + Self-RAG `StateGraph` under `app/`.
- Kept: PDF ingestion → chunking → FAISS → retrieval pipeline, SQLite checkpointing for threads, OpenRouter as the LLM provider, and the overall Streamlit chat UX (sidebar, thread list, PDF status).
- Removed: the generic ReAct tool-calling loop, the calculator/stock-price demo tools (out of scope for a research assistant and each an extra, unnecessary LLM-routable tool call), `torch`/`torchvision` as *direct* pins (still pulled in transitively by `sentence-transformers`, but no longer duplicated), the deprecated `duckduckgo-search` package in favor of the actively maintained `ddgs`.
- Added: relevance grading, query rewriting, web-search fallback, groundedness verification with capped regeneration, per-turn state resets, a deterministic chit-chat short-circuit, structured Pydantic-validated LLM outputs, modular `app/` package layout, Docker/`.dockerignore`/`.env.example`.

## Dependencies

**Added:** `pydantic` (explicit — structured grading/verification output), `ddgs` (replaces the deprecated `duckduckgo-search`).

**Removed:** `torch`, `torchvision` (as direct pins — still installed transitively via `sentence-transformers`), `duckduckgo-search` (superseded by `ddgs`), `openai` (unused directly; `langchain-openai` covers the OpenRouter client).

## Verification performed

- ✅ Full `StateGraph` builds and compiles with no warnings
- ✅ Simple greeting → single LLM call, retrieval/grading/verification all skipped
- ✅ Relevant PDF match → grade + generate (+ verify) only, no web search triggered
- ✅ Irrelevant PDF match → rewrite + web search fallback fires correctly, with real source URLs
- ✅ Self-RAG correction → exactly one regeneration when the answer is flagged unsupported, no loop
- ✅ Missing API key, malformed model JSON, and empty PDF uploads all fail gracefully with clear messages instead of crashing
- ✅ `frontend.py` boots under Streamlit and serves successfully
- ✅ All Python modules compile; every pinned dependency in `requirements.txt` exists on PyPI and resolves together with no conflicts
- ⚠️ Not run in this environment: an actual `docker build` (no Docker daemon available here) and a live OpenRouter call (no network access to openrouter.ai available here). The Dockerfile and OpenRouter client code follow the same patterns validated above and should work as-is — please run `docker build -t ai-research-assistant .` and a real chat turn locally to do the final confirmation.

## CRAG / Self-RAG feature confirmation

| Requirement | Status |
|---|---|
| FAISS RAG (upload, split, embed, retrieve, source tracking) | Implemented |
| Relevance grading (compact JSON, no LLM when docs empty) | Implemented |
| Query rewriting (only when retrieval insufficient) | Implemented |
| Web search fallback (only when needed, returns sources) | Implemented |
| Self-RAG groundedness check + max-one regeneration | Implemented |
| Skips verification for simple conversational messages | Implemented (regex chit-chat short-circuit) |
| LangGraph StateGraph with conditional edges | Implemented |
| OpenRouter via env vars, no hardcoded keys | Implemented |
| SQLite checkpointing, minimal stored state | Implemented |
| Streamlit UI: chat, upload, status, threads, citations, web indicator, reset | Implemented |
| Docker (Dockerfile, .dockerignore, requirements.txt), no secrets in image | Implemented |
| Graceful error handling across all specified failure modes | Implemented |
