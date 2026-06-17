# AI Chatbot with LangGraph, Tool Calling & Memory

An AI-powered chatbot built using LangGraph, Streamlit, and OpenRouter. The chatbot supports tool calling, persistent conversation memory using SQLite, web search, stock price retrieval, and mathematical calculations.

## Features

* Conversational AI powered by OpenRouter LLMs
* LangGraph workflow orchestration
* Persistent memory using SQLite Checkpointer
* Tool Calling support
* DuckDuckGo Web Search Tool
* Stock Price Retrieval Tool
* Calculator Tool
* Multi-thread conversation support
* Streamlit-based user interface
* Streaming responses

## Tech Stack

### Backend

* Python
* LangGraph
* LangChain
* OpenRouter
* SQLite

### Frontend

* Streamlit

### Tools

* DuckDuckGo Search
* Alpha Vantage Stock API
* Custom Calculator Tool

## Project Structure

```text
Chatbot_LangGraph/
│
├── backend.py
├── frontend.py
├── requirements.txt
├── chatbot.db
├── .env
└── README.md
```

## Workflow

```text
START
  │
  ▼
chat_node
  │
  ├── No Tool Required ─────► END
  │
  └── Tool Required
          │
          ▼
        tools
          │
          ▼
      chat_node
          │
          ▼
         END
```

## Supported Tools

### Calculator Tool

Performs:

* Addition
* Subtraction
* Multiplication
* Division

Example:

```text
What is 45 multiplied by 16?
```

### Stock Price Tool

Fetches latest stock prices using Alpha Vantage API.

Example:

```text
What is the stock price of INFY?
```

### Web Search Tool

Searches the internet using DuckDuckGo.

Example:

```text
Who is the current Prime Minister of India?
```

## Memory System

The chatbot uses SQLite Checkpointer for persistent memory.

Features:

* Conversation history persistence
* Multi-thread support
* Session restoration
* Long-running conversations

## Future Improvements

* RAG Integration
* PDF Chat Support
* Vector Database Integration
* Multi-Agent Workflows
* Conversation Summarization
* Voice Assistant Support
* Document Upload & Analysis

## Author

Aryan Thakur

Computer Science Engineering Student
JECRC Foundation

```
```
