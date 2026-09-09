# langfuse

A set of small, progressive Python scripts for learning [Langfuse](https://langfuse.com/) observability with OpenAI — starting from a basic connection check and ending with a fully traced RAG pipeline over the LangGraph docs.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your keys:

   ```
   LANGFUSE_SECRET_KEY=""
   LANGFUSE_PUBLIC_KEY=""
   LANGFUSE_BASE_URL=""
   OPENAI_API_KEY=""
   ```

## Scripts

| File | What it shows |
|---|---|
| [1-getting_started.py](1-getting_started.py) | Verifies the Langfuse connection using `@observe` and `get_client()`. |
| [2-first_trace.py](2-first_trace.py) | Simplest tracing: drop-in `langfuse.openai` wrapper auto-logs a chat completion. |
| [3-decorator_trace.py](3-decorator_trace.py) | Decorator-based tracing (`@observe()`) with nested spans and `update_current_span`. |
| [4-context_manager_trace.py](4-context_manager_trace.py) | Explicit trace/span/generation control using context managers (Langfuse v3 API). |
| [5-rag_obs.py](5-rag_obs.py) | End-to-end RAG pipeline (LangChain doc loading + chunking, ChromaDB vector store, OpenAI embeddings/chat) with every step traced in Langfuse. |

Run any script directly, e.g.:

```bash
python 1-getting_started.py
```

`5-rag_obs.py` indexes the markdown files in [docs/](docs/) (LangGraph reference notes) into a local ChromaDB store at `chroma_db/`, then answers a query about LangGraph using the retrieved context.

## Docs

The [docs/](docs/) folder contains sample LangGraph reference material used as the knowledge base for the RAG example:

- [langgraph-overview.md](docs/langgraph-overview.md)
- [langgraph-setup.md](docs/langgraph-setup.md)
- [langgraph-building-an-agent.md](docs/langgraph-building-an-agent.md)
- [langgraph-tools-and-memory.md](docs/langgraph-tools-and-memory.md)

## Dashboard

After running any script, view traces at [https://app.langfuse.com](https://app.langfuse.com) (or your self-hosted `LANGFUSE_BASE_URL`).
