# Setting Up LangGraph

## Installation

Install the core library from PyPI. Most projects also install `langchain` and a
model provider package (e.g. `langchain-openai`) since agents typically call an LLM
through LangChain's chat model interface:

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

If you want to run a local LangGraph API server for development (with the LangGraph
Studio UI), also install:

```bash
pip install "langgraph-cli[inmem]"
```

## Environment variables

Create a `.env` file in your project root with your model provider credentials:

```
OPENAI_API_KEY=sk-...
LANGGRAPH_API_KEY=...       # only needed for LangGraph Platform deployments
LANGCHAIN_TRACING_V2=true   # optional: enables LangSmith tracing
LANGCHAIN_API_KEY=...       # optional: LangSmith API key
```

Load it at the top of your script with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Minimal project structure

A typical LangGraph agent project looks like this:

```
my-agent/
├── .env
├── requirements.txt
├── graph.py          # defines the StateGraph, nodes, and edges
├── tools.py          # tool functions the agent can call
└── main.py           # entry point that invokes/streams the graph
```

## Running a graph locally

Once a graph is compiled with `graph = builder.compile()`, you can invoke it directly
in Python:

```python
result = graph.invoke({"messages": [("user", "What's the weather in Paris?")]})
print(result["messages"][-1].content)
```

Or run it as a local dev server with hot-reload and the LangGraph Studio debugging UI:

```bash
langgraph dev
```

This starts an API server (default `http://127.0.0.1:2024`) and opens a browser-based
graph visualizer where you can step through execution, inspect state at each node, and
replay from any checkpoint.

## Choosing a checkpointer

For local development, `MemorySaver` (in-memory) is enough:

```python
from langgraph.checkpoint.memory import MemorySaver
graph = builder.compile(checkpointer=MemorySaver())
```

For production, use a persistent backend such as `langgraph-checkpoint-sqlite` or
`langgraph-checkpoint-postgres` so state survives process restarts.
