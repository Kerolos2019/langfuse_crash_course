# Tools, Memory, and Human-in-the-Loop in LangGraph

## Giving an agent multiple tools

Tools are just Python functions decorated with `@tool` from `langchain_core.tools`.
The docstring becomes the tool description the model sees, and type hints become the
argument schema:

```python
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """Search internal documentation for a query and return matching snippets."""
    ...

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Use only after the user has explicitly confirmed the content."""
    ...

tools = [search_docs, send_email]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)
```

The model chooses which tool(s) to call based on the docstrings and the conversation —
you don't write routing logic per tool; `ToolNode` dispatches by name automatically.

## Short-term memory (within a single run)

Short-term memory is just the `messages` state described in the agent-building guide:
every LLM response and tool result accumulates in the same list for the duration of
one `invoke`/`stream` call.

## Long-term / cross-session memory

To remember things across separate conversations (e.g. user preferences), attach a
checkpointer keyed by a `thread_id`:

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "user-123"}}
graph.invoke({"messages": [("user", "My name is Alex.")]}, config)
graph.invoke({"messages": [("user", "What's my name?")]}, config)  # remembers "Alex"
```

Because the checkpointer persists state per `thread_id`, resuming with the same
`thread_id` continues the same conversation, even across process restarts if you use a
persistent backend like SQLite or Postgres instead of `MemorySaver`.

## Human-in-the-loop approval

To pause a graph before a sensitive tool call (like `send_email`), use an interrupt:

```python
from langgraph.types import interrupt, Command

def confirm_send(state: AgentState):
    decision = interrupt({"question": "Send this email?", "draft": state["draft"]})
    if decision != "yes":
        return {"messages": [("assistant", "Cancelled sending the email.")]}
    return Command(goto="send_email_node")
```

When the graph hits `interrupt(...)`, execution pauses and the payload is returned to
the caller. The caller resumes it later by invoking the graph again with a `Command`
containing the human's decision:

```python
graph.invoke(Command(resume="yes"), config)
```

This pattern — pause, wait for a human, resume — relies on the same checkpointing
mechanism used for long-term memory, which is why a checkpointer is required for any
graph that uses `interrupt`.

## Debugging with tracing

Pairing LangGraph with an observability tool (e.g. Langfuse or LangSmith) lets you see
each node's input/output, which tools were called, token usage, and latency for every
run — essential once an agent has more than two or three nodes.
