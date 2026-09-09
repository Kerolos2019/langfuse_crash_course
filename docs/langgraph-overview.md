# What is LangGraph?

LangGraph is a low-level orchestration framework, built by the LangChain team, for
building stateful, multi-step agents and workflows with large language models. Unlike
simple prompt-chaining libraries, LangGraph models an application as a **graph**:

- **Nodes** are units of work — usually a Python function or a call to an LLM/tool.
- **Edges** define how control flows between nodes, including conditional edges that
  branch based on the current state.
- **State** is a shared object (typically a `TypedDict` or Pydantic model) that is
  passed between nodes and updated as the graph executes.

This graph-based model makes LangGraph well suited for agents that need to loop,
retry, branch, or run multiple steps in an order that isn't known in advance — for
example, an agent that decides whether to call a tool, re-plan, or ask a clarifying
question before producing a final answer.

## Why not just use a plain LLM loop?

You can absolutely write a `while` loop that calls an LLM and executes tools. LangGraph
gives you that same pattern with extra guarantees that get hard to build yourself once
an agent grows past a toy example:

- **Persistence** — checkpoints of the graph state can be saved after every step, so a
  long-running or failed agent can resume exactly where it left off.
- **Human-in-the-loop** — a graph can pause before a node (e.g. before executing a
  risky tool call) and wait for a human to approve, edit, or reject the proposed action.
- **Streaming** — intermediate state updates, individual LLM tokens, and custom events
  can all be streamed to a caller as the graph runs.
- **Time travel / replay** — because state is checkpointed, you can rewind a graph to
  an earlier step, edit the state, and resume from there for debugging.
- **Composability** — graphs can be nested as subgraphs, making it easier to build
  multi-agent systems out of smaller, testable pieces.

## Core building blocks

| Concept | Description |
|---|---|
| `StateGraph` | The main class used to define nodes, edges, and the state schema. |
| `State` | A schema (TypedDict/Pydantic model) describing what data flows through the graph. |
| `Node` | A function `(state) -> partial_state_update` added to the graph with `add_node`. |
| `Edge` | A connection between nodes, added with `add_edge` (fixed) or `add_conditional_edges` (branching). |
| `Checkpointer` | A backend (e.g. in-memory, SQLite, Postgres) that persists state after each step. |
| `Command` | An object a node can return to update state *and* control routing in one step. |

LangGraph is commonly paired with LangChain's chat model and tool abstractions, but it
does not require LangChain — nodes can call any LLM SDK (OpenAI, Anthropic, etc.)
directly.
