# Building a Basic Agent with LangGraph

This walks through the classic "ReAct-style" agent: an LLM that can decide to call
tools in a loop until it has enough information to answer.

## 1. Define the state

The state is the data structure passed between every node. For a chat agent it's
usually just a running list of messages:

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` is a reducer: instead of overwriting the `messages` list on each
update, it appends new messages to it.

## 2. Define tools

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."

tools = [get_weather]
```

## 3. Bind tools to the model and define the agent node

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def call_model(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}
```

## 4. Add a tool-execution node and routing logic

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"
```

## 5. Wire the graph together

```python
from langgraph.graph import StateGraph, END, START

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
builder.add_edge("tools", "agent")  # loop back after a tool call

graph = builder.compile()
```

This produces a loop: `agent -> (tools -> agent)* -> END`. The agent node calls the
model; if the model requests a tool call, control passes to the `tools` node, which
executes it and appends a `ToolMessage` to state, then routes back to `agent` so the
model can see the tool's result and decide what to do next.

## 6. Invoke it

```python
result = graph.invoke({"messages": [("user", "What's the weather in Tokyo?")]})
for m in result["messages"]:
    m.pretty_print()
```

## Shortcut: prebuilt ReAct agent

For this common pattern, LangGraph ships a prebuilt constructor so you don't have to
wire the graph manually:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model, tools)
result = agent.invoke({"messages": [("user", "What's the weather in Tokyo?")]})
```
