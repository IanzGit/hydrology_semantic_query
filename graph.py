from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import load_hydrology_semantic_query_settings
from .node import (
    make_initialize_node,
    make_react_agent_node,
    make_react_finalize_node,
)
from .runtime import HydrologySemanticQueryServices, HydrologySemanticQueryState
from .tools import ALL_TOOL_NAMES, build_hydrology_semantic_query_tools
from .tools.common import tool_input_error


class HydrologySemanticQueryGraph(StateGraph):
    def __init__(self, state_schema, *, recursion_limit: int) -> None:
        super().__init__(state_schema)
        self.recursion_limit = recursion_limit

    def compile(self, *args, **kwargs):
        return super().compile(*args, **kwargs).with_config(
            {"recursion_limit": self.recursion_limit}
        )


def build_hydrology_semantic_query_graph(
    runtime,
    services: HydrologySemanticQueryServices | None = None,
) -> StateGraph:
    if services is None:
        settings = load_hydrology_semantic_query_settings()
        services = HydrologySemanticQueryServices(settings)
    else:
        settings = services.settings
    tools = build_hydrology_semantic_query_tools(services)
    graph = HydrologySemanticQueryGraph(
        HydrologySemanticQueryState,
        recursion_limit=2 * settings.max_agent_iterations + 6,
    )

    def after_initialize(state: HydrologySemanticQueryState) -> str:
        return "finish" if state.get("outcome") is not None else "agent"

    def after_agent(state: HydrologySemanticQueryState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "finish"
        last = messages[-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "finish"

    graph.add_node("initialize", make_initialize_node(runtime, services))
    graph.add_node(
        "agent",
        make_react_agent_node(runtime, services, tools, ALL_TOOL_NAMES),
    )
    graph.add_node(
        "tools",
        ToolNode(
            tools,
            name="tools",
            handle_tool_errors=tool_input_error,
        ),
    )
    graph.add_node("finalize", make_react_finalize_node(runtime, services))
    graph.add_edge(START, "initialize")
    graph.add_conditional_edges(
        "initialize",
        after_initialize,
        {"agent": "agent", "finish": "finalize"},
    )
    graph.add_conditional_edges(
        "agent",
        after_agent,
        {"tools": "tools", "finish": "finalize"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph
