from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .node import (
    make_query_finalize_node,
    make_query_prepare_node,
    make_query_react_node,
)
from .runtime import HydrologySemanticQueryServices
from .state import QueryAgentState
from .tools import ALL_TOOL_NAMES, build_hydrology_semantic_query_tools
from .tools.common import tool_input_error


class HydrologyQueryAgentGraph(StateGraph):
    def __init__(self, state_schema, *, recursion_limit: int) -> None:
        super().__init__(state_schema)
        self.recursion_limit = recursion_limit

    def compile(self, *args, **kwargs):
        return super().compile(*args, **kwargs).with_config(
            {"recursion_limit": self.recursion_limit}
        )


def build_query_agent_graph(
    runtime,
    services: HydrologySemanticQueryServices,
) -> StateGraph:
    tools = build_hydrology_semantic_query_tools(services)
    graph = HydrologyQueryAgentGraph(
        QueryAgentState,
        recursion_limit=2 * services.settings.max_agent_iterations + 5,
    )

    def after_prepare(state: QueryAgentState) -> str:
        return "finalize" if state.get("outcome") is not None else "agent"

    def after_agent(state: QueryAgentState) -> str:
        messages = state.get("messages", [])
        if messages and getattr(messages[-1], "tool_calls", None):
            return "tools"
        return "finalize"

    def after_tools(state: QueryAgentState) -> str:
        if state.get("last_tool_terminal") or state.get("query_count", 0) >= 1:
            return "finalize"
        return "agent"

    graph.add_node("prepare", make_query_prepare_node(services))
    graph.add_node(
        "agent",
        make_query_react_node(runtime, services, tools, ALL_TOOL_NAMES),
    )
    graph.add_node(
        "tools",
        ToolNode(tools, name="query_tools", handle_tool_errors=tool_input_error),
    )
    graph.add_node("finalize", make_query_finalize_node())
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare",
        after_prepare,
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "agent",
        after_agent,
        {"tools": "tools", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "tools",
        after_tools,
        {"agent": "agent", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph


__all__ = ["build_query_agent_graph"]
