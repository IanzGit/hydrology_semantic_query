from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .node import (
    make_chart_planner_node,
    make_report_generate_node,
    make_validate_render_node,
)
from .state import ReportAgentState


def build_report_agent_graph(
    runtime,
) -> StateGraph:
    graph = StateGraph(ReportAgentState)
    graph.add_node("planner", make_chart_planner_node(runtime))
    graph.add_node("validate_render", make_validate_render_node(runtime))
    graph.add_node("report", make_report_generate_node(runtime))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "validate_render")
    graph.add_edge("validate_render", "report")
    graph.add_edge("report", END)
    return graph


__all__ = ["build_report_agent_graph"]
