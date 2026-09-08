from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .node import (
    make_report_analysis_node,
    make_report_narrative_node,
    make_report_render_node,
)
from .state import ReportAgentState


def build_report_agent_graph(
    runtime,
    timezone: str,
) -> StateGraph:
    graph = StateGraph(ReportAgentState)
    graph.add_node("analysis", make_report_analysis_node(timezone))
    graph.add_node("narrative", make_report_narrative_node(runtime))
    graph.add_node("render", make_report_render_node())
    graph.add_edge(START, "analysis")
    graph.add_edge("analysis", "narrative")
    graph.add_edge("narrative", "render")
    graph.add_edge("render", END)
    return graph


__all__ = ["build_report_agent_graph"]
