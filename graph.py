from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .contracts import MainAgentAction, QueryOutcome
from .knowledge import load_business_playbooks
from .node import (
    make_finalize_node,
    make_initialize_node,
    make_main_plan_node,
    make_main_replan_node,
    make_query_dispatch_node,
    make_report_dispatch_node,
)
from .query_child import build_query_agent_graph
from .query_child.config import load_hydrology_semantic_query_settings
from .query_child.runtime import HydrologySemanticQueryServices
from .report_child import build_report_agent_graph
from .state import HydrologySemanticQueryState


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
    if services.business_playbooks is None:
        playbooks, warnings = load_business_playbooks()
        services.business_playbooks = playbooks
        services.startup_warnings.extend(warnings)
    query_graph = build_query_agent_graph(runtime, services).compile()
    report_graph = build_report_agent_graph(runtime, settings.timezone).compile()
    graph = HydrologySemanticQueryGraph(
        HydrologySemanticQueryState,
        recursion_limit=2 * settings.max_query_rounds + 10,
    )

    def after_initialize(state: HydrologySemanticQueryState) -> str:
        return "finalize" if state.get("outcome") is not None else "plan"

    def after_main(state: HydrologySemanticQueryState) -> str:
        if state.get("outcome") is not None and not state.get("task_results"):
            return "finalize"
        if state.get("main_action") == MainAgentAction.QUERY:
            return "query_agent"
        return "finalize"

    def after_finalize(state: HydrologySemanticQueryState) -> str:
        result = state.get("result")
        if result is not None and result.outcome == QueryOutcome.SUCCESS:
            return "report_agent"
        return "end"

    graph.add_node("initialize", make_initialize_node(runtime, services))
    graph.add_node("main_plan", make_main_plan_node(runtime, services))
    graph.add_node(
        "query_agent",
        make_query_dispatch_node(query_graph, services),
    )
    graph.add_node("main_replan", make_main_replan_node(runtime, services))
    graph.add_node("finalize", make_finalize_node(services))
    graph.add_node("report_agent", make_report_dispatch_node(report_graph))
    graph.add_edge(START, "initialize")
    graph.add_conditional_edges(
        "initialize",
        after_initialize,
        {"plan": "main_plan", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "main_plan",
        after_main,
        {"query_agent": "query_agent", "finalize": "finalize"},
    )
    graph.add_edge("query_agent", "main_replan")
    graph.add_conditional_edges(
        "main_replan",
        after_main,
        {"query_agent": "query_agent", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "finalize",
        after_finalize,
        {"report_agent": "report_agent", "end": END},
    )
    graph.add_edge("report_agent", END)
    return graph


__all__ = ["build_hydrology_semantic_query_graph"]
