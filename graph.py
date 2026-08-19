from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import load_hydrology_semantic_query_settings
from .nodes import (
    HydrologySemanticQueryServices,
    HydrologySemanticQueryState,
    after_catalog,
    after_execution,
    after_generation,
    after_intent,
    after_recovery,
    after_retrieval,
    after_validation,
    make_catalog_prepare_node,
    make_execution_node,
    make_finalize_node,
    make_generation_node,
    make_recovery_node,
    make_retrieval_intent_node,
    make_retrieval_node,
    make_validation_node,
)


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
    graph = HydrologySemanticQueryGraph(
        HydrologySemanticQueryState,
        recursion_limit=8 * (settings.max_retries + 2) + 8,
    )
    graph.add_node("prepare_catalog", make_catalog_prepare_node(services))
    graph.add_node("understand_query", make_retrieval_intent_node(runtime, services))
    graph.add_node("retrieve_context", make_retrieval_node(services))
    graph.add_node("generate_semantic_query", make_generation_node(runtime, services))
    graph.add_node("validate_semantic_query", make_validation_node(services))
    graph.add_node("execute_cube", make_execution_node(services))
    graph.add_node("recover", make_recovery_node(services))
    graph.add_node("finalize_result", make_finalize_node(runtime, services))
    graph.add_edge(START, "prepare_catalog")
    graph.add_conditional_edges(
        "prepare_catalog",
        after_catalog,
        {"understand": "understand_query", "finish": "finalize_result"},
    )
    graph.add_conditional_edges(
        "understand_query",
        after_intent,
        {"retrieve": "retrieve_context", "finish": "finalize_result"},
    )
    graph.add_conditional_edges(
        "retrieve_context",
        after_retrieval,
        {"generate": "generate_semantic_query", "finish": "finalize_result"},
    )
    graph.add_conditional_edges(
        "generate_semantic_query",
        after_generation,
        {"validate": "validate_semantic_query", "recover": "recover"},
    )
    graph.add_conditional_edges(
        "validate_semantic_query",
        after_validation,
        {"execute": "execute_cube", "recover": "recover"},
    )
    graph.add_conditional_edges(
        "execute_cube",
        after_execution,
        {"finish": "finalize_result", "recover": "recover"},
    )
    graph.add_conditional_edges(
        "recover",
        after_recovery,
        {"retry": "generate_semantic_query", "finish": "finalize_result"},
    )
    graph.add_edge("finalize_result", END)
    return graph
