from app.agents.base import AgentDefinition

from .graph import build_hydrology_semantic_query_graph

HYDROLOGY_SEMANTIC_QUERY_ID = "hydrology_semantic_query"

hydrology_semantic_query_definition = AgentDefinition(
    app_id=HYDROLOGY_SEMANTIC_QUERY_ID,
    name="水文语义查询智能体",
    description=(
        "通过 Cube 语义层完成自然语言水文问数，支持受治理指标、明细、趋势和 TopN 查询。"
    ),
    build_graph=build_hydrology_semantic_query_graph,
    enabled_by_default=True,
    tags=("hydrology", "semantic-layer", "cube"),
)
