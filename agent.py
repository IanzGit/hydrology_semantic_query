from app.agents.base import AgentDefinition

from .graph import build_hydrology_semantic_query_graph
from .query_child.config import HYDROLOGY_SEMANTIC_QUERY_ID

hydrology_semantic_query_definition = AgentDefinition(
    app_id=HYDROLOGY_SEMANTIC_QUERY_ID,
    name="水文语义查询智能体",
    description=(
        "通过 Plan-and-Execute 编排查询与报告子 Agent，基于受治理语义层完成水文综合分析。"
    ),
    build_graph=build_hydrology_semantic_query_graph,
    enabled_by_default=True,
    tags=("hydrology", "semantic-layer"),
)
