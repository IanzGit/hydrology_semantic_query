from app.agents.base import AgentDefinition

from .config import HYDROLOGY_SEMANTIC_QUERY_ID
from .graph import build_hydrology_semantic_query_graph

hydrology_semantic_query_definition = AgentDefinition(
    app_id=HYDROLOGY_SEMANTIC_QUERY_ID,
    name="水文语义查询智能体",
    description=(
        "通过 ReAct 循环和受治理语义层完成水文问数，支持指标、明细、趋势和 TopN 查询。"
    ),
    build_graph=build_hydrology_semantic_query_graph,
    enabled_by_default=True,
    tags=("hydrology", "semantic-layer"),
)
