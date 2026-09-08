"""水文语义查询场景子图注册入口。"""

from app.agents.scenarios.cqccri_smart_query.subgraph_registry import (
    SceneMemoryMode,
    register_scene_agent,
)

from .graph import build_hydrology_semantic_query_graph
from .state import HYDROLOGY_SEMANTIC_QUERY_SCENE_ID

register_scene_agent(
    HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
    build_hydrology_semantic_query_graph,
    checkpoint_ns="HYDROLOGY_SEMANTIC_QUERY_agent",
    state_version=1,
    memory_mode=SceneMemoryMode.SCENE_LOCAL,
)


__all__ = ["build_hydrology_semantic_query_graph"]
