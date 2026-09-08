from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from langgraph.channels import UntrackedValue

from app.agents.scenarios.cqccri_smart_query.subgraph.common_models import (
    CqccriBaseState,
    SmartQueryInput,
)

from .contracts import (
    ExecutionPlanRevision,
    MainAgentAction,
    QueryExecutionRecord,
    QueryOutcome,
    QueryTask,
    ReportSectionRequirement,
    RetrievalTrace,
    SemanticCatalogMode,
    SemanticColumn,
    SemanticQuery,
    SemanticQueryError,
    SemanticQueryResult,
    StepRecord,
    TaskExecutionResult,
)

HYDROLOGY_SEMANTIC_QUERY_SCENE_ID = "SWJC"
HYDROLOGY_SEMANTIC_QUERY_MEMORY_VERSION = 1


@dataclass(slots=True)
class HydrologySemanticQueryMemory:
    """水文场景 checkpoint 中持久化的版本化记忆。"""

    version: int = HYDROLOGY_SEMANTIC_QUERY_MEMORY_VERSION

    def to_dict(self) -> dict[str, int]:
        """转换为可序列化的记忆字典。"""
        return {"version": self.version}

    @classmethod
    def from_dict(cls, data: Any) -> HydrologySemanticQueryMemory:
        """从 checkpoint 恢复记忆，版本不匹配时安全重置。"""
        if isinstance(data, cls):
            data = data.to_dict()
        if (
            not isinstance(data, dict)
            or data.get("version") != HYDROLOGY_SEMANTIC_QUERY_MEMORY_VERSION
        ):
            return cls()
        return cls(version=HYDROLOGY_SEMANTIC_QUERY_MEMORY_VERSION)


class HydrologySemanticQueryState(CqccriBaseState, total=False):
    """水文语义查询场景 State；业务执行字段仅在本轮图内可见。"""

    # CQCCRI 公共输入和本轮容器不写入长期 checkpoint。
    sence_id: Annotated[str, UntrackedValue]
    smart_query_input: Annotated[SmartQueryInput, UntrackedValue]
    draft: Annotated[dict[str, Any], UntrackedValue]
    plan: Annotated[dict[str, Any], UntrackedValue]
    stream_outputs: Annotated[list[dict[str, Any]], UntrackedValue]

    # 原场景依赖的 AgentState 字段保留为场景私有轮内字段。
    query: Annotated[str, UntrackedValue]
    metadata: Annotated[dict[str, Any], UntrackedValue]
    answer: Annotated[str, UntrackedValue]

    standalone_question: Annotated[str | None, UntrackedValue]
    catalog_mode: Annotated[SemanticCatalogMode | None, UntrackedValue]
    retrieval_trace: Annotated[RetrievalTrace | None, UntrackedValue]
    selected_models: Annotated[list[str], UntrackedValue]
    semantic_query: Annotated[SemanticQuery | None, UntrackedValue]
    compiled_sql: Annotated[str | None, UntrackedValue]
    compiled_params: Annotated[list[Any], UntrackedValue]
    columns: Annotated[list[SemanticColumn], UntrackedValue]
    rows: Annotated[list[dict[str, Any]], UntrackedValue]
    steps: Annotated[list[StepRecord], UntrackedValue]
    warnings: Annotated[list[str], UntrackedValue]
    attempts: Annotated[int, UntrackedValue]
    stage: Annotated[str, UntrackedValue]
    error: Annotated[SemanticQueryError | None, UntrackedValue]
    outcome: Annotated[QueryOutcome | None, UntrackedValue]
    result: Annotated[SemanticQueryResult | None, UntrackedValue]
    query_history: Annotated[list[QueryExecutionRecord], UntrackedValue]
    matched_playbook: Annotated[str | None, UntrackedValue]
    plan_revisions: Annotated[list[ExecutionPlanRevision], UntrackedValue]
    pending_tasks: Annotated[list[QueryTask], UntrackedValue]
    report_sections: Annotated[list[ReportSectionRequirement], UntrackedValue]
    task_results: Annotated[list[TaskExecutionResult], UntrackedValue]
    current_task: Annotated[QueryTask | None, UntrackedValue]
    main_action: Annotated[MainAgentAction | None, UntrackedValue]
    direct_answer: Annotated[str, UntrackedValue]
    dispatch_count: Annotated[int, UntrackedValue]
    querying_blocked: Annotated[bool, UntrackedValue]

    hydrology_semantic_query_memory: HydrologySemanticQueryMemory


__all__ = [
    "HYDROLOGY_SEMANTIC_QUERY_MEMORY_VERSION",
    "HYDROLOGY_SEMANTIC_QUERY_SCENE_ID",
    "HydrologySemanticQueryMemory",
    "HydrologySemanticQueryState",
]
