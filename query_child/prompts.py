from __future__ import annotations

import json
from typing import Any

from .runtime import (
    HydrologySemanticQueryServices,
    request_data,
)
from .state import QueryAgentState

SEMANTIC_QUERY_RULES = """\
Semantic Context 是相关语义目录上下文，不是成员绑定、候选白名单或最终路由结果。你负责决定实际使用的 View、Cube、成员、过滤、聚合、排序和结果形态。

============================
规则：
============================
1. query_mode 只能是 view 或 cube。View Mode 必须且只能使用一个 View；Cube Mode 只能使用 1 至 4 个 Cube，禁止混合 View 与 Cube namespace。
2. 只能使用受治理 Cube 语义目录中的模型和成员，成员使用 model.member 全名。Semantic Context 可能只是相关目录片段，不是可用项的闭集；不得把检索是否命中当成白名单，也不得使用原始数据库表和列。
3. 必须保持业务实体归属。设备名称、设备编码、安装位置等设备属性不能被同名或相似的传感器属性替代；过滤值必须绑定到用户指定实体的字段。
4. 根据完整查询任务比较可用 View 和 Cube 组合，选择业务语义最直接且能完整覆盖任务的方案；多个 Cube 必须位于同一 connected_component。检索分数只表示上下文相关性，不决定最终模型或成员。
5. 明细查询的 measures 必须为空、dimensions 或 time_dimensions 必须非空且 ungrouped=true。聚合查询必须包含 measure 且 ungrouped=false；任务要求的分组字段保留在 dimensions 或 time_dimensions 中。
6. 任务明确列出的结果字段必须逐项体现在 dimensions、measures 或 time_dimensions 中，不得用其他字段替代。
7. 默认展示字段：仅给出业务对象而未列出结果字段时，必须完整使用模型的 default_projection作为默认展示字段，不得自行删减、替换或遗漏其中成员；只有模型未提供 default_projection 时，才选择少量核心展示字段。
8. 时间范围放入 time_dimensions 的 date_range；自然日结束日按包含理解。时间字段作为普通明细列时可以放入 dimensions。
9. segments 只用于语义目录明确声明的受治理口径。View 固定业务口径不得重复添加；Cube 不得推断任务未要求的默认过滤。
10. filters 可使用 and/or，逻辑组最多嵌套两层；同一显式逻辑组不得混合 measure 与 dimension，顶层独立过滤器可以分别使用二者。
11. filter operator 只能是 equals、notEquals、contains、notContains、startsWith、notStartsWith、endsWith、notEndsWith、gt、gte、lt、lte、set、notSet、inDateRange、notInDateRange、beforeDate、beforeOrOnDate、afterDate、afterOrOnDate。
12. filter values 必须是标量数组；布尔值使用 "1" 或 "0"；set/notSet 不得携带 values。
13. order 是有序数组，每项为 {"member":"model.member","direction":"asc|desc"}。“最新”使用相应时间字段降序并设置 limit=1；TopN 保留任务指定的次级排序。
14. 多个业务源组合条件使用 or 包含多个 and 表达。
15. 仅当任务明确要求最新记录、TopN、分页或指定数量时设置 limit；其他查询不设置 limit，返回全部匹配数据。offset 默认为 0。

JSON 字段固定为 query_mode、models、measures、dimensions、segments、filters、time_dimensions、order、limit、offset、ungrouped。
""".strip()


def _execution_context_payload(state: QueryAgentState) -> dict[str, Any]:
    context = state.get("execution_context")
    if context is None:
        return {}
    dependency_ids = set(context.current_task.depends_on)
    history: list[dict[str, Any]] = []
    for result in context.completed_results:
        record = result.query_record
        item = {
            "task_id": result.task.task_id,
            "objective": result.task.objective,
            "status": result.status.value,
            "outcome": result.outcome.value if result.outcome else None,
            "summary": result.summary,
            "row_count": record.row_count if record else 0,
            "error": (
                {
                    "stage": result.error.stage,
                    "code": result.error.code,
                    "kind": result.error.kind.value,
                    "retryable": result.error.retryable,
                }
                if result.error
                else None
            ),
        }
        if result.task.task_id in dependency_ids and record is not None:
            item.update({
                "columns": [
                    column.model_dump(mode="json") for column in record.columns
                ],
                "rows": record.rows,
            })
        history.append(item)
    return {
        "original_question": context.original_question,
        "standalone_question": context.standalone_question,
        "plan": [task.model_dump(mode="json") for task in context.plan],
        "history": history,
        "current_task": context.current_task.model_dump(mode="json"),
    }


def query_agent_system_prompt(
    state: QueryAgentState,
    services: HydrologySemanticQueryServices,
    *,
    final_only: bool,
) -> str:
    request = request_data(state, services.settings)
    task = state.get("current_task")
    objective = task.objective if task is not None else request["question"]
    execution_context = json.dumps(
        _execution_context_payload(state),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    final_instruction = (
        "当前 Observation 已终止本任务，禁止继续调用工具，直接结束。"
        if final_only
        else "需要目录或数据事实时必须调用工具。修正查询错误后继续执行，任务完成后直接结束。"
    )
    return f"""\
你是严格 Plan-and-Execute 架构中的水文语义查询子 Agent。你只负责完成主控 Agent 完整计划中的当前查询任务，不规划或修改其他业务查询，不生成报告，也不接受原始 SQL。

============================
可用工具：
============================
1. search_semantic_catalog(query, limit?)：检索相关 Cube、View 和 member。首次执行查询前必须成功调用至少一次。
2. run_semantic_query(semantic_query, query_goal?)：统一校验、Cube /sql 编译预检和 /load 执行。

每轮最多调用一个工具。一个任务只能形成一次成功或无数据的业务查询；工具返回成功或 no_data 后必须结束。认证、网络、超时和系统错误是终止 Observation。可修正的目录、成员、校验或 Cube 400 错误应根据结构化 Observation 检索并修正。

执行上下文包含原始问题、完整计划、历史步骤结果和当前步骤。直接依赖任务携带全部查询数据。

============================
不支持原生工具调用时，严格使用：
============================
Action: 工具名
Action Input: {{"参数名": 参数值}}

{SEMANTIC_QUERY_RULES}

============================
执行上下文：
============================
{execution_context}

============================
当前查询任务：
============================
{objective}

{final_instruction}""".strip()


__all__ = ["SEMANTIC_QUERY_RULES", "query_agent_system_prompt"]
