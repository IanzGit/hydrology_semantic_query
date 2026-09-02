from __future__ import annotations

import math
import time
from collections.abc import Sequence
from datetime import date, datetime
from itertools import combinations
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

from .models import (
    QueryExecutionRecord,
    ReportAnalysis,
    ReportAnalysisMethod,
    ReportFact,
    ReportFactCategory,
    ReportLimitation,
    ReportTask,
    SemanticColumn,
    SemanticQueryResult,
    StepStatus,
    TaskExecutionResult,
    TaskExecutionStatus,
)
from .report import REPORT_FAILURE_WARNING, analyze_result, generate_narrative
from .report_analysis import build_result_profile, finite_number
from .report_rendering import (
    build_result_outputs,
    compose_multi_task_structured_report,
)
from .runtime import build_step
from .state import ReportAgentState


def _dataset(record: QueryExecutionRecord) -> SemanticQueryResult:
    return SemanticQueryResult(
        outcome=record.outcome,
        semantic_query=record.semantic_query,
        columns=record.columns,
        rows=record.rows,
        row_count=record.row_count,
        attempts=record.attempt,
        query_count=1,
        query_history=[record],
        compiled_sql=record.compiled_sql,
        compiled_params=record.compiled_params,
        query_mode=record.semantic_query.query_mode,
        selected_models=record.selected_models,
    )


def _successful_datasets(
    task_results: Sequence[TaskExecutionResult],
) -> list[tuple[str, str, SemanticQueryResult]]:
    return [
        (
            item.task.task_id,
            item.task.objective,
            _dataset(item.query_record),
        )
        for item in task_results
        if item.status == TaskExecutionStatus.SUCCESS
        and item.query_record is not None
    ]


def _aggregate_result(
    result: SemanticQueryResult,
    datasets: Sequence[tuple[str, str, SemanticQueryResult]],
) -> SemanticQueryResult:
    if len(datasets) <= 1:
        return datasets[0][2].model_copy(deep=True) if datasets else result.model_copy(deep=True)
    columns = [
        SemanticColumn(
            name="query_context.task_id",
            title="任务ID",
            data_type="string",
            member_type="dimension",
        ),
        SemanticColumn(
            name="query_context.query_goal",
            title="查询目标",
            data_type="string",
            member_type="dimension",
        ),
    ]
    seen = {column.name for column in columns}
    rows: list[dict[str, Any]] = []
    selected_models: list[str] = []
    for task_id, objective, dataset in datasets:
        for model in dataset.selected_models:
            if model not in selected_models:
                selected_models.append(model)
        for column in dataset.columns:
            if column.name not in seen:
                seen.add(column.name)
                columns.append(column)
        rows.extend({
            "query_context.task_id": task_id,
            "query_context.query_goal": objective,
            **row,
        } for row in dataset.rows)
    return result.model_copy(update={
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "selected_models": selected_models,
        "presentation": None,
    }, deep=True)


def _prefix_analysis(
    task_id: str,
    objective: str,
    analysis: ReportAnalysis,
) -> tuple[list[ReportFact], list[ReportLimitation]]:
    facts = [
        fact.model_copy(update={
            "fact_id": f"{task_id}-{fact.fact_id}",
            "metadata": {
                **fact.metadata,
                "task_id": task_id,
                "query_goal": objective,
            },
        }, deep=True)
        for fact in analysis.facts
    ]
    limitations = [
        limitation.model_copy(update={
            "code": f"{task_id}-{limitation.code}",
            "message": f"任务“{objective}”：{limitation.message}",
        })
        for limitation in analysis.limitations
    ]
    return facts, limitations


def _time_key(value: Any, timezone: str, granularity: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                return date.fromisoformat(text).isoformat()
            except ValueError:
                return None
    else:
        return None
    zone = ZoneInfo(timezone)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    else:
        parsed = parsed.astimezone(zone)
    if granularity == "year":
        return f"{parsed.year:04d}"
    if granularity == "quarter":
        return f"{parsed.year:04d}-Q{(parsed.month - 1) // 3 + 1}"
    if granularity == "month":
        return f"{parsed.year:04d}-{parsed.month:02d}"
    if granularity == "week":
        iso_year, iso_week, _ = parsed.isocalendar()
        return f"{iso_year:04d}-W{iso_week:02d}"
    if granularity == "day":
        return parsed.date().isoformat()
    if granularity == "hour":
        parsed = parsed.replace(minute=0, second=0, microsecond=0)
    elif granularity == "minute":
        parsed = parsed.replace(second=0, microsecond=0)
    elif granularity == "second":
        parsed = parsed.replace(microsecond=0)
    return parsed.isoformat()


def _query_granularity(record: QueryExecutionRecord) -> str | None:
    granularities = {
        item.granularity
        for item in record.semantic_query.time_dimensions
        if item.granularity
    }
    return next(iter(granularities)) if len(granularities) == 1 else None


def _series(
    record: QueryExecutionRecord,
    timezone: str,
    granularity: str,
) -> tuple[dict[str, float] | None, str | None, str | None, str | None]:
    dataset = _dataset(record)
    profile = build_result_profile(dataset)
    if not profile.primary_time:
        return None, None, None, "缺少唯一主时间字段"
    if len(profile.measure_fields) != 1:
        return None, None, None, "缺少唯一主数值指标"
    measure = profile.measure_fields[0]
    values: dict[str, float] = {}
    for row in record.rows:
        key = _time_key(
            row.get(profile.primary_time),
            timezone,
            granularity,
        )
        number = finite_number(row.get(measure))
        if key is None or number is None:
            continue
        if key in values:
            return None, None, None, "同一时间点存在多条记录，无法无歧义配对"
        values[key] = number
    if not values:
        return None, None, None, "没有可配对的时间序列数值"
    title = next(
        (column.title for column in record.columns if column.name == measure),
        measure,
    )
    return values, measure, title, None


def _pearson(pairs: list[tuple[float, float]]) -> float | None:
    xs = [left for left, _ in pairs]
    ys = [right for _, right in pairs]
    mean_x = fmean(xs)
    mean_y = fmean(ys)
    numerator = sum((value - mean_x) * (other - mean_y) for value, other in pairs)
    denominator = math.sqrt(
        sum((value - mean_x) ** 2 for value in xs)
        * sum((value - mean_y) ** 2 for value in ys)
    )
    return numerator / denominator if denominator else None


def _correlation_evidence(
    report_task: ReportTask,
    timezone: str,
) -> tuple[list[ReportFact], list[ReportLimitation]]:
    records = {
        item.task.task_id: item.query_record
        for item in report_task.task_results
        if item.status == TaskExecutionStatus.SUCCESS
        and item.query_record is not None
    }
    facts: list[ReportFact] = []
    limitations: list[ReportLimitation] = []
    fact_number = 0
    for section in report_task.sections:
        if ReportAnalysisMethod.CORRELATION not in section.analysis_methods:
            continue
        available_ids = [task_id for task_id in section.source_task_ids if task_id in records]
        if len(available_ids) < 2:
            limitations.append(ReportLimitation(
                code=f"{section.section_id}-insufficient_sources",
                message=f"章节“{section.title}”缺少至少两个成功的来源任务，无法计算相关性。",
            ))
            continue
        for left_id, right_id in combinations(available_ids, 2):
            left = records[left_id]
            right = records[right_id]
            left_granularity = _query_granularity(left)
            right_granularity = _query_granularity(right)
            if (
                not left_granularity
                or not right_granularity
                or left_granularity != right_granularity
            ):
                limitations.append(ReportLimitation(
                    code=f"{section.section_id}-granularity-{left_id}-{right_id}",
                    message=f"任务 {left_id} 与 {right_id} 的时间粒度不一致，未计算相关性。",
                ))
                continue
            left_values, left_field, left_title, left_error = _series(
                left,
                timezone,
                left_granularity,
            )
            right_values, right_field, right_title, right_error = _series(
                right,
                timezone,
                right_granularity,
            )
            if left_error or right_error:
                limitations.append(ReportLimitation(
                    code=f"{section.section_id}-series-{left_id}-{right_id}",
                    message=(
                        f"任务 {left_id} 与 {right_id} 无法确定性对齐："
                        f"{left_error or right_error}。"
                    ),
                ))
                continue
            shared = sorted(set(left_values or {}) & set(right_values or {}))
            if len(shared) < 8:
                limitations.append(ReportLimitation(
                    code=f"{section.section_id}-sample-{left_id}-{right_id}",
                    message=(
                        f"任务 {left_id} 与 {right_id} 只有 {len(shared)} 个同粒度有效配对样本，"
                        "少于 8 个，未计算相关性。"
                    ),
                ))
                continue
            pairs = [
                ((left_values or {})[key], (right_values or {})[key])
                for key in shared
            ]
            coefficient = _pearson(pairs)
            if coefficient is None:
                limitations.append(ReportLimitation(
                    code=f"{section.section_id}-constant-{left_id}-{right_id}",
                    message=f"任务 {left_id} 与 {right_id} 的序列缺少有效方差，未计算相关性。",
                ))
                continue
            fact_number += 1
            facts.append(ReportFact(
                fact_id=f"cross-{fact_number:03d}",
                category=ReportFactCategory.CORRELATION,
                title=f"{left_title}与{right_title}相关性",
                display_text=(
                    f"在 {len(pairs)} 个同粒度时间配对样本上，“{left_title}”与“{right_title}”"
                    f"的 Pearson 相关系数为 {coefficient:.6g}；相关性不代表因果关系。"
                ),
                value={"sample_count": len(pairs), "coefficient": coefficient},
                evidence_fields=[left_field or "", right_field or ""],
                metadata={
                    "source_task_ids": [left_id, right_id],
                    "section_id": section.section_id,
                    "granularity": left_granularity,
                },
            ))
    return facts, limitations


def build_multi_task_analysis(
    report_task: ReportTask,
    aggregate_result: SemanticQueryResult,
    timezone: str,
) -> ReportAnalysis:
    facts: list[ReportFact] = []
    limitations: list[ReportLimitation] = []
    for task_id, objective, dataset in _successful_datasets(report_task.task_results):
        task_facts, task_limitations = _prefix_analysis(
            task_id,
            objective,
            analyze_result(dataset),
        )
        facts.extend(task_facts)
        limitations.extend(task_limitations)
    for item in report_task.task_results:
        if item.status == TaskExecutionStatus.SUCCESS:
            continue
        limitations.append(ReportLimitation(
            code=f"{item.task.task_id}-task-{item.status.value}",
            message=f"任务“{item.task.objective}”{item.summary or '未提供有效数据。'}",
        ))
    cross_facts, cross_limitations = _correlation_evidence(report_task, timezone)
    facts.extend(cross_facts)
    limitations.extend(cross_limitations)
    return ReportAnalysis(
        profile=build_result_profile(aggregate_result),
        facts=facts,
        limitations=limitations,
    )


def make_report_analysis_node(timezone: str):
    async def analyze(state: ReportAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        report_task = state["report_task"]
        datasets = _successful_datasets(report_task.task_results)
        aggregate = _aggregate_result(state["result"], datasets)
        analysis = build_multi_task_analysis(
            report_task,
            aggregate,
            timezone,
        )
        step = build_step(
            "report_analysis",
            started,
            attempt=1,
            status=StepStatus.SUCCESS,
            metadata={
                "fact_count": len(analysis.facts),
                "task_count": len(datasets),
            },
        )
        return {
            "aggregate_result": aggregate,
            "datasets": datasets,
            "analysis": analysis,
            "steps": [step],
        }

    return analyze


def make_report_narrative_node(runtime):
    async def narrative(state: ReportAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        warnings = list(state.get("warnings", []))
        steps = list(state.get("steps", []))
        try:
            draft = await generate_narrative(
                runtime,
                state["report_task"].original_question,
                state["analysis"],
                state["report_task"].sections,
            )
            steps.append(build_step(
                "report_narrative",
                started,
                attempt=1,
                status=StepStatus.SUCCESS,
                metadata={"insight_count": len(draft.insights)},
            ))
            return {"narrative": draft, "warnings": warnings, "steps": steps}
        except Exception as exc:
            if REPORT_FAILURE_WARNING not in warnings:
                warnings.append(REPORT_FAILURE_WARNING)
            steps.append(build_step(
                "report_narrative",
                started,
                attempt=1,
                status=StepStatus.FAILED,
                summary=str(exc)[:1000],
            ))
            return {"narrative": None, "warnings": warnings, "steps": steps}

    return narrative


def make_report_render_node():
    async def render(state: ReportAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        steps = list(state.get("steps", []))
        report = compose_multi_task_structured_report(
            state["aggregate_result"],
            state["report_task"].original_question,
            state["analysis"],
            state.get("narrative"),
            state["report_task"].sections,
            state["datasets"],
        )
        steps.append(build_step(
            "report_render",
            started,
            attempt=1,
            status=StepStatus.SUCCESS,
            metadata={
                "output_block_count": sum(
                    len(section.blocks) for section in report.sections
                ),
            },
        ))
        return {
            "report": report,
            "answer": report.summary,
            "steps": steps,
            "stream_outputs": build_result_outputs(
                state["aggregate_result"].model_copy(update={"presentation": report}),
                answer=report.summary,
                question=state["report_task"].original_question,
            ),
        }

    return render


__all__ = [
    "ReportAgentState",
    "build_multi_task_analysis",
    "make_report_analysis_node",
    "make_report_narrative_node",
    "make_report_render_node",
]
