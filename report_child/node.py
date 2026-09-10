from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.messages import stringify_message_content

from .models import (
    ChartPlanningResponse,
    QueryExecutionRecord,
    RenderedChart,
    ReportTask,
    SemanticQueryResult,
    StepStatus,
    TaskDataProfile,
    TaskExecutionResult,
    TaskExecutionStatus,
)
from .prompts import CHART_PLANNER_SYSTEM_PROMPT, report_generator_system_prompt
from .report import (
    REPORT_FAILURE_WARNING,
    assign_chart_ids,
    build_report_outputs,
    build_task_data_profiles,
    ensure_chart_references,
    parse_chart_planning_response,
    report_task_to_markdown,
    validate_chart_planning_response,
)
from .runtime import build_step
from .state import ReportAgentState
from .tool_call_parser import contains_internal_protocol

logger = logging.getLogger("uvicorn.error")


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


def successful_datasets(
    task_results: Sequence[TaskExecutionResult],
) -> list[tuple[str, str, SemanticQueryResult]]:
    return [
        (item.task.task_id, item.task.objective, _dataset(item.query_record))
        for item in task_results
        if item.status == TaskExecutionStatus.SUCCESS and item.query_record is not None
    ]


def chart_planner_response_format() -> dict[str, Any]:
    schema = ChartPlanningResponse.model_json_schema()

    def make_strict(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                value["required"] = list(properties)
                value["additionalProperties"] = False
            for child in value.values():
                make_strict(child)
        elif isinstance(value, list):
            for child in value:
                make_strict(child)

    make_strict(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hydrology_chart_planning",
            "strict": True,
            "schema": schema,
        },
    }


def _planner_request(
    report_task: ReportTask,
    profiles: Sequence[TaskDataProfile],
) -> dict[str, Any]:
    return {
        "user_question": report_task.original_question,
        "report_sections": [
            section.model_dump(mode="json")
            for section in report_task.sections
        ],
        "chart_budget": {"per_task": 4, "global": 8},
        "tasks": [profile.model_dump(mode="json") for profile in profiles],
    }


async def _invoke_chart_planner(
    runtime,
    messages: list[Any],
) -> tuple[str, ChartPlanningResponse, list[dict[str, str]]]:
    model = runtime.get_chat_model(streaming=False).bind(
        response_format=chart_planner_response_format(),
        extra_body={"enable_thinking": False},
        max_tokens=5000,
    )
    response = await model.ainvoke(messages, config={"callbacks": []})
    raw = stringify_message_content(response.content).strip()
    planning, errors = parse_chart_planning_response(raw)
    return raw, planning, errors


def make_chart_planner_node(runtime):
    async def plan(state: ReportAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        report_task = state["report_task"]
        datasets = successful_datasets(report_task.task_results)
        profiles = build_task_data_profiles(datasets)
        messages: list[Any] = [
            SystemMessage(content=CHART_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(
                _planner_request(report_task, profiles),
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )),
        ]
        raw = ""
        planning = ChartPlanningResponse()
        errors: list[dict[str, str]] = []
        warnings: list[str] = []
        try:
            raw, planning, errors = await _invoke_chart_planner(runtime, messages)
            status = StepStatus.SUCCESS
            summary = None
        except Exception as exc:
            status = StepStatus.FAILED
            summary = str(exc)[:1000]
            errors = [{"source_task_id": "*", "error": f"图表规划调用失败：{summary}"}]
            logger.warning("hydrology chart planner failed: attempt=1 error=%s", summary)
        candidate_count = sum(len(task.charts) for task in planning.tasks)
        logger.info(
            "hydrology chart planner: attempt=1 candidates=%s parse_errors=%s",
            candidate_count,
            len(errors),
        )
        step = build_step(
            "chart_planner",
            started,
            attempt=1,
            status=status,
            summary=summary,
            metadata={
                "task_count": len(datasets),
                "candidate_count": candidate_count,
                "parse_error_count": len(errors),
            },
        )
        return {
            "data_profiles": profiles,
            "chart_planning_response": planning,
            "planner_messages": messages,
            "planner_raw_response": raw,
            "planner_errors": errors,
            "planner_attempts": 1,
            "rendered_charts": [],
            "warnings": warnings,
            "steps": [step],
        }

    return plan


def _repair_task_ids(
    errors: Sequence[dict[str, str]],
    profiles: Sequence[TaskDataProfile],
) -> set[str]:
    all_ids = {profile.source_task_id for profile in profiles}
    requested = {
        error["source_task_id"]
        for error in errors
        if error.get("source_task_id") in all_ids
    }
    if any(error.get("source_task_id") == "*" for error in errors):
        return all_ids
    return requested


def _repair_request(
    profiles: Sequence[TaskDataProfile],
    errors: Sequence[dict[str, str]],
    accepted: Sequence[RenderedChart],
    repair_ids: set[str],
) -> dict[str, Any]:
    accepted_counts = Counter(chart.evidence.source_task_id for chart in accepted)
    return {
        "instruction": "只修正列出的无效或缺失计划，不要重写已保留图表。",
        "validation_errors": list(errors),
        "remaining_budget": {
            "global": max(0, 8 - len(accepted)),
            "per_task": {
                task_id: max(0, 4 - accepted_counts[task_id])
                for task_id in sorted(repair_ids)
            },
        },
        "retained_charts": [
            {
                "source_task_id": chart.evidence.source_task_id,
                "title": chart.plan.title,
                "chart_type": chart.plan.chart_type.value,
            }
            for chart in accepted
        ],
        "tasks": [
            profile.model_dump(mode="json")
            for profile in profiles
            if profile.source_task_id in repair_ids
        ],
    }


def make_validate_render_node(runtime):
    async def validate_render(state: ReportAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        report_task = state["report_task"]
        datasets = successful_datasets(report_task.task_results)
        profiles = state.get("data_profiles", [])
        first, validation_errors = validate_chart_planning_response(
            state.get("chart_planning_response", ChartPlanningResponse()),
            datasets,
            profiles,
        )
        first_errors = [*state.get("planner_errors", []), *validation_errors]
        for error in first_errors:
            logger.info(
                "hydrology chart plan validation: attempt=1 task_id=%s error=%s",
                error.get("source_task_id"),
                error.get("error"),
            )
        repair_ids = _repair_task_ids(first_errors, profiles)
        attempts = state.get("planner_attempts", 1)
        final_errors: list[dict[str, str]] = []
        accepted = list(first)
        raw = state.get("planner_raw_response", "")
        messages = list(state.get("planner_messages", []))
        if first_errors and repair_ids and len(accepted) < 8:
            correction = _repair_request(profiles, first_errors, accepted, repair_ids)
            correction_messages = [
                *messages,
                AIMessage(content=raw or json.dumps({"tasks": []})),
                HumanMessage(content=json.dumps(
                    correction,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                )),
            ]
            attempts = 2
            try:
                _, repaired_response, parse_errors = await _invoke_chart_planner(
                    runtime,
                    correction_messages,
                )
                logger.info(
                    "hydrology chart planner: attempt=2 candidates=%s parse_errors=%s",
                    sum(len(task.charts) for task in repaired_response.tasks),
                    len(parse_errors),
                )
                repaired, repaired_errors = validate_chart_planning_response(
                    repaired_response,
                    datasets,
                    profiles,
                    existing=accepted,
                    required_task_ids=repair_ids,
                )
                accepted.extend(repaired)
                final_errors = [*parse_errors, *repaired_errors]
            except Exception as exc:
                final_errors = [{
                    "source_task_id": "*",
                    "error": f"图表修正规划调用失败：{str(exc)[:1000]}",
                }]
        elif first_errors:
            final_errors = first_errors
        charts = assign_chart_ids(accepted, datasets)
        warnings = list(state.get("warnings", []))
        for error in final_errors:
            warning = f"图表计划已丢弃（{error['source_task_id']}）：{error['error']}"
            if warning not in warnings:
                warnings.append(warning)
            logger.warning("hydrology chart plan rejected: %s", warning)
        for chart in charts:
            evidence = chart.evidence
            categories = {
                point.get("name")
                for series in chart.series_data
                for point in series.get("data", [])
            }
            logger.info(
                "hydrology chart rendered: chart_id=%s task_id=%s unit=%s categories=%s series=%s time_points=%s points=%s truncated=%s",
                evidence.chart_id,
                evidence.source_task_id,
                evidence.unit,
                len(categories),
                len(chart.series_data),
                len(categories) if evidence.chart_type.value == "LINE" else 0,
                evidence.displayed_point_count,
                evidence.truncated,
            )
        logger.info(
            "hydrology chart planning complete: attempts=%s candidates=%s valid=%s rejected=%s chart_ids=%s",
            attempts,
            sum(len(task.charts) for task in state.get("chart_planning_response", ChartPlanningResponse()).tasks),
            len(charts),
            len(final_errors),
            [chart.evidence.chart_id for chart in charts],
        )
        step = build_step(
            "chart_validate_render",
            started,
            attempt=attempts,
            status=StepStatus.SUCCESS if not final_errors else StepStatus.SKIPPED,
            summary="；".join(error["error"] for error in final_errors)[:1000] or None,
            metadata={
                "planning_attempts": attempts,
                "valid_chart_count": len(charts),
                "rejected_count": len(final_errors),
                "chart_ids": [chart.evidence.chart_id for chart in charts],
            },
        )
        return {
            "rendered_charts": charts,
            "planner_attempts": attempts,
            "planner_errors": final_errors,
            "warnings": warnings,
            "steps": [*state.get("steps", []), step],
        }

    return validate_render


async def _generate_report_markdown(
    runtime,
    report_task: ReportTask,
    datasets: Sequence[tuple[str, str, SemanticQueryResult]],
    charts: Sequence[RenderedChart],
) -> str:
    messages = [
        SystemMessage(content=report_generator_system_prompt(has_charts=bool(charts))),
        HumanMessage(content=(
            f"用户问题：{report_task.original_question}\n\n"
            f"{report_task_to_markdown(report_task, datasets, charts)}"
        )),
    ]
    model = runtime.get_chat_model(streaming=True).bind(
        extra_body={"enable_thinking": False},
    )
    response = await model.ainvoke(messages, config={"callbacks": []})
    report = stringify_message_content(response.content).strip()
    if not report:
        raise ValueError("报告模型返回了空响应")
    if contains_internal_protocol(report):
        raise ValueError("报告模型返回了内部工具协议")
    return report


def make_report_generate_node(runtime):
    async def generate(state: ReportAgentState) -> dict[str, Any]:
        started = time.perf_counter()
        report_task: ReportTask = state["report_task"]
        datasets = successful_datasets(report_task.task_results)
        charts = state.get("rendered_charts", [])
        warnings = list(state.get("warnings", []))
        failure_summary: str | None = None
        try:
            answer = await _generate_report_markdown(
                runtime,
                report_task,
                datasets,
                charts,
            )
            status = StepStatus.SUCCESS
        except Exception as exc:
            answer = state.get("fallback_answer", "")
            if REPORT_FAILURE_WARNING not in warnings:
                warnings.append(REPORT_FAILURE_WARNING)
            failure_summary = str(exc)[:1000]
            status = StepStatus.FAILED
        answer = ensure_chart_references(answer, charts)
        outputs = build_report_outputs(datasets, charts)
        step = build_step(
            "report_generate",
            started,
            attempt=1,
            status=status,
            summary=failure_summary,
            metadata={
                "task_count": len(datasets),
                "chart_count": len(charts),
                "chart_ids": [chart.evidence.chart_id for chart in charts],
                "output_count": len(outputs),
            },
        )
        return {
            "answer": answer,
            "warnings": warnings,
            "steps": [*state.get("steps", []), step],
            "outputs": outputs,
        }

    return generate


__all__ = [
    "chart_planner_response_format",
    "make_chart_planner_node",
    "make_report_generate_node",
    "make_validate_render_node",
    "successful_datasets",
]
