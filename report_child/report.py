from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from app.agents.messages import stringify_message_content

from .models import (
    ReportAnalysis,
    ReportFact,
    ReportNarrativeDraft,
    ReportSectionRequirement,
    SemanticQueryResult,
)
from .prompts import report_narrative_messages
from .report_analysis import analyze_result, build_result_profile
from .report_rendering import (
    build_presentation_plan as _build_presentation_plan,
)
from .report_rendering import (
    build_result_outputs,
    compose_structured_report,
    render_structured_report,
)
from .tool_call_parser import contains_internal_protocol

REPORT_FAILURE_WARNING = "结构化叙事生成失败，已使用确定性分析报告。"
_NUMBER_TOKEN = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?%?")


def report_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hydrology_report_narrative",
            "strict": True,
            "schema": ReportNarrativeDraft.model_json_schema(),
        },
    }


def _fact_packet(
    question: str,
    analysis: ReportAnalysis,
    report_sections: list[ReportSectionRequirement] | None = None,
) -> dict[str, Any]:
    packet = {
        "question": question,
        "facts": [fact.model_dump(mode="json", exclude_none=True) for fact in analysis.facts],
        "limitations": [item.model_dump(mode="json") for item in analysis.limitations],
    }
    if report_sections is not None:
        packet["report_sections"] = [
            section.model_dump(mode="json") for section in report_sections
        ]
    return packet


def _numeric_tokens(text: str) -> set[str]:
    normalized: set[str] = set()
    for token in _NUMBER_TOKEN.findall(text):
        percent = token.endswith("%")
        value = token.removesuffix("%").lstrip("+")
        try:
            value = format(Decimal(value).normalize(), "f")
        except InvalidOperation:
            pass
        normalized.add(f"{value}%" if percent else value)
    return normalized


def _allowed_tokens(facts: list[ReportFact]) -> set[str]:
    tokens: set[str] = set()
    for fact in facts:
        tokens.update(_numeric_tokens(fact.display_text))
        tokens.update(_numeric_tokens(json.dumps(fact.value, ensure_ascii=False, default=str)))
    return tokens


def _validate_text_numbers(text: str, facts: list[ReportFact]) -> None:
    unexpected = _numeric_tokens(text) - _allowed_tokens(facts)
    if unexpected:
        raise ValueError(f"叙事包含未经事实引用支持的定量值: {sorted(unexpected)}")


def validate_narrative(
    raw: str,
    analysis: ReportAnalysis,
    report_sections: list[ReportSectionRequirement] | None = None,
) -> ReportNarrativeDraft:
    if not raw.strip():
        raise ValueError("报告模型返回了空响应")
    if contains_internal_protocol(raw):
        raise ValueError("报告模型返回了内部工具协议")
    draft = ReportNarrativeDraft.model_validate_json(raw)
    facts_by_id = {fact.fact_id: fact for fact in analysis.facts}
    all_text = json.dumps(draft.model_dump(mode="json"), ensure_ascii=False)
    if contains_internal_protocol(all_text):
        raise ValueError("报告叙事包含内部工具协议")
    _validate_text_numbers(f"{draft.title}\n{draft.executive_summary}", analysis.facts)
    valid_section_ids = (
        {section.section_id for section in report_sections}
        if report_sections is not None
        else None
    )
    sections_by_id = (
        {section.section_id: section for section in report_sections}
        if report_sections is not None
        else {}
    )
    for insight in draft.insights:
        if valid_section_ids is not None and insight.section_id not in valid_section_ids:
            raise ValueError(
                f"报告叙事引用了无效章节: {insight.section_id}"
            )
        invalid = [fact_id for fact_id in insight.fact_ids if fact_id not in facts_by_id]
        if invalid:
            raise ValueError(f"报告叙事引用了无效事实: {invalid}")
        referenced = [facts_by_id[fact_id] for fact_id in insight.fact_ids]
        if insight.section_id is not None:
            allowed_sources = set(
                sections_by_id[insight.section_id].source_task_ids
            )
            for fact in referenced:
                fact_sources = {
                    str(source)
                    for source in fact.metadata.get("source_task_ids", [])
                }
                if fact.metadata.get("task_id"):
                    fact_sources.add(str(fact.metadata["task_id"]))
                if allowed_sources and fact_sources.isdisjoint(allowed_sources):
                    raise ValueError(
                        f"报告叙事在章节 {insight.section_id} 引用了其他任务的事实"
                    )
        text = "\n".join([
            insight.title,
            insight.interpretation,
            insight.impact,
            insight.possible_cause,
            insight.recommendation,
        ])
        _validate_text_numbers(text, referenced)
    return draft


async def generate_narrative(
    runtime,
    question: str,
    analysis: ReportAnalysis,
    report_sections: list[ReportSectionRequirement] | None = None,
) -> ReportNarrativeDraft:
    messages = report_narrative_messages(
        _fact_packet(question, analysis, report_sections)
    )
    model = runtime.get_chat_model(streaming=True).bind(
        extra_body={"enable_thinking": False},
        response_format=report_response_format(),
    )
    response = await model.ainvoke(messages, config={"callbacks": []})
    return validate_narrative(
        stringify_message_content(response.content).strip(),
        analysis,
        report_sections,
    )


def build_presentation_plan(result: SemanticQueryResult, question: str):
    return _build_presentation_plan(result, question, build_result_profile(result))


async def generate_report(runtime, question: str, result: SemanticQueryResult) -> str:
    analysis = analyze_result(result)
    narrative = await generate_narrative(runtime, question, analysis)
    report = compose_structured_report(result, question, analysis, narrative)
    result.presentation = report
    return report.summary


def rows_to_markdown(result: SemanticQueryResult) -> str:
    if not result.columns:
        return ""
    lines = [
        "| " + " | ".join(column.title for column in result.columns) + " |",
        "| " + " | ".join("---" for _ in result.columns) + " |",
    ]
    for row in result.rows:
        values = []
        for column in result.columns:
            value = row.get(column.name)
            text = json.dumps(value, ensure_ascii=False, default=str) if isinstance(value, (dict, list)) else str(value if value is not None else "")
            values.append(text.replace("|", "\\|").replace("\n", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


__all__ = [
    "REPORT_FAILURE_WARNING",
    "analyze_result",
    "build_presentation_plan",
    "build_result_outputs",
    "build_result_profile",
    "compose_structured_report",
    "generate_narrative",
    "generate_report",
    "render_structured_report",
    "report_response_format",
    "rows_to_markdown",
    "validate_narrative",
]
