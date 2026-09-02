from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

REPORT_NARRATIVE_PROMPT = (
    "你是谨慎的水文综合分析助手。仅基于输入事实包返回符合 JSON Schema 的叙事草稿。"
    "每项洞察必须引用有效 fact_id；事实、可能原因和条件性建议必须分开。"
    "不得生成事实包中不存在的数字，不得将相关性表述为因果，不得把统计异常等同于水害风险。"
    "提供 report_sections 时，每项洞察必须填写有效 section_id 并服从对应章节目标。"
    "insights 最多 8 项，不输出 Markdown 或工具调用协议。"
)


def report_narrative_messages(fact_packet: dict[str, Any]) -> list[BaseMessage]:
    return [
        SystemMessage(content=REPORT_NARRATIVE_PROMPT),
        HumanMessage(content=json.dumps(
            fact_packet,
            ensure_ascii=False,
            separators=(",", ":"),
        )),
    ]


__all__ = ["REPORT_NARRATIVE_PROMPT", "report_narrative_messages"]
