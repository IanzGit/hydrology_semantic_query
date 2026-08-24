from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.messages import stringify_message_content

from .models import SemanticQueryResult

REPORT_FAILURE_WARNING = "Markdown 分析报告生成失败，已保留查询摘要。"


def rows_to_markdown(result: SemanticQueryResult) -> str:
    if not result.columns:
        return ""
    lines = [
        "| " + " | ".join(column.title for column in result.columns) + " |",
        "| " + " | ".join("---" for _ in result.columns) + " |",
    ]
    for row in result.rows[:50]:
        values = []
        for column in result.columns:
            value = row.get(column.name)
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value if value is not None else "")
            values.append(text.replace("|", "\\|").replace("\n", "<br>"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


async def generate_report(runtime, question: str, result: SemanticQueryResult) -> str:
    messages = [
        SystemMessage(content=(
            "你是谨慎的水文数据分析助手。用中文生成不超过 800 字的简洁 Markdown 报告，"
            "只总结数据可直接验证的结论，不得编造。"
        )),
        HumanMessage(content=f"用户问题：{question}\n\nCube 查询数据：\n{rows_to_markdown(result)}"),
    ]
    model = runtime.get_chat_model(streaming=True).bind(
        extra_body={"enable_thinking": False},
    )
    response = await model.ainvoke(messages, config={"callbacks": []})
    report = stringify_message_content(response.content).strip()
    if not report:
        raise ValueError("报告模型返回了空响应")
    return report
