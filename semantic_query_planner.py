from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from .models import SemanticContext, SemanticIntent, SemanticQuery
from .semantic_context import context_for_prompt

INTENT_SYSTEM_PROMPT = """你是水文语义查询意图解析器。只返回 JSON 对象，不得返回 SQL、Markdown、答案或额外字段。
将用户问题拆成轻量语义槽位：
- metrics：需要聚合、统计或计算的业务概念。
- dimensions：需要返回、分组或定位的业务属性。
- time：时间范围或时间语义，必须是单个字符串；没有则为 null，不得返回数组。
- filters：需要作为筛选条件的字段概念，不包含具体取值。
- sort：需要排序的业务概念。
- limit：用户明确要求的条数；没有则为 null。
使用用户原始业务词，不得发明 Cube 成员名。
JSON 字段固定为 metrics、dimensions、time、filters、sort、limit。
metrics、dimensions、filters、sort 必须始终是 JSON 数组，无内容时返回 []，不得返回 null。
""".strip()

SYSTEM_PROMPT = """你是水文 Cube 语义查询规划器。只能根据给定 Semantic Context 返回一个 JSON 对象，不得返回 SQL、Markdown、直接答案或额外字段。
规则：
1. query_mode 和 models 必须与 Context 完全一致。
2. View Mode 必须且只能使用一个 View；Cube Mode 可以使用 Context 中的 1 至 4 个 Cube；禁止混合 View 与 Cube namespace。
3. 只能使用 Allowed Members 中的完整 member 名称。
4. 聚合问题使用 measures；明细问题使用 dimensions 并设置 ungrouped=true，不得带 measure。
5. 时间范围只放入 time_dimensions，其内使用 date_range，自然日结束日按包含理解。
6. segments 只使用 Context 中的 segment。
7. filters 可使用递归 and/or，同一显式逻辑组不得混合 measure 与 dimension；顶层独立过滤器可分别使用二者。
8. order 是有序数组，每项为 {"member":"model.member","direction":"asc|desc"}。
9. “最新”使用时间降序且 limit=1；TopN 保留用户指定的次级排序。
10. 多个业务源组合条件要用 or 包含多个 and 表达。
11. filter operator 只能是 equals、notEquals、contains、notContains、startsWith、notStartsWith、endsWith、notEndsWith、gt、gte、lt、lte、set、notSet、inDateRange、notInDateRange、beforeDate、beforeOrOnDate、afterDate、afterOrOnDate。
12. filter values 必须是标量数组；布尔值必须使用 "1" 或 "0"；set/notSet 不得携带 values。
13. View 固定业务口径不得在 filters 中重复添加；Cube 是原始实体口径，不得推断额外默认过滤。
JSON 字段固定为 query_mode、models、measures、dimensions、segments、filters、time_dimensions、order、limit、offset、ungrouped。
""".strip()


def _conversation(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if value else "无"


def build_intent_messages(
    *,
    question: str,
    business_knowledge: str | None,
    conversation_context: Any,
) -> list[BaseMessage]:
    content = (
        f"用户问题：{question}\n"
        f"业务知识：{business_knowledge or '无'}\n"
        f"会话上下文：{_conversation(conversation_context)}"
    )
    return [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]


def build_messages(
    *,
    question: str,
    context: SemanticContext,
    business_knowledge: str | None,
    conversation_context: Any,
    max_rows: int,
    previous_query: SemanticQuery | None = None,
    previous_error: str | None = None,
) -> list[BaseMessage]:
    correction = ""
    if previous_error:
        previous = (
            previous_query.model_dump_json(by_alias=True, exclude_none=True)
            if previous_query
            else "未成功解析"
        )
        correction = f"\n上一次查询：{previous}\n上一次错误：{previous_error}\n请仅修正该错误。"
    content = (
        f"用户问题：{question}\n"
        f"业务知识：{business_knowledge or '无'}\n"
        f"会话上下文：{_conversation(conversation_context)}\n"
        f"本次最大返回行数：{max_rows}\n"
        f"Semantic Context：{context_for_prompt(context)}"
        f"{correction}"
    )
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=content)]


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型响应中不存在 JSON 对象")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型响应不是 JSON 对象")
    return payload


def parse_semantic_intent(text: str) -> SemanticIntent:
    try:
        return SemanticIntent.model_validate(_json_object(text))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"SemanticIntent 解析失败：{exc}") from exc


def parse_semantic_query(text: str) -> SemanticQuery:
    try:
        return SemanticQuery.model_validate(_json_object(text))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"SemanticQuery 解析失败：{exc}") from exc
