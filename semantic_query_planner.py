from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from .models import SemanticContext, SemanticQuery
from .semantic_context import context_for_prompt

SYSTEM_PROMPT = """你是水文 Cube 语义查询规划器。你必须根据用户的完整问题和 Semantic Context 全局规划查询，只返回一个 JSON 对象，不得返回 SQL、Markdown、直接答案或额外字段。
Semantic Context 是相关语义目录上下文，不是成员绑定、候选白名单或最终路由结果。你负责决定实际使用的 View、Cube、成员、过滤、聚合、排序和结果形态。
规则：
1. query_mode 只能是 view 或 cube。View Mode 必须且只能使用一个 View；Cube Mode 只能使用 1 至 4 个 Cube，禁止混合 View 与 Cube namespace。
2. 只能使用受治理 Cube 语义目录中的模型和成员，成员使用 model.member 全名。Semantic Context 可能只是相关目录片段，不是可用项的闭集；不得把检索是否命中当成白名单，也不得使用原始数据库表和列。
3. 必须保持业务实体归属。设备名称、设备编码、安装位置等设备属性不能被同名或相似的传感器属性替代；过滤值必须绑定到用户指定实体的字段。
4. 根据完整问题比较可用 View 和 Cube 组合，选择业务语义最直接且能完整覆盖问题的方案；多个 Cube 必须位于同一 connected_component。检索分数只表示上下文相关性，不决定最终模型或成员。
5. 明细查询的 measures 必须为空、dimensions 或 time_dimensions 必须非空且 ungrouped=true。聚合查询必须包含 measure 且 ungrouped=false；用户要求的分组字段保留在 dimensions 或 time_dimensions。
6. projection_role=filter_only 的成员只能用于 filters 或 segments，不得用于 dimensions、time_dimensions 或结果分组。
7. 用户明确列出的结果字段必须逐项体现在 dimensions、measures 或 time_dimensions 中，不得用其他字段替代。仅给出业务对象而未列字段时，使用模型的 default_projection；没有 default_projection 时选择少量核心展示字段。
8. 时间范围放入 time_dimensions 的 date_range；自然日结束日按包含理解。时间字段作为普通明细列时可以放入 dimensions。
9. segments 只用于语义目录明确声明的受治理口径。View 固定业务口径不得重复添加；Cube 不得推断用户未要求的默认过滤。
10. filters 可使用 and/or，逻辑组最多嵌套两层；同一显式逻辑组不得混合 measure 与 dimension，顶层独立过滤器可以分别使用二者。
11. filter operator 只能是 equals、notEquals、contains、notContains、startsWith、notStartsWith、endsWith、notEndsWith、gt、gte、lt、lte、set、notSet、inDateRange、notInDateRange、beforeDate、beforeOrOnDate、afterDate、afterOrOnDate。
12. filter values 必须是标量数组；布尔值使用 "1" 或 "0"；set/notSet 不得携带 values。
13. order 是有序数组，每项为 {"member":"model.member","direction":"asc|desc"}。“最新”使用相应时间字段降序并设置 limit=1；TopN 保留用户指定的次级排序。
14. 多个业务源组合条件使用 or 包含多个 and 表达。
15. limit 不得超过本次最大返回行数，offset 默认为 0。
JSON 字段固定为 query_mode、models、measures、dimensions、segments、filters、time_dimensions、order、limit、offset、ungrouped。
""".strip()


def _conversation(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if value else "无"


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
        correction = (
            f"\n上一次查询：{previous}"
            f"\n结构化错误反馈：{previous_error}"
            "\n根据错误阶段修正查询；不得改变用户未涉及的查询语义。"
        )
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


class StructuredOutputParseError(ValueError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        validation_errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.validation_errors = validation_errors or []


def _validation_details(exc: ValidationError) -> list[dict[str, Any]]:
    return [
        {
            "loc": list(item["loc"]),
            "type": item["type"],
            "msg": item["msg"],
        }
        for item in exc.errors(include_input=False)
    ]


def _parse_semantic_query(text: str) -> SemanticQuery:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredOutputParseError("json_syntax_error", str(exc)) from exc
    try:
        return SemanticQuery.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputParseError(
            "schema_validation_error",
            "SemanticQuery 未通过结构校验",
            validation_errors=_validation_details(exc),
        ) from exc


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    strict = deepcopy(schema)

    def transform(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                transform(item)
            return
        if not isinstance(value, dict):
            return
        value.pop("default", None)
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["required"] = list(properties)
            value["additionalProperties"] = False
        for child in value.values():
            transform(child)

    transform(strict)
    return strict


def _response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _filter_schema() -> dict[str, dict[str, Any]]:
    value_items = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
        ]
    }
    leaf = {
        "type": "object",
        "properties": {
            "member": {"type": "string"},
            "operator": {"$ref": "#/$defs/FilterOperator"},
            "values": {"type": "array", "items": value_items},
        },
        "required": ["member", "operator", "values"],
        "additionalProperties": False,
    }

    def group(operator: str, child: str) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                operator: {
                    "type": "array",
                    "items": {"$ref": f"#/$defs/{child}"},
                    "minItems": 1,
                },
            },
            "required": [operator],
            "additionalProperties": False,
        }

    return {
        "SemanticFilterLeaf": leaf,
        "SemanticFilterNested": {
            "anyOf": [
                {"$ref": "#/$defs/SemanticFilterLeaf"},
                group("and", "SemanticFilterLeaf"),
                group("or", "SemanticFilterLeaf"),
            ]
        },
        "SemanticFilter": {
            "anyOf": [
                {"$ref": "#/$defs/SemanticFilterLeaf"},
                group("and", "SemanticFilterNested"),
                group("or", "SemanticFilterNested"),
            ]
        },
    }


def semantic_query_response_format() -> dict[str, Any]:
    schema = _strict_schema(SemanticQuery.model_json_schema(by_alias=True))
    schema["$defs"].update(_filter_schema())
    return _response_format("hydrology_semantic_query", schema)


def parse_semantic_query(text: str) -> SemanticQuery:
    return _parse_semantic_query(text)
