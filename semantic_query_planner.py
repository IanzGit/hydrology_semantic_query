from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from .models import (
    QueryMode,
    QueryUnderstanding,
    RetrievalIntent,
    SemanticContext,
    SemanticQuery,
)
from .semantic_context import context_for_prompt

RETRIEVAL_INTENT_SYSTEM_PROMPT = """你负责理解用户问题的语义检索需求和结果形态。只返回 JSON 对象，不得返回 SQL、Markdown、答案或额外字段。
你的职责是指出：
1. 用户需要哪些业务语义 phrase；
2. 每个 phrase 的用途：select、filter、group；
3. 用户明确要求 count、sum、avg、min、max 时，将 aggregate 放在被聚合的业务对象上。
4. projection_mode 只能是 detail、aggregate、default；projection_policy 只能是 explicit、model_default、summary。
重要规则：
- 不生成 Cube member 名称、measures、dimensions、filters、segments、order、limit、timeDimensions 或 SQL。
- 不把“数量”“多少”“总数”等单独作为 phrase；“设备数量”应输出 phrase="设备", aggregate="count"。
- “当前”“最新”等查询操作词不是业务 member，不要单独作为 need。
- “具体情况、明细、列表、有哪些、分别是什么”使用 projection_mode=detail；“多少、数量、总数、统计”使用 projection_mode=aggregate；未明确结果形态使用 default。
- 仅给出业务对象时使用 projection_policy=model_default；明确列出字段时使用 explicit；聚合问题使用 summary。
- “具体情况”本身不是业务 member，不得作为 need。
- filter phrase 必须保留完整业务含义，例如“启用设备”，不要只输出“启用”。
- 使用用户原始业务词，不得发明 Cube 成员名。
JSON 字段固定为 needs、projection_mode、projection_policy。needs 必须始终是数组；每项只有 phrase、usage、aggregate 三个字段，aggregate 无值时为 null。
""".strip()

SYSTEM_PROMPT = """你是水文 Cube 语义查询规划器。只能根据给定 Semantic Context 返回一个 JSON 对象，不得返回 SQL、Markdown、直接答案或额外字段。
规则：
1. query_mode 只能是 view 或 cube；models 只能从 Context 的 candidate_models 中选择，且 query_mode 必须与所选模型类型一致。
2. View Mode 必须且只能使用一个候选 View；Cube Mode 只能使用候选 Cube 中的 1 至 4 个，禁止混合 View 与 Cube namespace。
3. 只能使用 Allowed Members 中的完整 member 名称。
4. projection_mode=detail 时，measures 必须为空、dimensions 必须非空且 ungrouped=true；projection_mode=aggregate 时，measures 必须非空且 ungrouped=false；projection_mode=default 时遵循 Context 和一般结构规则。
5. 时间范围只放入 time_dimensions，其内使用 date_range，自然日结束日按包含理解。
6. segments 只使用 Context 中的 segment。
7. filters 可使用递归 and/or，同一显式逻辑组不得混合 measure 与 dimension；顶层独立过滤器可分别使用二者。
8. order 是有序数组，每项为 {"member":"model.member","direction":"asc|desc"}。
9. “最新”使用时间降序且 limit=1；TopN 保留用户指定的次级排序。
10. 多个业务源组合条件要用 or 包含多个 and 表达。
11. filter operator 只能是 equals、notEquals、contains、notContains、startsWith、notStartsWith、endsWith、notEndsWith、gt、gte、lt、lte、set、notSet、inDateRange、notInDateRange、beforeDate、beforeOrOnDate、afterDate、afterOrOnDate。
12. filter values 必须是标量数组；布尔值必须使用 "1" 或 "0"；set/notSet 不得携带 values。
13. View 固定业务口径不得在 filters 中重复添加；Cube 是原始实体口径，不得推断额外默认过滤。
14. projection_policy=model_default 时，只能从 suggested_members 中形成默认明细投影，不能因为存在 count measure 而优先统计；projection_policy=explicit 时按字段级 needs 和 binding_candidates 选择字段；projection_policy=summary 时按 aggregate need 选择 measure，并保留用户明确要求的分组 dimensions。
15. “当前”“最新”等操作语义必须根据原始问题和 Context 中的受治理成员自行生成；检索上下文不提供预解析的查询结构。
16. Context 中的 binding_candidates 仅是业务语义与成员之间的检索候选，score 仅表示检索相关性。你必须结合用户问题和成员元数据自行决定实际使用哪些成员，不要求每个业务语义与成员一一对应。
JSON 字段固定为 query_mode、models、measures、dimensions、segments、filters、time_dimensions、order、limit、offset、ungrouped。
""".strip()


def _conversation(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if value else "无"


def build_retrieval_intent_messages(
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
        SystemMessage(content=RETRIEVAL_INTENT_SYSTEM_PROMPT),
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


def _parse_query_understanding(text: str) -> QueryUnderstanding:
    try:
        payload = _json_object(text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise StructuredOutputParseError("json_syntax_error", str(exc)) from exc
    try:
        return QueryUnderstanding.model_validate(payload)
    except ValidationError as exc:
        raise StructuredOutputParseError(
            "schema_validation_error",
            "QueryUnderstanding 未通过结构校验",
            validation_errors=_validation_details(exc),
        ) from exc


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


def query_understanding_response_format() -> dict[str, Any]:
    return _response_format(
        "hydrology_query_understanding",
        _strict_schema(QueryUnderstanding.model_json_schema()),
    )


def semantic_query_response_format(context: SemanticContext) -> dict[str, Any]:
    schema = _strict_schema(SemanticQuery.model_json_schema(by_alias=True))
    properties = schema["properties"]
    details = context.member_details
    members = {
        kind: [
            name
            for name in context.allowed_members
            if details.get(name, {}).get("kind") == kind
        ]
        for kind in ("measure", "dimension", "segment")
    }
    time_dimensions = [
        name
        for name in members["dimension"]
        if details.get(name, {}).get("type") == "time"
    ]
    filter_members = [*members["measure"], *members["dimension"]]
    order_members = [*members["measure"], *members["dimension"]]
    properties["query_mode"] = {
        "enum": [mode.value for mode in QueryMode],
        "type": "string",
    }
    properties["models"]["items"] = {
        "enum": context.candidate_models,
        "type": "string",
    }

    def restrict_array(field: str, candidates: list[str]) -> None:
        if candidates:
            properties[field]["items"] = {
                "enum": candidates,
                "type": "string",
            }
        else:
            properties[field]["maxItems"] = 0

    for field, kind in (
        ("measures", "measure"),
        ("dimensions", "dimension"),
        ("segments", "segment"),
    ):
        restrict_array(field, members[kind])
    definitions = schema["$defs"]
    if filter_members:
        filter_properties = definitions["SemanticFilter"]["properties"]
        filter_properties["member"] = {
            "anyOf": [
                {"enum": filter_members, "type": "string"},
                {"type": "null"},
            ]
        }
    else:
        properties["filters"]["maxItems"] = 0
        filter_properties = definitions["SemanticFilter"]["properties"]
    filter_properties["values"]["items"] = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
        ]
    }
    if time_dimensions:
        definitions["TimeDimension"]["properties"]["dimension"] = {
            "enum": time_dimensions,
            "type": "string",
        }
    else:
        properties["time_dimensions"]["maxItems"] = 0
    if order_members:
        definitions["OrderItem"]["properties"]["member"] = {
            "enum": order_members,
            "type": "string",
        }
    else:
        properties["order"]["maxItems"] = 0
    return _response_format("hydrology_semantic_query", schema)


def parse_retrieval_intent(text: str) -> RetrievalIntent:
    try:
        return RetrievalIntent.model_validate(_json_object(text))
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise ValueError(f"RetrievalIntent 解析失败：{exc}") from exc


def parse_query_understanding(text: str) -> QueryUnderstanding:
    return _parse_query_understanding(text)


def parse_semantic_query(text: str) -> SemanticQuery:
    return _parse_semantic_query(text)
