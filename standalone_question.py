from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, field_validator

SYSTEM_PROMPT = """你是水文语义查询的多轮问题改写器。结合会话上下文，把当前用户问题改写成无需读取历史也能理解的独立问题，只返回 JSON。
规则：
1. 补全当前问题省略的业务对象、指标、维度、过滤条件和查询范围。
2. “换成”“改为”等替换表达只替换用户指定的部分，保留其余查询语义。
3. 不得增加、删除或改变用户未表达的业务条件。
4. 保持当前问题的语言；保留“去年”“上个月”等相对时间表达，不改写成具体日期。
5. 当前问题已经语义完整时原样返回。
JSON 只能包含 standalone_question 字段。
""".strip()


class StandaloneQuestion(BaseModel):
    standalone_question: str

    model_config = ConfigDict(extra="forbid")

    @field_validator("standalone_question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("standalone_question 不能为空")
        return question


def build_messages(
    *,
    question: str,
    conversation_context: Any,
) -> list[BaseMessage]:
    context = json.dumps(
        conversation_context,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"会话上下文：{context}\n当前用户问题：{question}"),
    ]


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "hydrology_standalone_question",
            "strict": True,
            "schema": StandaloneQuestion.model_json_schema(),
        },
    }


def parse_standalone_question(text: str) -> str:
    return StandaloneQuestion.model_validate_json(text).standalone_question
