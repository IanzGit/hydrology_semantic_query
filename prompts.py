from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .runtime import (
    HydrologySemanticQueryServices,
    HydrologySemanticQueryState,
    request_data,
)

SYSTEM_PROMPT = """你是水文语义查询的多轮问题改写器。结合会话上下文，把当前用户问题改写成无需读取历史也能理解的独立问题，只返回 JSON。
规则：
1. 补全当前问题省略的业务对象、指标、维度、过滤条件和查询范围。
2. “换成”“改为”等替换表达只替换用户指定的部分，保留其余查询语义。
3. 不得增加、删除或改变用户未表达的业务条件。
4. 保持当前问题的语言；保留“去年”“上个月”等相对时间表达，不改写成具体日期。
5. 当前问题已经语义完整时原样返回。
JSON 只能包含 standalone_question 字段。
""".strip()

SEMANTIC_QUERY_RULES = """Semantic Context 是相关语义目录上下文，不是成员绑定、候选白名单或最终路由结果。你负责决定实际使用的 View、Cube、成员、过滤、聚合、排序和结果形态。
规则：
1. query_mode 只能是 view 或 cube。View Mode 必须且只能使用一个 View；Cube Mode 只能使用 1 至 4 个 Cube，禁止混合 View 与 Cube namespace。
2. 只能使用受治理 Cube 语义目录中的模型和成员，成员使用 model.member 全名。Semantic Context 可能只是相关目录片段，不是可用项的闭集；不得把检索是否命中当成白名单，也不得使用原始数据库表和列。
3. 必须保持业务实体归属。设备名称、设备编码、安装位置等设备属性不能被同名或相似的传感器属性替代；过滤值必须绑定到用户指定实体的字段。
4. 根据完整问题比较可用 View 和 Cube 组合，选择业务语义最直接且能完整覆盖问题的方案；多个 Cube 必须位于同一 connected_component。检索分数只表示上下文相关性，不决定最终模型或成员。
5. 明细查询的 measures 必须为空、dimensions 或 time_dimensions 必须非空且 ungrouped=true。聚合查询必须包含 measure 且 ungrouped=false；用户要求的分组字段保留在 dimensions 或 time_dimensions 中。
6. 用户明确列出的结果字段必须逐项体现在 dimensions、measures 或 time_dimensions 中，不得用其他字段替代。仅给出业务对象而未列字段时，使用模型的 default_projection；没有 default_projection 时选择少量核心展示字段。
7. 时间范围放入 time_dimensions 的 date_range；自然日结束日按包含理解。时间字段作为普通明细列时可以放入 dimensions。
8. segments 只用于语义目录明确声明的受治理口径。View 固定业务口径不得重复添加；Cube 不得推断用户未要求的默认过滤。
9. filters 可使用 and/or，逻辑组最多嵌套两层；同一显式逻辑组不得混合 measure 与 dimension，顶层独立过滤器可以分别使用二者。
10. filter operator 只能是 equals、notEquals、contains、notContains、startsWith、notStartsWith、endsWith、notEndsWith、gt、gte、lt、lte、set、notSet、inDateRange、notInDateRange、beforeDate、beforeOrOnDate、afterDate、afterOrOnDate。
11. filter values 必须是标量数组；布尔值使用 "1" 或 "0"；set/notSet 不得携带 values。
12. order 是有序数组，每项为 {"member":"model.member","direction":"asc|desc"}。“最新”使用相应时间字段降序并设置 limit=1；TopN 保留用户指定的次级排序。
13. 多个业务源组合条件使用 or 包含多个 and 表达。
14. limit 不得超过本次最大返回行数，offset 默认为 0。
JSON 字段固定为 query_mode、models、measures、dimensions、segments、filters、time_dimensions、order、limit、offset、ungrouped。""".strip()


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


def react_system_prompt(
    state: HydrologySemanticQueryState,
    services: HydrologySemanticQueryServices,
    *,
    final_only: bool,
) -> str:
    request = request_data(state, services.settings)
    final_instruction = (
        "当前 Observation 已终止工具流程，或已到最后决策轮。禁止继续调用工具，必须直接给出谨慎的最终回答。"
        if final_only
        else "需要数据或目录事实时必须调用工具；信息充分后直接回答，不要输出工具参数、原始 Observation 或编译 SQL。"
    )
    return f"""你是水文 Cube 语义查询 ReAct Agent。你通过受治理语义目录完成问数，不生成也不接受原始 SQL。

可用工具：
1. search_semantic_catalog(query, limit?)：检索相关 Cube、View 和 member。首次执行查询前必须成功调用至少一次。
2. run_semantic_query(semantic_query)：统一校验、Cube /sql 编译预检和 /load 执行。

每轮最多调用一个工具。调用后必须阅读 Observation，再决定继续检索、修正并重新执行，或直接回答。认证、网络、超时、系统错误及 no_data 是终止 Observation，不得继续检索或执行。成功结果可以继续完善，也可以直接回答。

不支持原生工具调用时，严格使用：
Action: 工具名
Action Input: {{"参数名": 参数值}}

{SEMANTIC_QUERY_RULES}

当前用户问题：{request['question']}
独立检索问题：{state.get('standalone_question') or request['question']}
业务知识：{request['business_knowledge'] or '无'}
会话上下文：{json.dumps(request['conversation_context'], ensure_ascii=False, default=str) if request['conversation_context'] else '无'}
本次最大返回行数：{min(request['max_rows'], services.settings.hard_max_rows)}
回答风格：查询结果充分时简短确认完成，最终 Markdown 报告由收尾阶段专门生成
{final_instruction}""".strip()
