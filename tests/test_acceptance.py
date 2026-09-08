from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.scenarios.cqccri_smart_query.subgraph.common_models import SmartQueryInput

from ..contracts import QueryOutcome, SemanticCatalogMode, TaskExecutionStatus
from ..graph import build_hydrology_semantic_query_graph
from ..knowledge import BusinessPlaybook
from ..query_child.client import CubeClientError
from ..query_child.config import HydrologySemanticQuerySettings
from ..query_child.runtime import HydrologySemanticQueryServices
from ..report_child.node import VISUALIZATION_FAILURE_WARNING
from ..report_child.report import REPORT_FAILURE_WARNING
from ..state import HYDROLOGY_SEMANTIC_QUERY_SCENE_ID


def _meta() -> dict[str, Any]:
    return {
        "cubes": [
            {
                "name": "base_device_info",
                "type": "cube",
                "public": True,
                "title": "设备信息",
                "description": "水文设备基础信息，设备名称属于此实体",
                "connectedComponent": 1,
                "meta": {
                    "business_domain": "hydrology",
                    "default_projection": ["name", "code"],
                },
                "measures": [],
                "dimensions": [
                    {
                        "name": "base_device_info.name",
                        "title": "设备名称",
                        "type": "string",
                        "public": True,
                    },
                    {
                        "name": "base_device_info.code",
                        "title": "设备编码",
                        "type": "string",
                        "public": True,
                    },
                ],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
            {
                "name": "base_device_x_value",
                "type": "cube",
                "public": True,
                "title": "传感器实时值",
                "description": "传感器当前值、状态和更新时间",
                "connectedComponent": 1,
                "meta": {"business_domain": "hydrology"},
                "measures": [
                    {
                        "name": "base_device_x_value.sample_count",
                        "title": "样本数",
                        "type": "number",
                        "public": True,
                    }
                ],
                "dimensions": [
                    {
                        "name": "base_device_x_value.sensor_current_value",
                        "title": "传感器当前值",
                        "type": "number",
                        "public": True,
                    },
                    {
                        "name": "base_device_x_value.sensor_status",
                        "title": "传感器状态",
                        "type": "string",
                        "public": True,
                    },
                    {
                        "name": "base_device_x_value.updated_at",
                        "title": "更新时间",
                        "type": "time",
                        "public": True,
                    },
                ],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
            {
                "name": "hydrology_water_quality_view",
                "type": "view",
                "public": True,
                "title": "水质汇总",
                "meta": {"business_domain": "hydrology"},
                "measures": [
                    {
                        "name": "hydrology_water_quality_view.sample_count",
                        "title": "样本数",
                        "type": "number",
                        "public": True,
                    }
                ],
                "dimensions": [
                    {
                        "name": "hydrology_water_quality_view.device_name",
                        "title": "设备名称",
                        "type": "string",
                        "public": True,
                    }
                ],
                "segments": [],
                "folders": [],
                "hierarchies": [],
            },
        ]
    }


def _query(
    *,
    model: str = "base_device_x_value",
    dimensions: list[str] | None = None,
    measures: list[str] | None = None,
    models: list[str] | None = None,
    filters: list[dict[str, Any]] | None = None,
    ungrouped: bool | None = None,
) -> str:
    selected_models = models or [model]
    selected_dimensions = dimensions or []
    selected_measures = measures or []
    if ungrouped is None:
        ungrouped = bool(selected_dimensions and not selected_measures)
    return json.dumps({
        "query_mode": (
            "view" if selected_models[0].startswith("hydrology_") else "cube"
        ),
        "models": selected_models,
        "measures": selected_measures,
        "dimensions": selected_dimensions,
        "segments": [],
        "filters": filters or [],
        "time_dimensions": [],
        "order": [],
        "limit": 10,
        "offset": 0,
        "ungrouped": ungrouped,
    }, ensure_ascii=False)


def _central_water_query() -> str:
    return _query(
        models=["base_device_info", "base_device_x_value"],
        dimensions=[
            "base_device_info.name",
            "base_device_x_value.sensor_current_value",
            "base_device_x_value.sensor_status",
            "base_device_x_value.updated_at",
        ],
        filters=[{
            "member": "base_device_info.name",
            "operator": "equals",
            "values": ["中央水仓"],
        }],
        ungrouped=True,
    )


def _task(task_id: str, objective: str, depends_on: list[str] | None = None) -> dict:
    return {
        "task_id": task_id,
        "objective": objective,
        "depends_on": depends_on or [],
        "condition": None,
    }


def _sections(task_ids: list[str]) -> list[dict]:
    return [
        {
            "section_id": "overview",
            "title": "总体情况",
            "objective": "概括查询范围与结果。",
            "source_task_ids": task_ids,
            "analysis_methods": ["overview"],
        },
        {
            "section_id": "conclusion",
            "title": "综合结论",
            "objective": "综合已有事实并说明局限。",
            "source_task_ids": task_ids,
            "analysis_methods": ["conclusion"],
        },
    ]


def _decision(
    action: str,
    tasks: list[dict],
    task_ids: list[str],
    *,
    matched_playbook: str | None = None,
    answer: str | None = None,
    summary: str = "更新执行计划。",
) -> str:
    return json.dumps({
        "action": action,
        "matched_playbook": matched_playbook,
        "query_tasks": tasks,
        "report_sections": [] if action == "respond" else _sections(task_ids),
        "direct_answer": answer,
        "summary": summary,
    }, ensure_ascii=False)


def _valid_report_response() -> str:
    return json.dumps({
        "title": "水质综合分析报告",
        "executive_summary": "查询结果已完成确定性核验。",
        "insights": [],
        "section_narratives": [
            {
                "section_id": "overview",
                "fact_ids": ["overview-q1-scope-001"],
                "analysis": "已按总体情况章节整理可验证事实。",
                "impact": "影响范围仅限当前查询结果。",
                "possible_cause": "当前证据不足以判断原因。",
                "conclusion": "总体情况以章节数据依据为准。",
                "recommendation": "建议结合章节局限继续核验。",
                "certainty": "medium",
            },
            {
                "section_id": "conclusion",
                "fact_ids": ["conclusion-q1-scope-001"],
                "analysis": "已综合现有章节事实。",
                "impact": "结论不超出当前数据范围。",
                "possible_cause": "当前证据不足以判断原因。",
                "conclusion": "综合结论以可验证事实为限。",
                "recommendation": "建议结合现场信息复核。",
                "certainty": "medium",
            },
        ],
    }, ensure_ascii=False)


def _valid_visualization_response() -> str:
    return json.dumps({
        "sections": [
            {
                "section_id": "overview",
                "charts": [],
                "no_chart_reason": "本章节使用事实和数据表表达。",
            },
            {
                "section_id": "conclusion",
                "charts": [],
                "no_chart_reason": "综合结论不需要重复图表。",
            },
        ],
    }, ensure_ascii=False)


class FakeModel:
    def __init__(self, responses: list[str | Exception | AIMessage]) -> None:
        self.responses = list(responses)
        self.messages: list[list[Any]] = []
        self.bindings: list[dict[str, Any]] = []
        self.bound_tools: list[list[Any]] = []

    def bind(self, **kwargs: Any) -> FakeModel:
        self.bindings.append(kwargs)
        return self

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> FakeModel:
        self.bound_tools.append(tools)
        self.bindings.append(kwargs)
        return self

    async def ainvoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        del config
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("FakeModel 没有剩余响应")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, AIMessage):
            return response
        return AIMessage(content=response)


class RoutingModel:
    def __init__(
        self,
        runtime: FakeRuntime,
        *,
        bindings: dict[str, Any] | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.bindings = bindings or {}
        self.tools = tools

    def bind(self, **kwargs: Any) -> RoutingModel:
        return RoutingModel(
            self.runtime,
            bindings={**self.bindings, **kwargs},
            tools=self.tools,
        )

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> RoutingModel:
        return RoutingModel(
            self.runtime,
            bindings={**self.bindings, **kwargs},
            tools=tools,
        )

    def _target(self, messages: list[Any]) -> FakeModel:
        system = next(
            (
                str(message.content)
                for message in messages
                if isinstance(message, SystemMessage)
            ),
            "",
        )
        if "查询子 Agent" in system:
            return self.runtime.query_model
        if "主控 Agent" in system:
            return self.runtime.main_model
        if "多轮问题改写器" in system:
            return self.runtime.context_model
        raise AssertionError(f"无法识别模型角色：{system[:100]}")

    async def ainvoke(self, messages: list[Any], config: Any = None) -> AIMessage:
        target = self._target(messages)
        if self.bindings:
            target.bindings.append(self.bindings)
        if self.tools is not None:
            target.bound_tools.append(self.tools)
        return await target.ainvoke(messages, config=config)


class FakeRuntime:
    def __init__(
        self,
        *,
        main_responses: list[str | Exception | AIMessage],
        query_responses: list[str | Exception | AIMessage] | None = None,
        report_responses: list[str | Exception | AIMessage] | None = None,
        visualization_response: str | Exception | AIMessage | None = None,
        context_responses: list[str | Exception | AIMessage] | None = None,
    ) -> None:
        self.main_model = FakeModel(main_responses)
        self.query_model = FakeModel(query_responses or [])
        self.report_model = FakeModel([
            (
                _valid_visualization_response()
                if visualization_response is None
                else visualization_response
            ),
            *(report_responses or [_valid_report_response()]),
        ])
        self.context_model = FakeModel(context_responses or [])

    def get_chat_model(self, streaming: bool) -> FakeModel | RoutingModel:
        return self.report_model if streaming else RoutingModel(self)


class FakeCubeClient:
    def __init__(
        self,
        *,
        sql_failures: list[Exception | None] | None = None,
        load_results: list[dict[str, Any] | Exception] | None = None,
    ) -> None:
        self.sql_failures = list(sql_failures or [])
        self.load_results = list(load_results or [])
        self.sql_queries: list[dict[str, Any]] = []
        self.load_queries: list[dict[str, Any]] = []
        self.events: list[str] = []

    async def get_meta(self) -> dict[str, Any]:
        self.events.append("meta")
        return _meta()

    async def get_sql(self, query: dict[str, Any]) -> tuple[str, list[Any]]:
        self.events.append("sql")
        self.sql_queries.append(query)
        if self.sql_failures:
            failure = self.sql_failures.pop(0)
            if failure is not None:
                raise failure
        return f"SELECT {len(self.sql_queries)} FROM governed_semantic_model", []

    async def load(self, query: dict[str, Any]) -> dict[str, Any]:
        self.events.append("load")
        self.load_queries.append(query)
        if self.load_results:
            result = self.load_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        dimensions = query.get("dimensions", [])
        measures = query.get("measures", [])
        row = {name: "value" for name in [*dimensions, *measures]}
        annotation = {
            "dimensions": {
                name: {"title": name, "type": "string"} for name in dimensions
            },
            "measures": {
                name: {"title": name, "type": "number"} for name in measures
            },
        }
        return {"data": [row], "annotation": annotation}


class TrackingServices(HydrologySemanticQueryServices):
    @property
    def retrieval_limits(self) -> list[int | None]:
        return self.catalog_search.retrieval_limits if self.catalog_search else []

    @property
    def retrieval_questions(self) -> list[str]:
        return self.catalog_search.retrieval_questions if self.catalog_search else []


def _settings(
    *,
    max_agent_iterations: int = 6,
    max_query_rounds: int = 5,
) -> HydrologySemanticQuerySettings:
    return HydrologySemanticQuerySettings(
        cube_url="http://cube/cubejs-api/v1",
        cube_token=None,
        timeout_seconds=1,
        continue_wait_retries=0,
        meta_cache_ttl_seconds=60,
        timezone="Asia/Shanghai",
        catalog_mode=SemanticCatalogMode.VECTOR,
        embedding_model=None,
        context_top_k=20,
        vector_index_path=None,
        embedding_batch_size=4,
        embedding_concurrency=1,
        auto_full_context_max_chars=30000,
        max_agent_iterations=max_agent_iterations,
        max_query_rounds=max_query_rounds,
    )


async def _invoke(
    *,
    main_responses: list[str | Exception | AIMessage],
    query_responses: list[str | Exception | AIMessage] | None = None,
    client: FakeCubeClient | None = None,
    question: str = "查询中央水仓水质",
    report_responses: list[str | Exception | AIMessage] | None = None,
    visualization_response: str | Exception | AIMessage | None = None,
    context_responses: list[str | Exception | AIMessage] | None = None,
    conversation_context: Any = None,
    metadata: dict[str, Any] | None = None,
    playbooks: tuple[BusinessPlaybook, ...] = (),
    max_agent_iterations: int = 6,
    max_query_rounds: int = 5,
    captured_outputs: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], FakeRuntime, TrackingServices, FakeCubeClient]:
    runtime = FakeRuntime(
        main_responses=main_responses,
        query_responses=query_responses,
        report_responses=report_responses,
        visualization_response=visualization_response,
        context_responses=context_responses,
    )
    cube = client or FakeCubeClient()
    services = TrackingServices(
        _settings(
            max_agent_iterations=max_agent_iterations,
            max_query_rounds=max_query_rounds,
        ),
        client=cube,
        embedding_client=None,
        business_playbooks=playbooks,
    )
    graph = build_hydrology_semantic_query_graph(runtime, services).compile()
    del metadata
    messages = []
    if isinstance(conversation_context, dict) and conversation_context.get(
        "previous_question"
    ):
        messages.append(HumanMessage(content=str(conversation_context["previous_question"])))
    messages.append(HumanMessage(content=question))
    graph_input = {
        "sence_id": HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
        "smart_query_input": SmartQueryInput(
            query=question,
            scene_id=HYDROLOGY_SEMANTIC_QUERY_SCENE_ID,
        ),
        "messages": messages,
    }
    if captured_outputs is None:
        state = await graph.ainvoke(graph_input)
    else:
        state = None
        async for event in graph.astream_events(graph_input, version="v2"):
            if event.get("event") != "on_chain_end":
                continue
            output = event.get("data", {}).get("output")
            if event.get("name") == "LangGraph":
                state = output
            elif isinstance(output, dict):
                captured_outputs.extend(output.get("stream_outputs", []))
        assert state is not None
    return state, runtime, services, cube


def _search(query: str, limit: int | None = None) -> AIMessage:
    args: dict[str, Any] = {"query": query}
    if limit is not None:
        args["limit"] = limit
    return AIMessage(content="", tool_calls=[{
        "name": "search_semantic_catalog",
        "args": args,
        "id": f"search-{len(query)}-{limit}",
    }])


def _run(query: str | dict[str, Any]) -> AIMessage:
    payload = json.loads(query) if isinstance(query, str) else query
    return AIMessage(content="", tool_calls=[{
        "name": "run_semantic_query",
        "args": {"semantic_query": payload},
        "id": f"run-{len(json.dumps(payload))}",
    }])


async def test_single_query_plan_execute_preserves_public_result_and_prompt_boundaries() -> None:
    q1 = _task("q1", "查询中央水仓水质")
    state, runtime, _, cube = await _invoke(
        main_responses=[
            _decision("query", [q1], ["q1"], summary="执行水质查询。"),
            _decision("report", [], ["q1"], summary="查询完成，生成报告。"),
        ],
        query_responses=[_search(q1["objective"]), _run(_central_water_query())],
    )

    result = state["result"]
    assert result.outcome == QueryOutcome.SUCCESS
    assert result.query_count == 1
    assert result.query_history[0].task_id == "q1"
    assert result.task_results[0].status == TaskExecutionStatus.SUCCESS
    assert result.presentation.summary == state["answer"]
    assert cube.events == ["meta", "sql", "load"]
    main_prompt = str(runtime.main_model.messages[0][0].content)
    query_prompt = str(runtime.query_model.messages[0][0].content)
    report_prompt = str(runtime.report_model.messages[0][0].content)
    assert "query_mode" not in main_prompt
    assert "business_knowledge" not in main_prompt
    assert "此请求级知识" not in main_prompt
    assert "查询子 Agent" in query_prompt
    assert "业务编排知识" not in query_prompt
    assert "Cube" not in report_prompt


async def test_direct_response_does_not_load_cube_or_create_result() -> None:
    state, _, _, cube = await _invoke(main_responses=[
        _decision(
            "respond",
            [],
            [],
            answer="我可以规划水文语义查询和综合分析报告。",
        )
    ], question="你能做什么")

    assert state["result"] is None
    assert state["answer"] == "我可以规划水文语义查询和综合分析报告。"
    assert cube.events == []


async def test_multi_step_queries_replan_sequentially_and_aggregate_history() -> None:
    q1 = _task("q1", "查询目标设备")
    q2 = _task("q2", "查询目标设备的实时状态", ["q1"])
    client = FakeCubeClient(load_results=[
        {
            "data": [{"base_device_info.name": "一号站"}],
            "annotation": {
                "dimensions": {
                    "base_device_info.name": {"title": "设备名称", "type": "string"}
                }
            },
        },
        {
            "data": [{"base_device_x_value.sensor_status": "正常"}],
            "annotation": {
                "dimensions": {
                    "base_device_x_value.sensor_status": {
                        "title": "状态",
                        "type": "string",
                    }
                }
            },
        },
    ])
    state, _, _, cube = await _invoke(
        client=client,
        question="先找设备，再查询状态",
        main_responses=[
            _decision("query", [q1, q2], ["q1", "q2"], summary="执行两步查询。"),
            _decision("query", [q2], ["q1", "q2"], summary="继续查询状态。"),
            _decision("report", [], ["q1", "q2"], summary="生成综合报告。"),
        ],
        query_responses=[
            _search(q1["objective"]),
            _run(_query(model="base_device_info", dimensions=["base_device_info.name"])),
            _search(q2["objective"]),
            _run(_query(dimensions=["base_device_x_value.sensor_status"])),
        ],
    )

    result = state["result"]
    assert result.query_count == 2
    assert [record.task_id for record in result.query_history] == ["q1", "q2"]
    assert [revision.revision for revision in result.plan_revisions] == [1, 2, 3]
    assert [item.status for item in result.task_results] == [
        TaskExecutionStatus.SUCCESS,
        TaskExecutionStatus.SUCCESS,
    ]
    assert cube.events == ["meta", "sql", "load", "meta", "sql", "load"]
    assert [section.id for section in result.presentation.sections] == [
        "overview",
        "conclusion",
    ]
    assert all(
        section.source_task_ids == ["q1", "q2"]
        for section in result.presentation.sections
    )


async def test_query_agent_refreshes_unknown_member_and_retries_within_one_task() -> None:
    q1 = _task("q1", "查询设备状态")
    invalid = _query(dimensions=["base_device_x_value.unknown_status"])
    valid = _query(dimensions=["base_device_x_value.sensor_status"])
    state, runtime, services, cube = await _invoke(
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[
            _search(q1["objective"]),
            _run(invalid),
            _search("unknown_member", 40),
            _run(valid),
        ],
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].attempts == 2
    assert state["result"].query_count == 1
    assert services.retrieval_limits == [None, 40]
    assert len(runtime.query_model.messages) == 4
    assert cube.events == ["meta", "sql", "load"]


async def test_query_agent_requires_catalog_search_before_execution() -> None:
    q1 = _task("q1", "查询中央水仓水质")
    state, runtime, _, cube = await _invoke(
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[
            _run(_central_water_query()),
            _search(q1["objective"]),
            _run(_central_water_query()),
        ],
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].attempts == 2
    assert "catalog_search_required" in str(runtime.query_model.messages[1])
    assert cube.events == ["meta", "sql", "load"]


async def test_correctable_cube_400_is_recompiled_without_new_main_task() -> None:
    q1 = _task("q1", "查询设备状态")
    query = _query(dimensions=["base_device_x_value.sensor_status"])
    error = CubeClientError(
        "Unknown member",
        code="cube_http_error",
        status_code=400,
        retryable_by_model=True,
    )
    client = FakeCubeClient(sql_failures=[error, None])
    state, _, services, cube = await _invoke(
        client=client,
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[
            _search(q1["objective"]),
            _run(query),
            _search("cube_http_error", 40),
            _run(query),
        ],
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["dispatch_count"] == 1
    assert state["result"].attempts == 2
    assert services.retrieval_limits == [None, 40]
    assert cube.events == ["meta", "sql", "sql", "load"]


async def test_no_data_replans_then_returns_no_data_without_report_call() -> None:
    q1 = _task("q1", "查询设备状态")
    client = FakeCubeClient(load_results=[{"data": [], "annotation": {}}])
    state, runtime, _, _ = await _invoke(
        client=client,
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[
            _search(q1["objective"]),
            _run(_query(dimensions=["base_device_x_value.sensor_status"])),
        ],
    )

    assert state["result"].outcome == QueryOutcome.NO_DATA
    assert state["task_results"][0].status == TaskExecutionStatus.NO_DATA
    assert runtime.report_model.messages == []


async def test_terminal_failure_after_success_generates_partial_report() -> None:
    q1 = _task("q1", "查询设备名称")
    q2 = _task("q2", "查询设备状态")
    failure = CubeClientError("Cube 网络请求失败", code="cube_network_error")
    client = FakeCubeClient(sql_failures=[None, failure])
    state, _, _, cube = await _invoke(
        client=client,
        main_responses=[
            _decision("query", [q1, q2], ["q1", "q2"]),
            _decision("query", [q2], ["q1", "q2"]),
            _decision("report", [], ["q1", "q2"]),
        ],
        query_responses=[
            _search(q1["objective"]),
            _run(_query(model="base_device_info", dimensions=["base_device_info.name"])),
            _search(q2["objective"]),
            _run(_query(dimensions=["base_device_x_value.sensor_status"])),
        ],
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert [item.status for item in state["task_results"]] == [
        TaskExecutionStatus.SUCCESS,
        TaskExecutionStatus.FAILED,
    ]
    assert state["querying_blocked"] is True
    assert "未完成" in "\n".join(state["result"].warnings)
    assert "任务失败" in state["answer"]
    assert cube.events == ["meta", "sql", "load", "meta", "sql"]


async def test_invalid_initial_plan_retries_once_then_returns_planner_error() -> None:
    state, runtime, _, cube = await _invoke(
        main_responses=["not-json", "still-not-json"],
    )

    assert state["result"].outcome == QueryOutcome.PLANNER_ERROR
    assert len(runtime.main_model.messages) == 2
    assert cube.events == []


async def test_invalid_replan_falls_back_to_remaining_plan() -> None:
    q1 = _task("q1", "查询设备名称")
    q2 = _task("q2", "查询设备状态")
    state, _, _, cube = await _invoke(
        main_responses=[
            _decision("query", [q1, q2], ["q1", "q2"]),
            "not-json",
            "still-not-json",
            _decision("report", [], ["q1", "q2"]),
        ],
        query_responses=[
            _search(q1["objective"]),
            _run(_query(model="base_device_info", dimensions=["base_device_info.name"])),
            _search(q2["objective"]),
            _run(_query(dimensions=["base_device_x_value.sensor_status"])),
        ],
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].query_count == 2
    assert any("安全回退计划" in warning for warning in state["result"].warnings)
    assert cube.events.count("load") == 2


async def test_task_budget_rejects_oversized_plan() -> None:
    tasks = [_task(f"q{index}", f"任务 {index}") for index in range(1, 7)]
    invalid = _decision("query", tasks, [task["task_id"] for task in tasks])
    state, _, _, cube = await _invoke(
        main_responses=[invalid, invalid],
        max_query_rounds=5,
    )

    assert state["result"].outcome == QueryOutcome.PLANNER_ERROR
    assert cube.events == []


async def test_one_query_task_cannot_execute_a_second_successful_query() -> None:
    q1 = _task("q1", "查询设备状态")
    state, runtime, _, cube = await _invoke(
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[
            _search(q1["objective"]),
            _run(_query(dimensions=["base_device_x_value.sensor_status"])),
            _run(_central_water_query()),
        ],
    )

    assert state["result"].query_count == 1
    assert cube.events == ["meta", "sql", "load"]
    assert len(runtime.query_model.responses) == 1


async def test_conversation_context_is_rewritten_before_main_planning() -> None:
    q1 = _task("q1", "查询中央水仓去年的水质")
    state, runtime, services, _ = await _invoke(
        question="那去年呢",
        conversation_context={"previous_question": "查询中央水仓今年的水质"},
        context_responses=[
            json.dumps(
                {"standalone_question": q1["objective"]},
                ensure_ascii=False,
            )
        ],
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[_search(q1["objective"]), _run(_central_water_query())],
    )

    assert state["standalone_question"] == q1["objective"]
    main_input = json.loads(runtime.main_model.messages[0][1].content)
    assert main_input["standalone_question"] == q1["objective"]
    assert services.retrieval_questions == [q1["objective"]]


async def test_loaded_playbook_is_visible_only_to_main_agent() -> None:
    playbook = BusinessPlaybook(
        name="涌水异常综合分析.md",
        content="适用问题：分析涌水异常\n查询步骤：查询涌水量。",
    )
    q1 = _task("q1", "查询涌水量")
    state, runtime, _, _ = await _invoke(
        question="分析涌水异常",
        playbooks=(playbook,),
        main_responses=[
            _decision(
                "query",
                [q1],
                ["q1"],
                matched_playbook=playbook.name,
            ),
            _decision(
                "report",
                [],
                ["q1"],
                matched_playbook=playbook.name,
            ),
        ],
        query_responses=[_search(q1["objective"]), _run(_central_water_query())],
    )

    assert state["result"].matched_playbook == playbook.name
    assert playbook.content in str(runtime.main_model.messages[0][0].content)
    assert playbook.content not in str(runtime.query_model.messages[0][0].content)
    assert playbook.content not in str(runtime.report_model.messages[0][0].content)


@pytest.mark.parametrize("report_response", ["", TimeoutError("report timeout")])
async def test_report_failure_uses_deterministic_custom_sections(
    report_response: str | Exception,
) -> None:
    q1 = _task("q1", "查询中央水仓水质")
    state, _, _, _ = await _invoke(
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[_search(q1["objective"]), _run(_central_water_query())],
        report_responses=[report_response],
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert "## 1. 总体情况" in state["answer"]
    assert "## 2. 综合结论" in state["answer"]
    assert any(REPORT_FAILURE_WARNING in warning for warning in state["result"].warnings)


async def test_visualization_failure_uses_section_level_fallback() -> None:
    q1 = _task("q1", "查询中央水仓水质")
    state, _, _, _ = await _invoke(
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[_search(q1["objective"]), _run(_central_water_query())],
        visualization_response=TimeoutError("visualization timeout"),
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    assert state["result"].presentation.protocol_version == "1.1"
    assert any(
        VISUALIZATION_FAILURE_WARNING in warning
        for warning in state["result"].warnings
    )
    step = next(
        item
        for item in state["result"].steps
        if item.stage == "report_visualization_plan"
    )
    assert step.status.value == "failed"


async def test_streaming_emits_plan_query_and_report_outputs() -> None:
    q1 = _task("q1", "查询中央水仓水质")
    outputs: list[dict[str, Any]] = []
    state, _, _, _ = await _invoke(
        main_responses=[
            _decision("query", [q1], ["q1"]),
            _decision("report", [], ["q1"]),
        ],
        query_responses=[_search(q1["objective"]), _run(_central_water_query())],
        captured_outputs=outputs,
    )

    assert state["result"].outcome == QueryOutcome.SUCCESS
    output_types = {output["output_type"] for output in outputs}
    assert "CHAIN_OF_THOUGHT" in output_types
    assert "LLM_STREAM" in output_types
    assert "TABLE_OUTPUT" in output_types


def test_main_graph_contains_plan_execute_control_nodes() -> None:
    runtime = FakeRuntime(main_responses=[])
    services = TrackingServices(
        _settings(),
        client=FakeCubeClient(),
        embedding_client=None,
        business_playbooks=(),
    )
    graph = build_hydrology_semantic_query_graph(runtime, services)

    assert set(graph.nodes) == {
        "cqccri_entry",
        "initialize",
        "main_plan",
        "query_agent",
        "main_replan",
        "finalize",
        "report_agent",
        "cqccri_result",
        "clean",
    }
