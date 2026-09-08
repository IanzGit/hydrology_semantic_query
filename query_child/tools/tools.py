from __future__ import annotations

from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool
from langgraph.types import Command
from pydantic import Field

from ...contracts import SemanticQuery
from ..runtime import (
    HydrologySemanticQueryServices,
)
from .run_semantic_query import run_semantic_query_service
from .search_semantic_catalog import (
    CatalogSearchRuntime,
    search_semantic_catalog_service,
)

ALL_TOOL_NAMES = {"search_semantic_catalog", "run_semantic_query"}


def build_hydrology_semantic_query_tools(
    services: HydrologySemanticQueryServices,
) -> list[BaseTool]:
    search_runtime = CatalogSearchRuntime(services)
    services.catalog_search = search_runtime

    @tool(
        "search_semantic_catalog",
        description="按当前用户问题检索相关的受治理 Cube、View 和 member。首次执行语义查询前必须调用。",
    )
    async def search_semantic_catalog(
        query: Annotated[str, Field(min_length=1)],
        runtime: ToolRuntime,
        limit: Annotated[int | None, Field(ge=1, le=40)] = None,
    ) -> Command:
        return await search_semantic_catalog_service(
            query=query,
            runtime=runtime,
            limit=limit,
            search_runtime=search_runtime,
            services=services,
        )

    @tool(
        "run_semantic_query",
        description="按明确的本轮查询目标校验并执行受治理的 SemanticQuery，内部固定先调用 Cube /sql 预检，再调用 /load。禁止传入原始 SQL。",
    )
    async def run_semantic_query(
        semantic_query: SemanticQuery,
        runtime: ToolRuntime,
        query_goal: Annotated[str | None, Field(min_length=1)] = None,
    ) -> Command:
        return await run_semantic_query_service(
            semantic_query=semantic_query,
            query_goal=query_goal,
            runtime=runtime,
            services=services,
        )

    return [search_semantic_catalog, run_semantic_query]


__all__ = ["ALL_TOOL_NAMES", "build_hydrology_semantic_query_tools"]
