import ast
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = SCENARIO_DIR / "query_child" / "tools"
SHARED_TOOLS_DIR = SCENARIO_DIR.parents[1] / "tools" / "hydrology_semantic_query"
PROJECT_DIR = SCENARIO_DIR.parents[3]
LEGACY_SCENARIO_MODULES = {
    "catalog.py",
    "client.py",
    "config.py",
    "models.py",
    "main_agent.py",
    "nodes.py",
    "output.py",
    "presentation_models.py",
    "presentation_planner.py",
    "presentation_renderers.py",
    "reporting.py",
    "report.py",
    "report_agent.py",
    "report_analysis.py",
    "report_rendering.py",
    "result_profile.py",
    "semantic_catalog.py",
    "semantic_catalog_retriever.py",
    "semantic_cube_client.py",
    "semantic_query_validator.py",
    "standalone_question.py",
    "runtime.py",
    "tool_call_parser.py",
    "query_agent.py",
    "workflow.py",
}
LEGACY_IMPORTS = {
    f"app.agents.scenarios.hydrology_semantic_query.{path.removesuffix('.py')}"
    for path in LEGACY_SCENARIO_MODULES
}
LEGACY_IMPORTS.update({
    "app.agents.scenarios.hydrology_semantic_query.client",
    "app.agents.scenarios.hydrology_semantic_query.config",
    "app.agents.scenarios.hydrology_semantic_query.models",
    "app.agents.scenarios.hydrology_semantic_query.runtime",
    "app.agents.scenarios.hydrology_semantic_query.tools",
    "app.agents.scenarios.hydrology_semantic_query.tools.common",
    "app.agents.scenarios.hydrology_semantic_query.tools.run_semantic_query",
    "app.agents.scenarios.hydrology_semantic_query.tools.search_semantic_catalog",
    "app.agents.scenarios.hydrology_semantic_query.tools.tools",
    "app.agents.tools.hydrology_semantic_query",
    "app.agents.tools.hydrology_semantic_query.common",
    "app.agents.tools.hydrology_semantic_query.services",
    "app.agents.tools.hydrology_semantic_query.services.run_semantic_query",
    "app.agents.tools.hydrology_semantic_query.services.search_semantic_catalog",
    "app.agents.scenarios.hydrology_semantic_query.catalog.parser",
    "app.agents.scenarios.hydrology_semantic_query.constants",
    "app.agents.scenarios.hydrology_semantic_query.domain.catalog",
    "app.agents.scenarios.hydrology_semantic_query.domain.query",
    "app.agents.scenarios.hydrology_semantic_query.domain.result",
    "app.agents.scenarios.hydrology_semantic_query.runtime.errors",
    "app.agents.scenarios.hydrology_semantic_query.runtime.request",
    "app.agents.scenarios.hydrology_semantic_query.runtime.services",
    "app.agents.scenarios.hydrology_semantic_query.runtime.state",
    "app.agents.scenarios.hydrology_semantic_query.runtime.steps",
    "app.agents.scenarios.hydrology_semantic_query.workflow.finalization",
    "app.agents.scenarios.hydrology_semantic_query.workflow.initialization",
    "app.agents.scenarios.hydrology_semantic_query.workflow.question",
    "app.agents.scenarios.hydrology_semantic_query.workflow.react",
    "app.agents.scenarios.hydrology_semantic_query.cube.client",
    "app.agents.scenarios.hydrology_semantic_query.presentation.models",
    "app.agents.scenarios.hydrology_semantic_query.presentation.output",
    "app.agents.scenarios.hydrology_semantic_query.presentation.planner",
    "app.agents.scenarios.hydrology_semantic_query.presentation.planning",
    "app.agents.scenarios.hydrology_semantic_query.presentation.profile",
    "app.agents.scenarios.hydrology_semantic_query.presentation.renderers",
    "app.agents.scenarios.hydrology_semantic_query.presentation.rendering",
    "app.agents.scenarios.hydrology_semantic_query.presentation.reporting",
    "app.agents.tools.hydrology_semantic_query.catalog.documents",
    "app.agents.tools.hydrology_semantic_query.catalog.embeddings",
    "app.agents.tools.hydrology_semantic_query.catalog.index_cache",
    "app.agents.tools.hydrology_semantic_query.catalog.retriever",
    "app.agents.tools.hydrology_semantic_query.catalog.scoring",
    "app.agents.tools.hydrology_semantic_query.catalog.tool",
    "app.agents.tools.hydrology_semantic_query.messages",
    "app.agents.tools.hydrology_semantic_query.query.pipeline",
    "app.agents.tools.hydrology_semantic_query.query.response",
    "app.agents.tools.hydrology_semantic_query.query.tool",
    "app.agents.tools.hydrology_semantic_query.query.validator",
    "app.agents.tools.hydrology_semantic_query.registry",
})


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                package = list(
                    path.relative_to(PROJECT_DIR).with_suffix("").parts[:-1]
                )
                target = package[: len(package) - node.level + 1]
                target.extend(node.module.split("."))
                imports.add(".".join(target))
            else:
                imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


def test_legacy_mixed_modules_do_not_exist() -> None:
    assert not {
        path.name
        for path in SCENARIO_DIR.glob("*.py")
        if path.name in LEGACY_SCENARIO_MODULES
    }
    assert (TOOLS_DIR / "tools.py").is_file()
    assert not list(SHARED_TOOLS_DIR.rglob("*.py"))


def test_package_uses_the_moderately_consolidated_layout() -> None:
    assert {path.name for path in SCENARIO_DIR.glob("*.py")} == {
        "__init__.py",
        "agent.py",
        "contracts.py",
        "graph.py",
        "knowledge.py",
        "node.py",
        "prompts.py",
        "state.py",
    }
    assert {
        path.name for path in (SCENARIO_DIR / "query_child").glob("*.py")
    } == {
        "__init__.py",
        "client.py",
        "config.py",
        "graph.py",
        "models.py",
        "node.py",
        "prompts.py",
        "runtime.py",
        "state.py",
        "tool_call_parser.py",
    }
    assert {
        path.name for path in (SCENARIO_DIR / "report_child").glob("*.py")
    } == {
        "__init__.py",
        "graph.py",
        "models.py",
        "node.py",
        "prompts.py",
        "report.py",
        "report_analysis.py",
        "report_rendering.py",
        "runtime.py",
        "state.py",
        "tool_call_parser.py",
    }
    assert not (SCENARIO_DIR / "presentation").exists()
    assert not (SCENARIO_DIR / "cube").exists()
    assert (SCENARIO_DIR / "semantic").is_dir()
    assert not (SCENARIO_DIR / "semantic" / "client.py").exists()
    assert {path.name for path in TOOLS_DIR.glob("*.py")} == {
        "__init__.py",
        "common.py",
        "run_semantic_query.py",
        "search_semantic_catalog.py",
        "tools.py",
    }
    assert not list((SCENARIO_DIR / "tools").glob("*.py"))
    assert not (TOOLS_DIR / "services").exists()


def test_child_agents_only_import_shared_parent_contracts() -> None:
    child_names = {"query_child", "report_child"}
    offenders: dict[str, list[str]] = {}
    for child_name in child_names:
        child_dir = SCENARIO_DIR / child_name
        for path in child_dir.rglob("*.py"):
            package = list(path.relative_to(SCENARIO_DIR).with_suffix("").parts[:-1])
            parent_imports: list[str] = []
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level == 0:
                    continue
                target = package[: len(package) - node.level + 1]
                if node.module:
                    target.extend(node.module.split("."))
                if (
                    target
                    and target[0] not in child_names
                    and target != ["contracts"]
                ):
                    parent_imports.append(".".join(target))
                if target and target[0] in child_names and target[0] != child_name:
                    parent_imports.append(".".join(target))
            if parent_imports:
                offenders[str(path.relative_to(PROJECT_DIR))] = sorted(parent_imports)
    assert offenders == {}


def test_parent_state_and_contracts_do_not_import_child_agents() -> None:
    forbidden = (
        "app.agents.scenarios.hydrology_semantic_query.query_child",
        "app.agents.scenarios.hydrology_semantic_query.report_child",
    )
    for path in (SCENARIO_DIR / "contracts.py", SCENARIO_DIR / "state.py"):
        assert not {
            imported
            for imported in _imports(path)
            if imported.startswith(forbidden)
        }


def test_repository_has_no_legacy_hydrology_imports() -> None:
    offenders: dict[str, list[str]] = {}
    for source_root in (PROJECT_DIR / "app", PROJECT_DIR / "tests"):
        for path in source_root.rglob("*.py"):
            matched = sorted(_imports(path) & LEGACY_IMPORTS)
            if matched:
                offenders[str(path.relative_to(PROJECT_DIR))] = matched
    assert offenders == {}


def test_tools_do_not_depend_on_workflow_or_graph() -> None:
    forbidden_prefixes = (
        "app.agents.scenarios.hydrology_semantic_query.graph",
        "app.agents.scenarios.hydrology_semantic_query.node",
        "app.agents.scenarios.hydrology_semantic_query.nodes",
        "app.agents.scenarios.hydrology_semantic_query.workflow",
    )
    offenders: dict[str, list[str]] = {}
    for path in TOOLS_DIR.rglob("*.py"):
        matched = sorted(
            imported
            for imported in _imports(path)
            if imported.startswith(forbidden_prefixes)
        )
        if matched:
            offenders[str(path.relative_to(PROJECT_DIR))] = matched
    assert offenders == {}


def test_stable_public_entrypoints_are_importable() -> None:
    from app.agents.scenarios.hydrology_semantic_query import (
        ExecutionPlanRevision,
        QueryTask,
        ReportSectionRequirement,
        SemanticCatalogMode,
        SemanticQuery,
        SemanticQueryResult,
        StructuredReport,
        TaskExecutionResult,
    )
    from app.agents.scenarios.hydrology_semantic_query.agent import (
        HYDROLOGY_SEMANTIC_QUERY_ID,
        hydrology_semantic_query_definition,
    )
    from app.agents.scenarios.hydrology_semantic_query.graph import (
        build_hydrology_semantic_query_graph,
    )
    from app.agents.scenarios.hydrology_semantic_query.query_child.client import CubeClient
    from app.agents.scenarios.hydrology_semantic_query.query_child.config import (
        HydrologySemanticQuerySettings,
        load_hydrology_semantic_query_settings,
        normalize_cube_url,
    )
    from app.agents.scenarios.hydrology_semantic_query.query_child.tools import (
        ALL_TOOL_NAMES,
        build_hydrology_semantic_query_tools,
    )

    assert SemanticCatalogMode
    assert ExecutionPlanRevision
    assert QueryTask
    assert ReportSectionRequirement
    assert SemanticQuery
    assert SemanticQueryResult
    assert StructuredReport
    assert TaskExecutionResult
    assert HYDROLOGY_SEMANTIC_QUERY_ID
    assert hydrology_semantic_query_definition
    assert HydrologySemanticQuerySettings
    assert load_hydrology_semantic_query_settings
    assert normalize_cube_url
    assert CubeClient
    assert build_hydrology_semantic_query_graph
    assert ALL_TOOL_NAMES
    assert build_hydrology_semantic_query_tools
