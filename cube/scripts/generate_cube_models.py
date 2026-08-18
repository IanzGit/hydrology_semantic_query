from __future__ import annotations

import argparse
import fnmatch
import keyword
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects.mysql import BIT, TINYINT
from sqlalchemy.engine import URL, Engine, make_url
from sqlalchemy.sql.sqltypes import Boolean, Date, DateTime, Float, Integer, Numeric, Time

DEFAULT_EXCLUDE_PATTERNS = (
    "*_log",
    "*_bak",
    "tmp_*",
    "sys_*",
    "*_history_backup",
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "generated"


class CubeModelGenerationError(ValueError):
    pass


def normalize_name(value: str, prefix: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]", "_", value).strip("_").lower()
    if not normalized:
        normalized = prefix
    if not normalized[0].isalpha():
        normalized = f"{prefix}_{normalized}"
    if keyword.iskeyword(normalized):
        normalized = f"{normalized}_{prefix}"
    return normalized


def to_cube_type(sql_type: Any) -> str:
    if isinstance(sql_type, Boolean):
        return "boolean"
    if isinstance(sql_type, BIT) and getattr(sql_type, "length", None) == 1:
        return "boolean"
    if isinstance(sql_type, TINYINT) and getattr(sql_type, "display_width", None) == 1:
        return "boolean"
    if isinstance(sql_type, (DateTime, Date, Time)):
        return "time"
    if isinstance(sql_type, (Integer, Numeric, Float)):
        return "number"
    return "string"


def discover_relations(
    inspector: Any,
    requested_tables: list[str] | None,
    exclude_patterns: list[str],
) -> list[tuple[str, str]]:
    relations = {name: "table" for name in inspector.get_table_names()}
    for name in inspector.get_view_names():
        relations.setdefault(name, "view")
    if requested_tables:
        requested = set(requested_tables)
        missing = sorted(requested - relations.keys())
        if missing:
            raise CubeModelGenerationError(f"数据库中不存在对象：{', '.join(missing)}")
        selected = requested
    else:
        patterns = (*DEFAULT_EXCLUDE_PATTERNS, *exclude_patterns)
        selected = {
            name
            for name in relations
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        }
    if not selected:
        raise CubeModelGenerationError("筛选后没有可生成的表或 View")
    return [(name, relations[name]) for name in sorted(selected)]


def introspect_relations(
    inspector: Any,
    relations: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    infos = []
    used_cube_names: dict[str, str] = {}
    for relation_name, relation_type in relations:
        cube_name = normalize_name(relation_name, "cube")
        if cube_name in used_cube_names:
            raise CubeModelGenerationError(
                f"Cube 名称冲突：{used_cube_names[cube_name]} 与 {relation_name} 都转换为 {cube_name}"
            )
        used_cube_names[cube_name] = relation_name
        columns = list(inspector.get_columns(relation_name))
        if not columns:
            raise CubeModelGenerationError(f"对象 {relation_name} 没有可用字段")
        dimension_names: dict[str, str] = {}
        for column in columns:
            column_name = str(column["name"])
            dimension_name = normalize_name(column_name, "field")
            if dimension_name in dimension_names:
                raise CubeModelGenerationError(
                    f"对象 {relation_name} 字段名称冲突："
                    f"{dimension_names[dimension_name]} 与 {column_name} 都转换为 {dimension_name}"
                )
            dimension_names[dimension_name] = column_name
        if relation_type == "table":
            primary_key = inspector.get_pk_constraint(relation_name) or {}
            foreign_keys = list(inspector.get_foreign_keys(relation_name) or [])
        else:
            primary_key = {}
            foreign_keys = []
        infos.append({
            "name": relation_name,
            "kind": relation_type,
            "cube_name": cube_name,
            "columns": columns,
            "column_names": {str(column["name"]): normalize_name(str(column["name"]), "field") for column in columns},
            "primary_keys": list(primary_key.get("constrained_columns") or []),
            "foreign_keys": foreign_keys,
        })
    return infos


def _skip_record(source: dict[str, Any], foreign_key: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "source_table": source["name"],
        "source_columns": list(foreign_key.get("constrained_columns") or []),
        "target_table": foreign_key.get("referred_table"),
        "target_columns": list(foreign_key.get("referred_columns") or []),
        "reason": reason,
    }


def build_joins_and_candidates(
    infos: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, Any]]]]:
    infos_by_name = {info["name"]: info for info in infos}
    infos_by_cube = {info["cube_name"]: info for info in infos}
    joins = {info["name"]: [] for info in infos}
    skipped_foreign_keys = []
    actual_fk_columns: dict[str, set[str]] = {info["name"]: set() for info in infos}
    for source in infos:
        target_counts = Counter(
            foreign_key.get("referred_table")
            for foreign_key in source["foreign_keys"]
            if len(foreign_key.get("constrained_columns") or []) == 1
            and len(foreign_key.get("referred_columns") or []) == 1
            and foreign_key.get("referred_table")
        )
        for foreign_key in source["foreign_keys"]:
            source_columns = list(foreign_key.get("constrained_columns") or [])
            target_columns = list(foreign_key.get("referred_columns") or [])
            actual_fk_columns[source["name"]].update(source_columns)
            target_name = foreign_key.get("referred_table")
            if len(source_columns) != 1 or len(target_columns) != 1:
                skipped_foreign_keys.append(_skip_record(source, foreign_key, "composite_foreign_key"))
                continue
            target = infos_by_name.get(target_name)
            if target is None:
                skipped_foreign_keys.append(_skip_record(source, foreign_key, "target_not_generated"))
                continue
            if not source["primary_keys"]:
                skipped_foreign_keys.append(_skip_record(source, foreign_key, "source_primary_key_missing"))
                continue
            if not target["primary_keys"]:
                skipped_foreign_keys.append(_skip_record(source, foreign_key, "target_primary_key_missing"))
                continue
            if source_columns[0] not in source["column_names"] or target_columns[0] not in target["column_names"]:
                skipped_foreign_keys.append(_skip_record(source, foreign_key, "column_not_found"))
                continue
            if target_counts[target_name] > 1:
                skipped_foreign_keys.append(_skip_record(source, foreign_key, "duplicate_target_cube"))
                continue
            target_dimension = target["column_names"][target_columns[0]]
            joins[source["name"]].append({
                "name": target["cube_name"],
                "relationship": "many_to_one",
                "sql": f"{{CUBE}}.{source_columns[0]} = {{{target['cube_name']}.{target_dimension}}}",
            })
    candidates = []
    for source in infos:
        for column in source["columns"]:
            column_name = str(column["name"])
            if column_name in actual_fk_columns[source["name"]]:
                continue
            if column_name.lower().endswith("_id"):
                target_stem = column_name[:-3]
            elif column_name.endswith("Id"):
                target_stem = column_name[:-2]
            else:
                continue
            target = infos_by_cube.get(normalize_name(target_stem, "cube"))
            if target is None or target is source:
                continue
            target_id = next(
                (name for name in target["column_names"] if name.lower() == "id"),
                None,
            )
            if target_id is None:
                continue
            candidates.append({
                "source_table": source["name"],
                "source_column": column_name,
                "target_table": target["name"],
                "target_column": target_id,
                "confidence": 0.9,
                "reason": "column_name_matches_target_id",
            })
    for value in joins.values():
        value.sort(key=lambda item: item["name"])
    skipped_foreign_keys.sort(
        key=lambda item: (item["source_table"], str(item["target_table"]), item["source_columns"])
    )
    candidates.sort(
        key=lambda item: (item["source_table"], item["source_column"], item["target_table"])
    )
    return joins, {
        "join_candidates": candidates,
        "skipped_foreign_keys": skipped_foreign_keys,
    }


def build_models(
    infos: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    joins, report = build_joins_and_candidates(infos)
    models = {}
    for info in infos:
        primary_keys = set(info["primary_keys"])
        dimensions = []
        for column in info["columns"]:
            column_name = str(column["name"])
            dimension = {
                "name": info["column_names"][column_name],
                "sql": column_name,
                "type": to_cube_type(column.get("type")),
            }
            if column_name in primary_keys:
                dimension["primary_key"] = True
            dimensions.append(dimension)
        cube = {
            "name": info["cube_name"],
            "sql_table": info["name"],
            "public": False,
            "measures": [{"name": "count", "type": "count"}],
            "dimensions": dimensions,
        }
        if joins[info["name"]]:
            cube["joins"] = joins[info["name"]]
        models[f"{info['cube_name']}.yml"] = {"cubes": [cube]}
    return models, report


def write_artifacts(
    output_dir: Path,
    models: dict[str, dict[str, Any]],
    report: dict[str, list[dict[str, Any]]],
) -> None:
    if output_dir.is_symlink():
        raise CubeModelGenerationError(f"输出目录不能是符号链接：{output_dir}")
    output_dir = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise CubeModelGenerationError(f"输出路径不是目录：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    backup_dir = output_dir.parent / f".{output_dir.name}-backup"
    try:
        cubes_dir = temporary_dir / "model" / "cubes"
        cubes_dir.mkdir(parents=True)
        for filename, model in sorted(models.items()):
            (cubes_dir / filename).write_text(
                yaml.safe_dump(model, allow_unicode=True, sort_keys=False, width=1000),
                encoding="utf-8",
            )
        (temporary_dir / "join_candidates.yml").write_text(
            yaml.safe_dump(report, allow_unicode=True, sort_keys=False, width=1000),
            encoding="utf-8",
        )
        if backup_dir.exists():
            raise CubeModelGenerationError(f"备份目录已存在：{backup_dir}")
        if output_dir.exists():
            output_dir.rename(backup_dir)
        try:
            temporary_dir.rename(output_dir)
        except Exception:
            if backup_dir.exists() and not output_dir.exists():
                backup_dir.rename(output_dir)
            raise
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def database_url_from_environment() -> URL:
    def first(*names: str) -> str | None:
        return next((os.environ[name] for name in names if name in os.environ), None)

    host = first("CUBEJS_DB_HOST", "MYSQL_HOST")
    database = first("CUBEJS_DB_NAME", "MYSQL_DATABASE")
    username = first("CUBEJS_DB_USER", "MYSQL_USER")
    password = first("CUBEJS_DB_PASS", "MYSQL_PASSWORD") or ""
    port_value = first("CUBEJS_DB_PORT", "MYSQL_PORT") or "3306"
    missing = [
        label
        for label, value in (("HOST", host), ("DATABASE", database), ("USER", username))
        if not value
    ]
    if missing:
        raise CubeModelGenerationError(
            "缺少数据库连接配置：" + ", ".join(missing)
        )
    try:
        port = int(port_value)
    except ValueError as exc:
        raise CubeModelGenerationError(f"数据库端口无效：{port_value}") from exc
    return URL.create(
        "mysql+pymysql",
        username=username,
        password=password,
        host=host,
        port=port,
        database=database,
        query={"charset": "utf8mb4"},
    )


def create_database_engine(database_url: str | None) -> Engine:
    url = make_url(database_url) if database_url else database_url_from_environment()
    if url.drivername == "mysql":
        url = url.set(drivername="mysql+pymysql")
    if url.get_backend_name() not in {"mysql", "mariadb"}:
        raise CubeModelGenerationError("仅支持 MySQL 或 MariaDB 数据库")
    if url.get_driver_name() in {"asyncmy", "aiomysql"}:
        raise CubeModelGenerationError("生成脚本需要同步 MySQL 驱动，请使用 mysql+pymysql")
    return create_engine(url, pool_pre_ping=True)


def generate(
    engine: Engine,
    requested_tables: list[str] | None,
    exclude_patterns: list[str],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[int, int, int]:
    try:
        database_inspector = inspect(engine)
        relations = discover_relations(database_inspector, requested_tables, exclude_patterns)
        infos = introspect_relations(database_inspector, relations)
        models, report = build_models(infos)
        write_artifacts(output_dir, models, report)
        return len(models), len(report["join_candidates"]), len(report["skipped_foreign_keys"])
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 MySQL 生成待审核的水文 Cube 基础模型")
    parser.add_argument("--database-url")
    parser.add_argument("--table", action="append", dest="tables")
    parser.add_argument("--exclude-pattern", action="append", default=[])
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    engine = None
    try:
        engine = create_database_engine(arguments.database_url)
        model_count, candidate_count, skipped_count = generate(
            engine,
            arguments.tables,
            arguments.exclude_pattern,
        )
    except CubeModelGenerationError as exc:
        if engine is not None:
            engine.dispose()
        print(f"生成失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        if engine is not None:
            engine.dispose()
        print(f"生成失败：{type(exc).__name__}", file=sys.stderr)
        return 1
    print(
        f"生成完成：{model_count} 个 Cube 草稿，{candidate_count} 个关联候选，"
        f"{skipped_count} 个未入模外键。输出目录：{DEFAULT_OUTPUT_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
