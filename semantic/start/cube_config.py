from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv.main import dotenv_values

_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_SENSITIVE_KEY_PATTERN = re.compile(r"(?i)(PASS|PASSWORD|TOKEN|SECRET|API_KEY|APIKEY)")
_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(\b(?:PASS|PASSWORD|TOKEN|SECRET|API_KEY|APIKEY)\b\s*[:=]\s*)([^\s,;]+)"
)
_ASSIGNMENT_KEY_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
_SCENARIO_DIR = Path(__file__).resolve().parents[2]


class CubeConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class CubeStartConfig:
    model_name: str
    model_path: Path
    cube_host: str
    cube_port: int
    dev_mode: str
    database_type: str
    database_host: str
    database_name: str
    sensitive_values: tuple[str, ...]

    def shell_lines(self) -> tuple[str, ...]:
        return (
            self.model_name,
            str(self.model_path),
            self.cube_host,
            str(self.cube_port),
            self.dev_mode,
            self.database_type,
            self.database_host,
            self.database_name,
        )


def _load_values(path: Path, label: str) -> dict[str, str]:
    if not path.is_file():
        raise CubeConfigError(f"缺少{label}：{path}")
    seen: set[str] = set()
    duplicate_keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ASSIGNMENT_KEY_PATTERN.match(line)
        if match is None:
            continue
        key = match.group(1)
        if key in seen and key not in duplicate_keys:
            duplicate_keys.append(key)
        seen.add(key)
    if duplicate_keys:
        names = "、".join(duplicate_keys)
        raise CubeConfigError(f"Cube配置中的 {names} 重复，请确保只启用一套模型和数据库配置")
    raw_values = dotenv_values(path)
    return {key: value.strip() for key, value in raw_values.items() if value is not None}


def _required(values: dict[str, str], key: str, label: str) -> str:
    value = values.get(key, "")
    if not value:
        raise CubeConfigError(f"{label}缺少必填项 {key}")
    if any(character in value for character in ("\r", "\n", "\0")):
        raise CubeConfigError(f"{label}中的 {key} 包含非法控制字符")
    return value


def _port(value: str, key: str, label: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise CubeConfigError(f"{label}中的 {key} 必须是有效端口") from exc
    if not 0 < port < 65536:
        raise CubeConfigError(f"{label}中的 {key} 必须在1到65535之间")
    return port


def _name(value: str, key: str, label: str) -> str:
    if _NAME_PATTERN.fullmatch(value) is None:
        raise CubeConfigError(f"{label}中的 {key} 只能包含字母、数字、下划线和连字符")
    return value


def load_cube_config(
    env_path: Path,
    *,
    models_root: Path | None = None,
) -> CubeStartConfig:
    env_path = env_path.resolve()
    models_root = (models_root or _SCENARIO_DIR / "semantic" / "model").resolve()
    values = _load_values(env_path, "Cube配置")
    cube_host = _required(values, "CUBE_HOST", "Cube配置")
    cube_port = _port(
        _required(values, "CUBE_PORT", "Cube配置"),
        "CUBE_PORT",
        "Cube配置",
    )
    dev_mode = _required(values, "CUBEJS_DEV_MODE", "Cube配置").lower()
    if dev_mode not in {"true", "false"}:
        raise CubeConfigError("Cube配置中的 CUBEJS_DEV_MODE 只能是true或false")
    model_name = _name(
        _required(values, "MODEL_NAME", "Cube配置"),
        "MODEL_NAME",
        "Cube配置",
    )
    database_type = _required(values, "CUBEJS_DB_TYPE", "Cube配置")
    database_host = _required(values, "CUBEJS_DB_HOST", "Cube配置")
    _port(
        _required(values, "CUBEJS_DB_PORT", "Cube配置"),
        "CUBEJS_DB_PORT",
        "Cube配置",
    )
    database_name = _required(values, "CUBEJS_DB_NAME", "Cube配置")
    _required(values, "CUBEJS_DB_USER", "Cube配置")
    _required(values, "CUBEJS_DB_PASS", "Cube配置")
    model_path = models_root / model_name
    for directory_name in ("cubes", "views"):
        directory = model_path / directory_name
        if not directory.is_dir() or not any(directory.glob("*.yml")):
            raise CubeConfigError(f"模型目录缺少有效的 {directory_name}：{directory}")
    sensitive_values = tuple(
        dict.fromkeys(
            value for key, value in values.items() if value and _SENSITIVE_KEY_PATTERN.search(key)
        )
    )
    return CubeStartConfig(
        model_name=model_name,
        model_path=model_path.resolve(),
        cube_host=cube_host,
        cube_port=cube_port,
        dev_mode=dev_mode,
        database_type=database_type,
        database_host=database_host,
        database_name=database_name,
        sensitive_values=sensitive_values,
    )


def redact_text(value: str, sensitive_values: tuple[str, ...]) -> str:
    redacted = value
    for sensitive_value in sensitive_values:
        redacted = redacted.replace(sensitive_value, "***")
    return _ASSIGNMENT_PATTERN.sub(r"\1***", redacted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="解析并校验Cube启动配置")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--redact-stdin", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_cube_config(arguments.env_file)
    except CubeConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if arguments.redact_stdin:
        print(redact_text(sys.stdin.read(), config.sensitive_values), end="")
        return 0
    print("\n".join(config.shell_lines()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
